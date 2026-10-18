from typing import Tuple

import numpy as np
import torch
from rav.third_party.facenet_pytorch import MTCNN
from torch import nn
from torchvision import transforms

from rav.face.arcface import Backbone
# from face_replace.training.criteria.utils import extract_faces_and_landmarks
from einops import rearrange
from kornia.geometry.transform.crop2d import warp_affine

from rav.face.matlab_cp2tform import get_similarity_transform_for_cv2
import einops 
from diffusers.models.attention_processor import Attention, F
from diffusers import FluxTransformer2DModel

from rav.face.adaface import load_pretrained_model_adaface, l2_norm

REFERNCE_FACIAL_POINTS_RELATIVE = np.array([[38.29459953, 51.69630051],
                                            [72.53179932, 51.50139999],
                                            [56.02519989, 71.73660278],
                                            [41.54930115, 92.3655014],
                                            [70.72990036, 92.20410156]
                                            ]) / 112  # Original points are 112 * 96 added 8 to the x axis to make it 112 * 112


def verify_load(missing_keys, unexpected_keys):
    if unexpected_keys:
        RuntimeError(f"Found unexpected keys in state dict while loading the encoder:\n{unexpected_keys}")

    filtered_missing = [key for key in missing_keys if not "extract_kv1" in key]
    if filtered_missing:
        RuntimeError(f"Missing keys in state dict while loading the encoder:\n{filtered_missing}")


def detect_face(images: torch.Tensor, mtcnn: torch.nn.Module) -> torch.Tensor:
    """ Detect faces in the images using MTCNN. If no face is detected, use the whole image. """
    with torch.no_grad():
        images = rearrange(images, "b c h w -> b h w c")
        if images.dtype != torch.uint8:
            images = ((images * 0.5 + 0.5) * 255).type(torch.uint8)  # Unnormalize
        _, _, landmarks = mtcnn(images, landmarks=True)
        return landmarks

def extract_faces_and_landmarks(images: torch.Tensor,
                                output_size: int = 112,
                                mtcnn: torch.nn.Module = None,
                                reference_points=REFERNCE_FACIAL_POINTS_RELATIVE):
    """
    detect faces in the images and crop them (in a differentiable way) to 112x112 using MTCNN.
    """
    landmarks_batched = detect_face(images, mtcnn=mtcnn)
    affine_transformations = []
    invalid_indices = []
    for i, landmarks in enumerate(landmarks_batched):
        if landmarks is None:
            invalid_indices.append(i)
            affine_transformations.append(np.eye(2, 3).astype(np.float32))
        else:
            affine_transformations.append(
                get_similarity_transform_for_cv2(
                    landmarks[0].astype(np.float32), reference_points.astype(np.float32) * output_size
                )
            )
    affine_transformations = torch.from_numpy(
        np.stack(affine_transformations).astype(np.float32)
    ).to(device=images.device, dtype=torch.float32)

    invalid_indices = torch.tensor(invalid_indices).to(device=images.device)

    fp_images = images.to(torch.float32)
    return warp_affine(fp_images, affine_transformations,
                       dsize=(output_size, output_size)).to(dtype=images.dtype), invalid_indices


class Arcface_extractor(nn.Module):
    """
    Use pretrained facenet model to extract features from the face of the predicted image and target image.
    Facenet expects 112x112 images, so we crop the face using MTCNN and resize it to 112x112.
    Then we use the cosine similarity between the features to calculate the loss. (The cosine similarity is 1 - cosine distance).
    Also notice that the outputs of facenet are normalized so the dot product is the same as cosine distance.
    """
    def __init__(self, pretrained_arcface_path:str='./checkpoints/arcface/model_ir_se50.pth', device='cuda', dtype=torch.float32, 
                 using_curricular=False, using_adaface=False, output_penultimate_layer=False, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.dtype = dtype
        self.mtcnn = MTCNN(device=self.device)
        self.mtcnn.forward = self.mtcnn.detect
        self.mtcnn.requires_grad_(False)
        self.mtcnn.eval()
        self.facenet_input_size = 112  # Has to be 112, can't find weights for 224 size.
        self.output_penultimate_layer = output_penultimate_layer

        self.apply_l2_norm = False
        self.to_BGR = False

        if using_curricular:
            self.facenet = Backbone(112, num_layers=100, mode='ir', drop_ratio=0.4, affine=False)
            self.facenet.load_state_dict(torch.load(pretrained_arcface_path))
        elif using_adaface:
            self.facenet = load_pretrained_model_adaface(architecture='ir_50')
            self.to_BGR = True

        else:
            self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se', output_penultimate_layer=self.output_penultimate_layer)
            self.facenet.load_state_dict(torch.load(pretrained_arcface_path))
        
        self.face_pool = torch.nn.AdaptiveAvgPool2d((self.facenet_input_size, self.facenet_input_size))
        self.facenet.requires_grad_(False)
        self.facenet.eval()
        self.facenet.to(device=self.device, dtype=torch.float32)  # not implemented for half precision
        self.face_pool.to(device=self.device, dtype=torch.float32)  # not implemented for half precision
        self.visualization_resize = transforms.Resize((self.facenet_input_size, self.facenet_input_size),
                                                      interpolation=transforms.InterpolationMode.BICUBIC)
        self.reference_facial_points = np.array([[38.29459953, 51.69630051],
                                                 [72.53179932, 51.50139999],
                                                 [56.02519989, 71.73660278],
                                                 [41.54930115, 92.3655014],
                                                 [70.72990036, 92.20410156]
                                                 ])  # Original points are 112 * 96 added 8 to the x axis to make it 112 * 112
        

    def extract_feats(self, x: torch.Tensor):
        """ Extract features from the face of the image using facenet model. """
        x = self.face_pool(x)
        if self.to_BGR:
            x = torch.flip(x, dims=[1])
        x_feats = self.facenet(x)
        if self.apply_l2_norm:
            x_feats = l2_norm(x_feats)
        return x_feats

    def _get_arcface_embed(self, x):
        '''
        input tensor `x`: normalize to -1~1
        '''
        x = x.to(dtype=torch.float32)
        B, _, _, _ = x.size()

        with torch.no_grad():
            encoder_pixel_values_face, source_invalid_indices = extract_faces_and_landmarks(x, mtcnn=self.mtcnn)
            source_invalid_indices = source_invalid_indices.long()

        valid_indices = []  
        for i in range(x.shape[0]):
            if i not in source_invalid_indices:
                valid_indices.append(i)
        valid_indices = torch.tensor(valid_indices).to(device=x.device)

        if self.output_penultimate_layer:
            arcface_emb = torch.zeros((B, 512, 7, 7)).to(dtype=self.dtype, device=self.device)
        else:
            arcface_emb = torch.zeros((B, 1, 512)).to(dtype=self.dtype, device=self.device)

        if valid_indices.shape[0] != 0:
            with torch.no_grad():
                # arcface_emb[valid_indices] = self.extract_feats(encoder_pixel_values_face[valid_indices]) # [B, 512] or  [B, 512, 7, 7]
                arcface_emb = self.extract_feats(encoder_pixel_values_face).float() # [B, 512] or  [B, 512, 7, 7]

                if self.output_penultimate_layer:
                    # assign invalid idx (no detected face) with zero tensors
                    arcface_emb[source_invalid_indices] = torch.zeros((len(source_invalid_indices), 512, 7, 7)).to(dtype=self.dtype, device=self.device)
                else:
                    # assign invalid idx (no detected face) with zero tensors.
                    # arcface_emb here is [B, 512] (2D); the RHS must match that
                    # rank — [N, 512], NOT [N, 1, 512]. The trailing L=1 axis is
                    # added back by the einops.rearrange below.
                    arcface_emb[source_invalid_indices.tolist()] = torch.zeros((len(source_invalid_indices.tolist()), 512)).to(dtype=self.dtype, device=self.device)
                    arcface_emb = einops.rearrange(arcface_emb, '(B L) C -> B L C', L=1).to(dtype=self.dtype, device=self.device)
            
        return arcface_emb


    @torch._dynamo.disable
    def forward(self, x: torch.Tensor):
        arcface_emb = self._get_arcface_embed(x) # [B, 1, 512]
        return arcface_emb    

    

if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms

    identity_module = Arcface_extractor(pretrained_arcface_path='./checkpoints/arcface/model_ir_se50.pth', device='cuda')

    identity_module.mtcnn.requires_grad_(False)
    identity_module.facenet.requires_grad_(False)
    identity_module.mtcnn.eval()
    identity_module.facenet.eval()

    # img_path = "./dataset/train/images1024x1024/69000/69999.png"
    # sharp_img = Image.open(img_path).convert("RGB")
    # sharp_img = sharp_img.resize((1024, 1024))

    # norm_for_idloss = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # to_tensor = transforms.ToTensor()
    # face_refs = norm_for_idloss(to_tensor(sharp_img))[None]

    face_refs = torch.randn((1,3,512,512))
    face_refs = face_refs.cuda()

    identity_features = identity_module.forward(face_refs) 
    print(identity_features.shape) # [B, 1, 512]