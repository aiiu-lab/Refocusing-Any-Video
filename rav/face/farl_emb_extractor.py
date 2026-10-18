import torch
import clip
from PIL import Image
import torch.nn as nn
import numpy as np
from kornia.geometry.transform import warp_affine
from einops import rearrange
import torchvision.transforms.functional as F
import torchvision
from rav.third_party.facenet_pytorch import MTCNN


def detect_face(images: torch.Tensor, mtcnn: torch.nn.Module):
    """ Return both boxes and landmarks. """
    B,_,_,_ = images.shape
    with torch.no_grad():
        # MTCNN expects HWC for detection
        images_hwc = rearrange(images, "b c h w -> b h w c")
        if images_hwc.dtype != torch.uint8:
            # Unnormalize from [-1, 1] to [0, 255]
            images_hwc = ((images_hwc * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
        
        # boxes shape: [Batch, Num_Faces, 4] -> [x1, y1, x2, y2]
        boxes, _, landmarks = mtcnn(images_hwc, landmarks=True)
        if boxes is None:
            return [None] * B, [None] * B
        return boxes, landmarks


def extract_faces_loose(images: torch.Tensor,
                        output_size: int = 224, # FaRL standard is 224
                        mtcnn: torch.nn.Module = None,
                        expand_ratio: float = 0.5): # 50% expansion for "loose" context
    """
    Detect faces and crop them loosely (maintaining aspect ratio, no warping)
    to output_size x output_size in a differentiable way using Kornia.

    input `images`: [B,3,H,W], normalize to -1~1
    """
    # 1. Detect landmarks/boxes
    # Assuming detect_face returns [Batch, Num_Faces, 5, 2] landmarks
    boxes_batched, landmarks_batched = detect_face(images, mtcnn=mtcnn)
    
    affine_transformations = []
    invalid_indices = []
    B, C, H, W = images.shape

    for i in range(B):
        box = boxes_batched[i]
        
        # Check if a face was actually found
        if box is None or len(box) == 0:
            invalid_indices.append(i)
            affine_transformations.append(np.eye(2, 3).astype(np.float32))
        else:
            # Use the first detected face [x1, y1, x2, y2]
            x1, y1, x2, y2 = box[0].astype(np.float32)
            
            # 2. Determine Center and Size from the Detector Box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = x2 - x1, y2 - y1
            
            # 3. Calculate "Loose" Side Length
            # For FaRL, even at 0.0 expansion, the detector box is usually 
            # much better than the landmark box.
            side = max(bw, bh) * (1.0 + expand_ratio)
            
            # 4. Construct Matrix
            s = output_size / side
            tx = (output_size / 2.0) - (s * cx)
            ty = (output_size / 2.0) - (s * cy)
            
            matrix = np.array([[s, 0, tx], [0, s, ty]], dtype=np.float32)
            affine_transformations.append(matrix)

    # Convert to Tensor for Kornia
    affine_transformations = torch.from_numpy(
        np.stack(affine_transformations)
    ).to(device=images.device, dtype=torch.float32)

    invalid_indices = torch.tensor(invalid_indices).to(device=images.device)
    
    # 5. Differentiable Crop & Resize
    # Using warp_affine ensures the process is backprop-friendly
    cropped_faces = warp_affine(
        images.to(torch.float32), 
        affine_transformations,
        dsize=(output_size, output_size)
    ).to(dtype=images.dtype)

    return cropped_faces, invalid_indices



class FaRLFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path="./checkpoints/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth", 
                 device="cuda", crop_expand_ratio=0.3):
        super().__init__()
        self.device = device
        # 1. Load the standard ViT-B/16 structure
        self.model, self.preprocess = clip.load("ViT-B/16", device=device)
        self.crop_expand_ratio = crop_expand_ratio
        self.farl_input_size = 224
        
        # 2. Load FaRL pre-trained weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle cases where the state_dict is nested (standard for FaRL checkpoints)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Load weights into the visual backbone
        # We only need the visual part, but loading into the CLIP model is easier
        self.model.load_state_dict(state_dict, strict=False)
        # self.model = self.model.visual
        self.model.requires_grad_(False)
        self.model.eval()

        self.mtcnn = MTCNN(device=self.device)
        self.mtcnn.forward = self.mtcnn.detect
        self.mtcnn.requires_grad_(False)
        self.mtcnn.eval()

        self.norm_clip = torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                                          std=(0.26862954, 0.26130258, 0.27577711))
        


    def load_image(self, image_path):
        """
        output tensor [B,3,H,W], range -1~1, RGB
        """
        image = Image.open(image_path)
        image = F.to_tensor(image)
        image = (image - 0.5) / 0.5
        image = image.unsqueeze(0)
        return image

    def preprocess_clip(self, img):
        """
        input `img`: [B,3,H,W], -1~1
        """
        img = (img + 1.0) / 2.0
        img = self.norm_clip(img)
        return img
        

    
    @torch._dynamo.disable
    def forward(self, image=None, image_path=None):
        """
        input `image`: [B,3,H,W], range -1~1, RGB

        output: [B, 197, 768], dtype fp16
        """
        if image is None:
            image = self.load_image(image_path).to(self.device)
        B,_,_,_ = image.shape

        with torch.no_grad():
            cropped_faces, invalid_indices =  extract_faces_loose(image, output_size=self.farl_input_size, mtcnn=self.mtcnn, expand_ratio=self.crop_expand_ratio)
            # [B, 3, 224, 224], -1 ~ 1
            # return cropped_faces # for debug

        valid_indices = []
        for i in range(B):
            if i not in invalid_indices:
                valid_indices.append(i)
        valid_indices = torch.tensor(valid_indices).to(device=cropped_faces.device)

        final_cropped_faces = torch.ones(B, 3, self.farl_input_size, self.farl_input_size) * (-1)
        final_cropped_faces = final_cropped_faces.to(self.device)

        if valid_indices.shape[0] != 0:
            final_cropped_faces[valid_indices] = cropped_faces[valid_indices]

        final_cropped_faces = self.preprocess_clip(final_cropped_faces)
        # [B, 3, 224, 224]

        with torch.no_grad():
            visual = self.model.visual
            final_cropped_faces = final_cropped_faces.to(visual.conv1.weight.dtype)

            # Patch Embedding
            x = visual.conv1(final_cropped_faces)  # Shape: [B, 768, 14, 14]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # Shape: [B, 768, 196]
            x = x.permute(0, 2, 1)  # Shape: [B, 196, 768]
            
            # Add CLS token
            cls_token = visual.class_embedding.to(x.dtype).view(1, 1, -1).expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)  # Shape: [B, 197, 768]
            
            # Add Positional Embedding
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)
            
            # Transformer Layers (The core facial representation learning)
            x = x.permute(1, 0, 2)  # NLD -> LND for CLIP's transformer
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD: [B, 197, 768]
            
            # Final layer norm is usually applied before using features
            x = visual.ln_post(x)
        
        return x


if __name__ == "__main__":
    # Usage
    input_image = "./data/examples/reference.jpg"
    farl_ckpt = "./checkpoints/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth"

    extractor = FaRLFeatureExtractor(farl_ckpt)
    features = extractor.forward(input_image)

    print(features.min(), features.max())
    print(f"Feature shape: {features.shape}") # Expected: torch.Size([1, 197, 768])

    # face = (features + 1.0) / 2.0
    # face = face[0].permute(1,2,0).detach().cpu().numpy()
    # print(face.shape)
    # print(face.min(), face.max())

    # import matplotlib.pyplot as plt
    # plt.imshow(face)
    # plt.show()