import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from dataclasses import dataclass
import math

# Importing the base model you provided
from rav.models.svd_unet_new import UNetSpatioTemporalConditionModel, UNetSpatioTemporalConditionOutput
from rav.face.arcface_emb_extractor import Arcface_extractor
from rav.face.farl_emb_extractor import FaRLFeatureExtractor

from diffusers.models.attention_processor import AttnProcessor2_0


def build_2d_sincos_pos_embed(grid_size, embed_dim):
    # grid_size: int (e.g. 16 for a 16x16 grid)
    h = w = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')  # [H, W]

    # Split embed_dim between H and W encodings
    assert embed_dim % 4 == 0
    dim_each = embed_dim // 4
    omega = 1.0 / (10000 ** (torch.arange(dim_each) / dim_each))  # [dim_each]

    # Outer product: [H*W, dim_each]
    enc_h = torch.outer(grid_h.flatten(), omega)
    enc_w = torch.outer(grid_w.flatten(), omega)

    pos_embed = torch.cat([
        enc_h.sin(), enc_h.cos(),
        enc_w.sin(), enc_w.cos()
    ], dim=-1)  # [H*W, embed_dim]

    return pos_embed  # [256, 1024]




class ImageProjModel3(nn.Module):
    """Projects ArcFace embeddings to the UNet cross-attention dimension."""

    """
    this implementation follows UniPortrait, using penultimate layer of ArcFace instead of final layer.
    difference with original version (see above): 
    - 1 linear layer -> 2 linear layer, with an extra GELU non-linearity in between.
    """
    def __init__(self, cross_attention_dim=1024, input_embeddings_dim=512, add_pos_emb=True):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.add_pos_emb = add_pos_emb
        if self.add_pos_emb:
            pos_embed = build_2d_sincos_pos_embed(7, input_embeddings_dim)  # [49, 512]
            self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # [1, 49, 512]

        self.norm_in = nn.LayerNorm(input_embeddings_dim)
        self.id_embedding_mapping = nn.Sequential(
            nn.Linear(input_embeddings_dim, cross_attention_dim),
            nn.GELU(),
            nn.Linear(cross_attention_dim, cross_attention_dim),
        )
        self.norm_out = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        # id_embeds: [B, 512, 7, 7]
        B, C, H, W = id_embeds.shape
        id_embeds = id_embeds.permute(0, 2, 3, 1).reshape(B, H * W, C)    # [B*T, 49, 512]
        if self.add_pos_emb:
            id_embeds = id_embeds + self.pos_embed
        id_embeds = self.norm_in(id_embeds)
        x = self.id_embedding_mapping(id_embeds) # [B, 49, 1024]
        x = self.norm_out(x)
        return x
    
class ImageProjModel3_Farl(nn.Module):
    """Projects ArcFace embeddings to the UNet cross-attention dimension."""

    """
    this implementation follows UniPortrait, using penultimate layer of ArcFace instead of final layer.
    difference with original version (see above): 
    - 1 linear layer -> 2 linear layer, with an extra GELU non-linearity in between.
    """
    def __init__(self, cross_attention_dim=1024, input_embeddings_dim=768):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim

        self.norm_in = nn.LayerNorm(input_embeddings_dim)
        self.repr_embedding_mapping = nn.Sequential(
            nn.Linear(input_embeddings_dim, cross_attention_dim),
            nn.GELU(),
            nn.Linear(cross_attention_dim, cross_attention_dim),
        )
        self.norm_out = nn.LayerNorm(cross_attention_dim)

    def forward(self, repr_embeds):
        repr_embeds = self.norm_in(repr_embeds)
        x = self.repr_embedding_mapping(repr_embeds) # [B, 49, 1024]
        x = self.norm_out(x)
        return x


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x

class IDAttention(nn.Module):
    def __init__(self, *, dim=1024, dim_head=64, heads=16):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim * 4, dim, bias=False),
        )

    def forward(self, x, attention_mask=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            attention_mask (torch.Tensor): attention mask
                shape (b, n1, 1)
        """
        b, l, _ = x.shape
        x = self.norm1(x)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        attn_out = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v,
        )
        attn_out = attn_out.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)

        x = x + attn_out
        x = x + self.ffn(x)
        return x


class KeypointEncoder(nn.Module):
    """
    Encodes a Gaussian-blurred 5-keypoint heatmap into spatial tokens for
    fine-grained spatial anchoring of IP-Adapter identity features.

    Input:  [B*T, 3, H, W]  float32  in [0, 1]
    Output: [B*T, 256, 1024]  — 256 spatial tokens (16×16 grid), each 1024-dim.

    For a 512×512 input the CNN downsamples to 64×64 (/8),
    then AdaptiveAvgPool2d compresses to 16×16 = 256 tokens.
    """
    def __init__(self, out_dim: int = 1024, add_pos_emb=True):
        super().__init__()
        self.add_pos_emb = add_pos_emb

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),    # /2  → 256
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),    
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),   # /4  → 128
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),   
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # /8  → 64
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  
            nn.SiLU(),
            nn.Conv2d(256, out_dim, 3, stride=1, padding=1), 
        )
        # Compress spatial extent to a fixed 16×16 grid → 256 tokens
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        if self.add_pos_emb:
            pos_embed = build_2d_sincos_pos_embed(16, out_dim)  # [256, 1024]
            self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # [1, 256, 1024]

    def forward(self, keypoint_image: torch.Tensor) -> torch.Tensor:
        # keypoint_image: [B*T, 3+3, H, W] in [0, 1]
        feat = self.encoder(keypoint_image)                      # [B*T, out_dim, H/8, W/8]
        feat = self.pool(feat)                                   # [B*T, out_dim, 16, 16]
        B, C, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(B, h * w, C)    # [B*T, 256, out_dim]
        if self.add_pos_emb:
            feat = feat + self.pos_embed
        feat = self.proj(feat)                                   # [B*T, 256, out_dim]
        feat = self.norm(feat)
        return feat                                              # [B*T, 256, 1024]


class IPMaskedSpatioTemporalAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=1024, scale=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.base_processor = AttnProcessor2_0()  # ← delegates standard path

        # IP-Adapter specific weights (Trainable)
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        
        # State variables for injection
        self.ip_adapter_image_embeds = None
        self.ip_adapter_mask = None
        
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs
    ):
        # Save input before base processor runs (needed for IP query)
        input_hidden_states = hidden_states

        # 1. Standard path — fully delegated, bit-identical to original SVD
        hidden_states = self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )

        # 2. IP-Adapter path — only runs when embeddings are injected
        if self.ip_adapter_image_embeds is not None:
            # print("doing IP-adapter stuff!!")
            batch_size, sequence_length, _ = input_hidden_states.shape

            query = attn.to_q(input_hidden_states)
            query = attn.head_to_batch_dim(query)

            ip_key   = self.to_k_ip(self.ip_adapter_image_embeds)
            ip_value = self.to_v_ip(self.ip_adapter_image_embeds)
            ip_key   = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
            ip_hidden_states   = torch.bmm(ip_attention_probs, ip_value)
            ip_hidden_states   = attn.batch_to_head_dim(ip_hidden_states)
            ip_hidden_states = attn.to_out[0](ip_hidden_states)   # Linear projection
            ip_hidden_states = attn.to_out[1](ip_hidden_states)   # Dropout (usually identity at inference)

            hidden_states = hidden_states + self.scale * ip_hidden_states

        return hidden_states

class Track4GenRefiner(nn.Module):
    """
    Track4Gen Refiner Module (Rϕ).

    Inserted at the upsampler output of SVD's 3rd decoder block (up_blocks[2], 640-ch).
    Architecture per the paper:
      • Layers 1–7 : Conv2d(640→640, k=3, s=1, p=1) → BatchNorm2d → ReLU
      • Layer  8   : Conv2d(640→640, k=3, s=1, p=1) → ReLU
      • Zero-conv  : Conv2d(640→640, k=1), weights & bias init to 0
      • Residual   : hidden_states = hidden_states + zero_conv(refiner_output)

    Initialised as the identity mapping: because the zero-conv outputs all-zeros
    at the start of training, the residual contribution is zero and the UNet
    hidden states pass through unchanged.
    """

    CHANNELS = 640  # fixed to up_blocks[2] output channels in SVD

    def __init__(self, channels: int = CHANNELS, num_layers: int = 8):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            if i < num_layers - 1:
                # Layers 1–7: Conv → BN → ReLU
                layers.append(nn.BatchNorm2d(channels))
            # Layer 8: Conv → ReLU (no BN, as per the paper)
            layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*layers)

        # Zero-conv — initialised to zeros so the refiner = identity at init
        self.zero_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, H, W]  — shape is preserved throughout
        # h̃ = R_ϕ(h)  — the refined features BEFORE zero_conv.
        # • Exposed via self.last_refined_features for the correspondence loss (L_corr).
        # • Detached before entering zero_conv so that the diffusion loss (L_diff)
        #   backprop does NOT flow back through zero_conv into the refiner's conv layers.
        #   Only L_corr gradients train the refiner — this matches the ✂️ in Figure 2.
        refined = self.conv_layers(x)                      # h̃^{1:N}  [B*T, C, H, W]
        self.last_refined_features = refined               # expose for L_corr in training loop
        refined = refined.detach() # ✂️ detach: L_diff stops here
        residual = self.zero_conv(refined)        
        return x + residual                                # identity at init; learns a refinement


class SVDIPAdapterWrapper(nn.Module):
    def __init__(self, unet: UNetSpatioTemporalConditionModel, 
                 arcface_extractor: nn.Module = None,
                 farl_extractor: nn.Module = None):
        super().__init__()
        self.unet = unet
        if arcface_extractor is not None:
            self.arcface = arcface_extractor # Frozen
        else:
            self.arcface = Arcface_extractor(pretrained_arcface_path='./checkpoints/arcface/model_ir_se50.pth', device='cuda',
                                             output_penultimate_layer=True)
            
        if farl_extractor is not None:
            self.farl = farl_extractor # Frozen
        else:
            self.farl = FaRLFeatureExtractor(checkpoint_path="./checkpoints/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth", 
                                            device="cuda", crop_expand_ratio=0.3)
        
        self.arcface.requires_grad_(False)
        self.arcface.mtcnn.eval()
        self.arcface.facenet.eval()

        self.image_proj1 = ImageProjModel3(
            cross_attention_dim=1024, 
            input_embeddings_dim=512, 
            add_pos_emb=True,
        )

        self.image_proj2 = ImageProjModel3_Farl(
            cross_attention_dim=1024, 
            input_embeddings_dim=768, 
        )

        # Keypoint Encoder (Trainable) — encodes 5-landmark heatmap into 256 spatial tokens
        self.keypoint_encoder = KeypointEncoder(out_dim=1024)

        # self.id_attn = IDAttention(dim=1024, dim_head=64, heads=16)
        self.id_attn = nn.Sequential(IDAttention(dim=1024, dim_head=64, heads=16),
                                     IDAttention(dim=1024, dim_head=64, heads=16),
                                     IDAttention(dim=1024, dim_head=64, heads=16),
                                     IDAttention(dim=1024, dim_head=64, heads=16))
        
        self.set_ip_adapter_processors()

        # --- Track4Gen Refiner ---
        # Trainable module hooked onto the upsampler of up_blocks[2] (640-ch).
        # The hook fires automatically during self.unet(...) — no changes to the
        # UNet forward pass are required.
        self.refiner = Track4GenRefiner(channels=Track4GenRefiner.CHANNELS)
        self._refiner_hook_handle = None
        self._register_refiner_hook()

    def _register_refiner_hook(self):
        """
        Registers a forward hook on self.unet.up_blocks[2].upsamplers[0].

        The hook intercepts the upsampler output and passes it through the
        Track4Gen refiner, replacing the output with x + zero_conv(refiner(x)).

        Call this again after loading a checkpoint, since hooks are not saved
        in state_dict / pickle:
            model.load_state_dict(ckpt)
            model._register_refiner_hook()
        """
        # Guard: verify the target module exists before registering
        upsampler = self.unet.up_blocks[2].upsamplers[0]

        refiner = self.refiner  # capture in closure — avoids `self` reference issues

        def _hook(module: nn.Module, input: Tuple, output: torch.Tensor) -> torch.Tensor:
            return refiner(output)

        # Remove any previously registered hook (safe to call on a fresh handle=None)
        if self._refiner_hook_handle is not None:
            self._refiner_hook_handle.remove()

        self._refiner_hook_handle = upsampler.register_forward_hook(_hook)

    def set_ip_adapter_processors(self):
        """Injects the custom processors into the UNet."""
        attn_procs = {}
        for name, _ in self.unet.attn_processors.items():
            # Only apply to Spatial Cross-Attention layers
            # In SVD, these usually contain "transformer_blocks" but not "temporal_transformer_blocks"
            if "attn2" in name and "temporal" not in name:
                cross_attention_dim = 1024 # Standard for SVD
                hidden_size = self.unet.config.block_out_channels[0] # Temporary, will be inferred
                
                # Logic to find the correct hidden_size for each block
                if "down_blocks.0" in name: hidden_size = 320
                elif "down_blocks.1" in name: hidden_size = 640
                elif "down_blocks.2" in name: hidden_size = 1280
                elif "mid_block" in name: hidden_size = 1280
                elif "up_blocks.1" in name: hidden_size = 1280
                elif "up_blocks.2" in name: hidden_size = 640
                elif "up_blocks.3" in name: hidden_size = 320

                attn_procs[name] = IPMaskedSpatioTemporalAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim
                )
            else:
                # Use default for temporal layers and self-attention
                attn_procs[name] = self.unet.attn_processors[name]
        
        self.unet.set_attn_processor(attn_procs)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        reference_image: torch.Tensor = None, # [B, 3, H, W], range -1~1
        cross_attention_kwargs = None,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:

        if reference_image is not None:
            cross_attention_kwargs = cross_attention_kwargs or {}

            # 1. Extract Face Identity (ArcFace/FaRL are frozen)
            with torch.no_grad():
                face_id_embeds = self.arcface(reference_image.float()) # [B, 512, 7, 7]
                face_id_embeds = face_id_embeds.to(sample.dtype)
                face_repr_embeds = self.farl(reference_image.float()) # [B, 197, 768] 
                face_repr_embeds = face_repr_embeds.to(sample.dtype)

            # 2. Project Face Identity (Trainable)
            id_tokens = self.image_proj1(face_id_embeds) # [B, 49, 1024]
            repr_tokens = self.image_proj2(face_repr_embeds) # [B, 197, 1024]
            
            # SVD flattens batch and frames: [B, T, C, H, W] -> [B*T, C, H, W]
            # We need to repeat the face tokens for every frame in the video.
            num_frames = sample.shape[1]
            id_tokens = id_tokens.repeat_interleave(num_frames, dim=0) # [B*T, 49, 1024]
            repr_tokens = repr_tokens.repeat_interleave(num_frames, dim=0) # [B*T, 197, 1024]
            ip_tokens = torch.cat([id_tokens, repr_tokens], dim=1) # [B*T, 49+197, 1024]

            # 2b. Encode keypoint/control frames when available. Reference-only
            # inference still uses the identity tokens above.
            keypoint_frames = cross_attention_kwargs.get("keypoint_frames", None)
            blur_frames = cross_attention_kwargs.get("blur_frames", None)
            if (keypoint_frames is not None) and (blur_frames is not None):
                # Flatten [B, T, 3, H, W] -> [B*T, 3, H, W]
                kp_flat = keypoint_frames.flatten(0, 1).to(sample.dtype)   # [B*T, 3, H, W]
                blur_flat = blur_frames.flatten(0, 1).to(sample.dtype)   # [B*T, 3, H, W]
                kp_tokens = self.keypoint_encoder(torch.cat([kp_flat, blur_flat], dim=1))   # [B*T, 256, 1024]
                ip_tokens = torch.cat([ip_tokens, kp_tokens], dim=1) # [B*T, 49+197(+256), 1024]

            ip_tokens = self.id_attn(ip_tokens)

            # 4. INJECT STATE INTO PROCESSORS
            for name, proc in self.unet.attn_processors.items():
                if isinstance(proc, IPMaskedSpatioTemporalAttnProcessor):
                    proc.ip_adapter_image_embeds = ip_tokens
        
        out =  self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            return_dict=return_dict,
        )
        
        # 4. CLEANUP STATE (Crucial to prevent memory leaks or stale conditioning)
        for name, proc in self.unet.attn_processors.items():
            if isinstance(proc, IPMaskedSpatioTemporalAttnProcessor):
                proc.ip_adapter_image_embeds = None

        return out