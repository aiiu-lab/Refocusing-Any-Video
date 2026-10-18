"""Combined classical/neural bokeh renderer for optional video preprocessing."""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .classical import ModuleRenderScatter
from .neural import ARNet, IUNet

def gaussian_blur(x, r, sigma=None):
    """Gaussian blur helper function for neural renderer."""
    r = int(round(r))
    if sigma is None:
        sigma = 0.3 * (r - 1) + 0.8
    x_grid, y_grid = torch.meshgrid(torch.arange(-int(r), int(r) + 1), torch.arange(-int(r), int(r) + 1))
    kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / 2 / sigma ** 2)
    kernel = kernel.float() / kernel.sum()
    kernel = kernel.expand(1, 1, 2*r+1, 2*r+1).to(x.device)
    x = F.pad(x, pad=(r, r, r, r), mode='replicate')
    x = F.conv2d(x, weight=kernel, padding=0)
    return x



class CombinedBokehRenderer:
    """
    Combined renderer using both classical and neural renderers.
    
    This class processes batches of video frames to generate bokeh effects.
    It should be used OUTSIDE the dataset to enable parallel data loading.
    
    Args:
        classical_renderer_path (str): Path to classical renderer checkpoint (optional)
        arnet_path (str): Path to ARNet checkpoint
        iunet_path (str): Path to IUNet checkpoint
        defocus_scale (float): Defocus scaling parameter (default: 10.0)
        gamma_min (float): Minimum gamma for normalization (default: 1.0)
        gamma_max (float): Maximum gamma for normalization (default: 5.0)
        device (str): Device for rendering (default: 'cuda')
    """
    
    def __init__(
        self,
        arnet_path: str,
        iunet_path: str,
        defocus_scale: float = 10.0,
        gamma_min: float = 1.0,
        gamma_max: float = 5.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.defocus_scale = defocus_scale
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.device = device
        
        # Initialize classical renderer
        self.classical_renderer = ModuleRenderScatter().to(device)
        self.classical_renderer.eval()
        
        # Initialize neural renderers
        self.arnet = ARNet(
            shuffle_rate=2, in_channels=5, out_channels=4, middle_channels=128,
            num_block=3, share_weight=False, connect_mode='distinct_source',
            use_bn=False, activation='elu'
        ).to(device)
        self.arnet.eval()
        self.arnet.requires_grad_(False)
        
        self.iunet = IUNet(
            shuffle_rate=2, in_channels=8, out_channels=3, middle_channels=64,
            num_block=3, share_weight=False, connect_mode='distinct_source',
            use_bn=False, activation='elu'
        ).to(device)
        self.iunet.eval()
        self.iunet.requires_grad_(False)
        
        # Load checkpoints
        checkpoint = torch.load(arnet_path, map_location=device, weights_only=False)
        self.arnet.load_state_dict(checkpoint['model'])
        checkpoint = torch.load(iunet_path, map_location=device, weights_only=False)
        self.iunet.load_state_dict(checkpoint['model'])
    
    def render_single_frame(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        zero_depth: float,
        K: float,
        gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render a single frame with combined classical and neural renderers.
        
        Args:
            image: Input image [1, C, H, W] in [0, 1] range
            depth: Depth map [1, 1, H, W] in [0, 1] range
            zero_depth: Zero depth parameter
            K: K parameter
            gamma: Gamma parameter
            
        Returns:
            Tuple of (bokeh_pred, bokeh_classical, bokeh_neural) each [1, C, H, W]
        """
        with torch.no_grad():
            # Calculate defocus (preserve sign!)
            defocus = K * (depth - zero_depth) / self.defocus_scale
            
            # Classical rendering
            bokeh_classical, defocus_dilate = self.classical_renderer(image**gamma, defocus*self.defocus_scale)
            bokeh_classical = bokeh_classical ** (1/gamma)
            defocus_dilate = defocus_dilate / self.defocus_scale
            
            # Normalize gamma for neural network
            gamma_normalized = (gamma - self.gamma_min) / (self.gamma_max - self.gamma_min)
            adapt_scale = max(defocus.abs().max().item(), 1)
            
            # Initial neural rendering at reduced scale
            image_re = F.interpolate(image, scale_factor=1/adapt_scale, mode='bilinear', align_corners=True)
            defocus_re = 1 / adapt_scale * F.interpolate(defocus, scale_factor=1/adapt_scale, mode='bilinear', align_corners=True)
            bokeh_neural, error_map = self.arnet(image_re, defocus_re, gamma_normalized)
            error_map = F.interpolate(error_map, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
            bokeh_neural.clamp_(0, 1e5)
        
            # Progressive refinement
            for scale in range(int(np.log2(adapt_scale))):
                ratio = 2**(scale+1) / adapt_scale
                h_re, w_re = int(ratio * image.shape[2]), int(ratio * image.shape[3])
                image_re = F.interpolate(image, size=(h_re, w_re), mode='bilinear', align_corners=True)
                defocus_re = ratio * F.interpolate(defocus, size=(h_re, w_re), mode='bilinear', align_corners=True)
                defocus_dilate_re = ratio * F.interpolate(defocus_dilate, size=(h_re, w_re), mode='bilinear', align_corners=True)
                bokeh_neural_refine = self.iunet(image_re, defocus_re.clamp(-1, 1), bokeh_neural, gamma_normalized).clamp(0, 1e5)
                mask = gaussian_blur(((defocus_dilate_re < 1) * (defocus_dilate_re > -1)).float(), 0.005 * (defocus_dilate_re.shape[2] + defocus_dilate_re.shape[3]))
                bokeh_neural = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(bokeh_neural, size=(h_re, w_re), mode='bilinear', align_corners=True)
        
            # Final refinement at full resolution
            bokeh_neural_refine = self.iunet(image, defocus.clamp(-1, 1), bokeh_neural, gamma_normalized).clamp(0, 1e5)
            mask = gaussian_blur(((defocus_dilate < 1) * (defocus_dilate > -1)).float(), 0.005 * (defocus_dilate.shape[2] + defocus_dilate.shape[3]))
            bokeh_neural = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(bokeh_neural, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
            
            # Combine classical and neural
            bokeh_pred = bokeh_classical * (1 - error_map) + bokeh_neural * error_map
            
            # Clamp outputs
            bokeh_pred = bokeh_pred.clamp(0, 1)
            bokeh_classical = bokeh_classical.clamp(0, 1)
            bokeh_neural = bokeh_neural.clamp(0, 1)
            
            return bokeh_pred, bokeh_classical, bokeh_neural
    
    def render_batch_optimized(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        [OPTIMIZED] Render a batch of video sequences with full batch processing.
        
        This version processes all B*T frames together in batches for maximum GPU efficiency.
        Uses the maximum adapt_scale across all frames to ensure uniform processing.
        
        Expected speedup: ~20-50x compared to sequential frame-by-frame processing.
        
        Args:
            batch: Dictionary from dataloader containing:
                - 'frames': [B, T, C, H, W]
                - 'depths': [B, T, 1, H, W]
                - 'zero_depth': [B]    -- single focal plane shared across all T frames, OR
                                [B, T] -- time-varying focal plane, one value per frame
                - 'K': [B] (tensor or list/ndarray)
                - 'gamma': [B] (tensor or list/ndarray)
                
        Returns:
            Dictionary containing:
                - 'bokeh_pred': [B, T, C, H, W] in [-1, 1] range
                - 'bokeh_classical': [B, T, C, H, W] in [-1, 1] range
                - 'bokeh_neural': [B, T, C, H, W] in [-1, 1] range
        """
        frames = batch['frames'].to(self.device)  # [B, T, C, H, W]
        depths = batch['depths'].to(self.device)  # [B, T, 1, H, W]
        
        # Convert params to tensors if needed
        if not isinstance(batch['zero_depth'], torch.Tensor):
            zero_depth = torch.stack(batch['zero_depth']).to(self.device).float()  # [T, B]
            zero_depth = zero_depth.transpose(0, 1)  # [B, T]
            # zero_depth = torch.tensor(batch['zero_depth'], device=self.device).float()
        else:
            zero_depth = batch['zero_depth'].to(self.device).float()
        # zero_depth can be [B] (same focal plane for all frames) or [B, T] (per-frame focal plane)
            
        if not isinstance(batch['K'], torch.Tensor):
            K = torch.tensor(batch['K'], device=self.device).float()
        else:
            K = batch['K'].to(self.device).float()
            
        if not isinstance(batch['gamma'], torch.Tensor):
            gamma = torch.tensor(batch['gamma'], device=self.device).float()
        else:
            gamma = batch['gamma'].to(self.device).float()
        
        B, T, C, H, W = frames.shape
        
        # Flatten batch and time dimensions: [B, T, C, H, W] -> [B*T, C, H, W]
        frames_flat = frames.reshape(B * T, C, H, W)
        depths_flat = depths.reshape(B * T, 1, H, W)
        
        # Expand parameters to match B*T frames
        # zero_depth: [B] -> broadcast same focal plane for each frame
        #             [B, T] -> per-frame focal plane; must already be (B, T)
        if zero_depth.dim() == 1:
            # Shape [B] -> [B*T]: same zero_depth for all T frames of each video
            zero_depth_flat = zero_depth.unsqueeze(1).expand(B, T).reshape(B * T)  # [B*T]
        else:
            # Shape [B, T] -> [B*T]: time-varying zero_depth
            assert zero_depth.shape == (B, T), (
                f"Expected zero_depth shape ({B}, {T}) for time-varying mode, got {zero_depth.shape}"
            )
            zero_depth_flat = zero_depth.reshape(B * T)  # [B*T]
        K_flat = K.unsqueeze(1).expand(B, T).reshape(B * T)  # [B*T]
        gamma_flat = gamma.unsqueeze(1).expand(B, T).reshape(B * T)  # [B*T]
        
        with torch.no_grad():
            # Calculate defocus for all frames (preserve sign!)
            # defocus_flat: [B*T, 1, H, W]
            defocus_flat = K_flat.view(B*T, 1, 1, 1) * (depths_flat - zero_depth_flat.view(B*T, 1, 1, 1)) / self.defocus_scale
            
            # Prepare gamma tensors for batched operations
            gamma_flat_view = gamma_flat.view(B*T, 1, 1, 1)  # For power operations
            
            # Classical rendering (batched)
            bokeh_classical_flat, defocus_dilate_flat = self.classical_renderer(
                frames_flat ** gamma_flat_view, 
                defocus_flat * self.defocus_scale
            )
            bokeh_classical_flat = bokeh_classical_flat ** (1 / gamma_flat_view)
            defocus_dilate_flat = defocus_dilate_flat / self.defocus_scale
            
            # Normalize gamma for neural network
            gamma_normalized = (gamma_flat - self.gamma_min) / (self.gamma_max - self.gamma_min)
            # Reshape to [B*T, 1, 1, 1] for broadcasting: when ARNet does ones_like(x[:,:1]) * gamma
            # x[:,:1] has shape [B*T, 1, H, W]
            gamma_normalized = gamma_normalized.view(B * T, 1, 1, 1)  # [B*T] -> [B*T, 1, 1, 1]
            
            # ===== KEY OPTIMIZATION: Use maximum adapt_scale across ALL frames =====
            adapt_scale = max(defocus_flat.abs().max().item(), 1)
            # print(f"  [Batch Rendering] Using adapt_scale={adapt_scale:.2f} for all {B*T} frames")
            
            # Initial neural rendering at reduced scale (batched)
            scale_factor = 1 / adapt_scale
            frames_re = F.interpolate(frames_flat, scale_factor=scale_factor, mode='bilinear', align_corners=True)
            defocus_re = scale_factor * F.interpolate(defocus_flat, scale_factor=scale_factor, mode='bilinear', align_corners=True)
            
            # ARNet processes all frames together
            # gamma is [B*T, 1, 1, 1], will broadcast properly with x[:,:1] which is [B*T, 1, H, W]
            bokeh_neural_flat, error_map_flat = self.arnet(
                frames_re, 
                defocus_re, 
                gamma_normalized  # [B*T, 1, 1, 1]
            )
            error_map_flat = F.interpolate(error_map_flat, size=(H, W), mode='bilinear', align_corners=True)
            bokeh_neural_flat.clamp_(0, 1e5)
            
            # Progressive refinement (batched)
            num_scales = int(np.log2(adapt_scale))
            # print(f"  [Batch Rendering] Running {num_scales} progressive refinement scales")
            
            for scale_idx in range(num_scales):
                ratio = 2**(scale_idx+1) / adapt_scale
                h_re, w_re = int(ratio * H), int(ratio * W)
                
                # Resize all frames to this scale
                frames_re = F.interpolate(frames_flat, size=(h_re, w_re), mode='bilinear', align_corners=True)
                defocus_re = ratio * F.interpolate(defocus_flat, size=(h_re, w_re), mode='bilinear', align_corners=True)
                defocus_dilate_re = ratio * F.interpolate(defocus_dilate_flat, size=(h_re, w_re), mode='bilinear', align_corners=True)
                
                # IUNet refinement (batched)
                bokeh_neural_refine = self.iunet(
                    frames_re, 
                    defocus_re.clamp(-1, 1), 
                    bokeh_neural_flat, 
                    gamma_normalized  # [B*T, 1, 1, 1]
                ).clamp(0, 1e5)
                
                # Compute mask (batched)
                mask = gaussian_blur(
                    ((defocus_dilate_re < 1) * (defocus_dilate_re > -1)).float(), 
                    0.005 * (h_re + w_re)
                )
                
                # Blend (batched)
                bokeh_neural_flat = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(
                    bokeh_neural_flat, size=(h_re, w_re), mode='bilinear', align_corners=True
                )
            
            # Final refinement at full resolution (batched)
            bokeh_neural_refine = self.iunet(
                frames_flat, 
                defocus_flat.clamp(-1, 1), 
                bokeh_neural_flat, 
                gamma_normalized  # [B*T, 1, 1, 1]
            ).clamp(0, 1e5)
            
            mask = gaussian_blur(
                ((defocus_dilate_flat < 1) * (defocus_dilate_flat > -1)).float(), 
                0.005 * (H + W)
            )
            
            bokeh_neural_flat = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(
                bokeh_neural_flat, size=(H, W), mode='bilinear', align_corners=True
            )
            
            # Combine classical and neural (batched)
            bokeh_pred_flat = bokeh_classical_flat * (1 - error_map_flat) + bokeh_neural_flat * error_map_flat
            
            # Clamp outputs
            bokeh_pred_flat = bokeh_pred_flat.clamp(0, 1)
            bokeh_classical_flat = bokeh_classical_flat.clamp(0, 1)
            bokeh_neural_flat = bokeh_neural_flat.clamp(0, 1)
        
        # Reshape back to [B, T, C, H, W]
        bokeh_pred_video = bokeh_pred_flat.reshape(B, T, C, H, W)
        bokeh_classical_video = bokeh_classical_flat.reshape(B, T, C, H, W)
        bokeh_neural_video = bokeh_neural_flat.reshape(B, T, C, H, W)
        
        # Normalize to [-1, 1] range
        bokeh_pred_video = bokeh_pred_video * 2.0 - 1.0
        bokeh_classical_video = bokeh_classical_video * 2.0 - 1.0
        bokeh_neural_video = bokeh_neural_video * 2.0 - 1.0
        
        return {
            'bokeh_pred': bokeh_pred_video,
            'bokeh_classical': bokeh_classical_video,
            'bokeh_neural': bokeh_neural_video,
            'defocus_map': defocus_flat.reshape(B, T, 1, H, W)  # Optional: return defocus map for analysis
        }
    
    def render_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Render a batch of video sequences.
        
        This is a wrapper that calls the optimized batched version.
        The old sequential version is kept as render_batch_sequential() for reference.
        
        Args:
            batch: Dictionary from dataloader
                
        Returns:
            Dictionary containing rendered videos
        """
        return self.render_batch_optimized(batch)
    
    def render_batch_sequential(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        [LEGACY] Sequential frame-by-frame rendering (for reference/debugging).
        
        This is the original implementation - kept for comparison.
        Use render_batch() or render_batch_optimized() instead for better performance.
        """
        frames = batch['frames'].to(self.device)  # [B, T, C, H, W]
        depths = batch['depths'].to(self.device)  # [B, T, 1, H, W]
        # zero_depth: [B] (same focal plane) or [B, T] (per-frame focal plane)
        zero_depth = batch['zero_depth']
        K = batch['K']  # [B]
        gamma = batch['gamma']  # [B]
        
        B, T, C, H, W = frames.shape
        
        # Determine whether zero_depth is time-varying
        # Accept both tensor and list/ndarray; normalise to a tensor for indexing
        if not isinstance(zero_depth, torch.Tensor):
            zero_depth = torch.tensor(zero_depth, device=self.device).float()
        else:
            zero_depth = zero_depth.to(self.device).float()
        time_varying_zd = zero_depth.dim() == 2  # True -> shape [B, T]
        
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K, device=self.device).float()
        else:
            K = K.to(self.device).float()
        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma, device=self.device).float()
        else:
            gamma = gamma.to(self.device).float()
        
        bokeh_pred_batch = []
        bokeh_classical_batch = []
        bokeh_neural_batch = []
        
        with torch.no_grad():
            for b in range(B):
                bokeh_pred_seq = []
                bokeh_classical_seq = []
                bokeh_neural_seq = []
                
                for t in range(T):
                    frame = frames[b, t:t+1]  # [1, C, H, W]
                    depth = depths[b, t:t+1]  # [1, 1, H, W]
                    
                    bokeh_pred, bokeh_classical, bokeh_neural = self.render_single_frame(
                        frame, depth,
                        zero_depth[b, t].item() if time_varying_zd else zero_depth[b].item(),
                        K[b].item(), gamma[b].item()
                    )
                    
                    bokeh_pred_seq.append(bokeh_pred.squeeze(0))  # [C, H, W]
                    bokeh_classical_seq.append(bokeh_classical.squeeze(0))
                    bokeh_neural_seq.append(bokeh_neural.squeeze(0))
                
                bokeh_pred_batch.append(torch.stack(bokeh_pred_seq, dim=0))  # [T, C, H, W]
                bokeh_classical_batch.append(torch.stack(bokeh_classical_seq, dim=0))
                bokeh_neural_batch.append(torch.stack(bokeh_neural_seq, dim=0))
        
        # Stack batches
        bokeh_pred_video = torch.stack(bokeh_pred_batch, dim=0)  # [B, T, C, H, W]
        bokeh_classical_video = torch.stack(bokeh_classical_batch, dim=0)
        bokeh_neural_video = torch.stack(bokeh_neural_batch, dim=0)
        
        # Normalize to [-1, 1] range
        bokeh_pred_video = bokeh_pred_video * 2.0 - 1.0
        bokeh_classical_video = bokeh_classical_video * 2.0 - 1.0
        bokeh_neural_video = bokeh_neural_video * 2.0 - 1.0
        
        return {
            'bokeh_pred': bokeh_pred_video,
            'bokeh_classical': bokeh_classical_video,
            'bokeh_neural': bokeh_neural_video
        }

