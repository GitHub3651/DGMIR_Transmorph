"""
Custom Dataset for TransMorph - Aligned with DGMIR data loading
Author: GitHub Copilot
Description: Dataset class for loading MR-CT registration data from dataset/ directory structure
"""

import os
import nibabel as nib
import torch
import torch.utils.data as data
import numpy as np


def imgnorm(img):
    """Min-Max normalization to [0, 1]"""
    max_v = np.max(img)
    min_v = np.min(img)
    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def center_crop(data, target_size):
    """
    Perform center cropping of a 3D numpy array.
    
    Args:
        data (numpy.ndarray): The input 3D array to crop.
        target_size (tuple): The target size (depth, height, width).
    
    Returns:
        numpy.ndarray: The center-cropped array.
    """
    depth, height, width = data.shape
    target_depth, target_height, target_width = target_size
    
    start_d = (depth - target_depth) // 2
    start_h = (height - target_height) // 2
    start_w = (width - target_width) // 2
    
    return data[
           start_d:start_d + target_depth,
           start_h:start_h + target_height,
           start_w:start_w + target_width
           ]


class CustomDataset(data.Dataset):
    """
    Custom Dataset for TransMorph - Loads data from dataset/ directory structure
    Compatible with DGMIR data loading approach
    
    Directory structure:
        dataset/
            train/ or val/ or test/
                fixed/
                    image/
                        img{ID}_tcia_MR.nii.gz
                    seg/
                        seg{ID}_tcia_MR.nii.gz
                moving/
                    image/
                        img{ID}_tcia_CT.nii.gz
                    seg/
                        seg{ID}_tcia_CT.nii.gz
    """
    
    def __init__(self, data_root, split='train', vol_size=(192, 160, 192)):
        """
        Args:
            data_root: Root directory containing dataset/ folder
            split: 'train', 'val', or 'test'
            vol_size: Target volume size after center cropping (D, H, W)
        """
        self.data_root = data_root
        self.split = split
        self.vol_size = vol_size
        
        # Build file list
        self.data_list = self._build_file_list()
        self.n_data = len(self.data_list)
        
        print(f"[CustomDataset] Loaded {self.n_data} samples from {split} split")
    
    def _build_file_list(self):
        """Build list of sample IDs from directory structure"""
        split_dir = os.path.join(self.data_root, self.split)
        fixed_img_dir = os.path.join(split_dir, 'fixed', 'image')
        
        # Get all sample IDs from fixed image directory
        sample_list = []
        if os.path.exists(fixed_img_dir):
            for fname in sorted(os.listdir(fixed_img_dir)):
                if fname.startswith('img') and fname.endswith('_tcia_MR.nii.gz'):
                    # Extract sample ID (e.g., '0002' from 'img0002_tcia_MR.nii.gz')
                    sample_id = fname.replace('img', '').replace('_tcia_MR.nii.gz', '')
                    sample_list.append(sample_id)
        
        return sample_list
    
    def __getitem__(self, item):
        """
        Returns:
            fixed: Fixed image (moving MR), shape (D, H, W)
            moving: Moving image (fixed CT), shape (D, H, W)
            fixed_seg: Fixed segmentation, shape (D, H, W)
            moving_seg: Moving segmentation, shape (D, H, W)
        """
        sample_id = self.data_list[item]
        split_dir = os.path.join(self.data_root, self.split)
        
        # Construct file paths
        fixed_seg_path = os.path.join(split_dir, 'fixed', 'seg', f'seg{sample_id}_tcia_MR.nii.gz')
        moving_seg_path = os.path.join(split_dir, 'moving', 'seg', f'seg{sample_id}_tcia_CT.nii.gz')
        fixed_img_path = os.path.join(split_dir, 'fixed', 'image', f'img{sample_id}_tcia_MR.nii.gz')
        moving_img_path = os.path.join(split_dir, 'moving', 'image', f'img{sample_id}_tcia_CT.nii.gz')
        
        # Load NIfTI files
        fixed_seg = nib.load(fixed_seg_path)
        moving_seg = nib.load(moving_seg_path)
        fixed = nib.load(fixed_img_path)
        moving = nib.load(moving_img_path)
        
        # Convert to numpy arrays
        fixed = np.array(fixed.dataobj)
        moving = np.array(moving.dataobj)
        fixed_seg = np.array(fixed_seg.dataobj)
        moving_seg = np.array(moving_seg.dataobj)
        
        # Center crop
        fixed = center_crop(fixed, self.vol_size)
        moving = center_crop(moving, self.vol_size)
        fixed_seg = center_crop(fixed_seg, self.vol_size)
        moving_seg = center_crop(moving_seg, self.vol_size)
        
        # Normalization (only if mean > 1, indicating non-normalized data)
        if np.mean(fixed) > 1:
            fixed = imgnorm(fixed)
            moving = imgnorm(moving)
        
        # Convert to torch tensors (D, H, W) - 与 DGMIR 一致，不添加 channel 维度
        # DataLoader 会添加 batch 维度变成 (B, D, H, W)
        # 训练脚本中会 unsqueeze 添加 channel 维度变成 (B, 1, D, H, W)
        fixed = torch.tensor(fixed, dtype=torch.float32)
        moving = torch.tensor(moving, dtype=torch.float32)
        fixed_seg = torch.tensor(fixed_seg, dtype=torch.float32)
        moving_seg = torch.tensor(moving_seg, dtype=torch.float32)
        
        return fixed, moving, fixed_seg, moving_seg
    
    def __len__(self):
        return self.n_data
