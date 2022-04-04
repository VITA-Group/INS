import os, sys
import math, random
import numpy as np
import torch
import json
import cv2
from pdb import set_trace as st

class BaseNeRFDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, split='train', subsample=0, cam_id=False, rgb=True, with_mask=False):

        super().__init__()

        self.split = split

        # Read metadata
        with open(os.path.join(root_dir, 'meta.json'), 'r') as f:
            self.meta_dict = json.load(f)
            
            required_keys = ['near', 'far']
            if not np.all([(k in self.meta_dict) for k in required_keys]):
                raise IOError('Missing required meta data')
        
        # Construct loaded filename
        rgbs_name, rays_name = 'rgbs_' + split, 'rays_' + split
        if with_mask:
            masks_name = 'mask_' + split
        # add subsample suffix
        if subsample != 0:
            rgbs_name, rays_name = rgbs_name + f'_x{subsample}', rays_name + f'_x{subsample}'
            if with_mask:
                masks_name = masks_name + f'_x{subsample}'
        # add extension name
        rgbs_name, rays_name = rgbs_name + '.npy', rays_name + '.npy'
        if with_mask:
            masks_name = masks_name + '.npy'

        self.rays = np.load(os.path.join(root_dir, rays_name)) # [N, H, W, ro+rd, 3]

        # RGB files may not exist considering exhibit set
        if rgb:
            rgb_path = os.path.join(root_dir, rgbs_name)
            self.rgbs = np.load(rgb_path) # [N, H, W, C]
        
        if with_mask:
            mask_path = os.path.join(root_dir, masks_name)
            self.masks = np.load(mask_path) # [N, H, W, C]
        else:
            self.masks = None
        
        # add camera ids
        if cam_id:
            ids = np.arange(self.rays.shape[0], dtype=np.float32) # [N,]
            ids = np.reshape(ids, [-1, 1, 1, 1, 1]) # [N, 1, 1, 1, 1]
            ids = np.tile(ids, (1,)+self.rays.shape[1:-1]+(1,)) # [N, H, W, ro+rd, 3]

            # Necessary check
            # for i in range(self.rays.shape[0]):
            #     assert np.all(ids[i] == i)

            self.rays = np.concatenate([self.rays, ids], -1) # [N, H, W, ro+rd, 3+id]

        # Basic attributes
        self.height = self.rays.shape[1]
        self.width = self.rays.shape[2]

        self.image_count = self.rays.shape[0]
        self.image_step = self.height * self.width

    def num_images(self):
        return self.image_count
        
    def height_width(self):
        return self.height, self.width

    def near_far(self):
        return self.meta_dict['near'], self.meta_dict['far']

class BatchNeRFDataset(BaseNeRFDataset):

    def __init__(self, root_dir, split='train', subsample=0, cam_id=False):

        super().__init__(root_dir, split=split, subsample=subsample, cam_id=cam_id, rgb=True)

        # Cast to tensors
        self.rays = torch.from_numpy(self.rays).float()
        self.rgbs = torch.from_numpy(self.rgbs).float()

        if split == 'train':
            self.rays = self.rays.reshape([-1, 2, self.rays.shape[-1]]) # [N * H * W, ro+rd, 3(+id)]
            self.rgbs = self.rgbs.reshape([-1, self.rgbs.shape[-1]]) # [N * H * W, 3(+id)]
        else:
            self.rays = self.rays.permute([0, 3, 1, 2, 4]) # [N, ro+rd, H, W, 3(+id)]
            
    def __len__(self):            
        return self.rays.shape[0]

    def __getitem__(self, i):
        # Prohibit multiple workers
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is not None:
        #     raise ValueError("Error BatchNerfDataset does not support multi-processing")

        if self.split == 'train':
            return dict(rays = self.rays[i], target_s = self.rgbs[i]) # [3,]
        else:
            return dict(rays = self.rays[i], target_s = self.rgbs[i]) # [ro+rd, H, W, 3]


class PatchNeRFDataset(BaseNeRFDataset):

    def __init__(self, root_dir, split='train', subsample=0, cam_id=False, patch_size=48, style_path=None, with_mask=False, 
    rand_style=False,  mixed_styles=None, patch_stride=1):

        super().__init__(root_dir, split=split, subsample=subsample, cam_id=cam_id, rgb=True, with_mask=with_mask)

        self.crop_size = patch_size
        self.with_mask = with_mask
        self.rand_style = rand_style
        self.mixed_styles = mixed_styles
        self.single_style_path = style_path
        self.ps = patch_stride
        # Cast to tensors
        self.rays = torch.from_numpy(self.rays).float()
        self.rgbs = torch.from_numpy(self.rgbs).float()
        # use the provided soft alpha mask in training set on NeRF-Synthetic dataset
        if self.with_mask:
            self.masks = torch.from_numpy(self.masks).float()
            # self.masks = (self.masks >= 0.5).float()
        else:
            self.masks = torch.ones_like(self.rgbs)[..., 0:1]
        self.n_samples = self.rays.shape[0]
        self.img_h = self.rays.shape[1]
        self.img_w = self.rays.shape[2]
        self.blank_style_num = 0
        if split == 'train':
            if self.mixed_styles not in [None, "None"]:
                from glob import glob
                img_paths = glob(f"{mixed_styles}/*")
                self.style_num = len(img_paths) + self.blank_style_num # -> add blank style
                self.img_style = np.zeros([self.style_num, self.img_h, self.img_w, 3]).astype(np.float32)
                for idx, i in enumerate(img_paths):
                    img_style = cv2.imread(i)
                    img_style = cv2.cvtColor(img_style, cv2.COLOR_BGR2RGB).astype(np.float32)
                    img_style = cv2.resize(img_style, (self.img_w, self.img_h)) / 255.
                    self.img_style[idx + self.blank_style_num, ...] = img_style # -> set idx 0 as zero style
            if self.single_style_path not in [None, 'None']:
                if os.path.exists(self.single_style_path):
                    self.style_num = 1
                    img_style = cv2.imread(self.single_style_path)
                    img_style = cv2.cvtColor(img_style, cv2.COLOR_BGR2RGB).astype(np.float32)
                    # if True:
                    #     min_edge = min(img_style.shape[:2])
                    #     cent_h, cent_w = img_style.shape[0]//2, img_style.shape[1]//2
                    #     img_style = img_style[cent_h-min_edge//2:cent_h+min_edge//2, cent_w-min_edge//2:cent_w+min_edge//2, :]
                    self.img_style = img_style / 255.
        else:
            self.rays = self.rays.permute([0, 3, 1, 2, 4]) # [N, ro+rd, H, W, 3(+id)]
        
        print(f"[Dataset info]: Random style patch is {self.rand_style}, mixtured styles is {self.mixed_styles}, single style path is {self.single_style_path}, \n \
                   data mask is {self.with_mask}, image resolution is {self.img_h, self.img_w}, patch stride is {self.ps}")
    
    def __len__(self):
        return self.rays.shape[0]

    def __getitem__(self, i):
        # Prohibit multiple workers
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is not None:
        #     raise ValueError("Error BatchNerfDataset does not support multi-processing")
        if self.split == 'train':
            pass
        else:
            return dict(rays = self.rays[i], target_s = self.rgbs[i]) # [ro+rd, H, W, 3]
# Containing only rays for rendering, no rgb groundtruth
class ExhibitNeRFDataset(BaseNeRFDataset):

    def __init__(self, root_dir, subsample=0):
        super().__init__(root_dir, split='exhibit', subsample=subsample, cam_id=False, rgb=False)

        self.rays = torch.from_numpy(self.rays).float()
        self.rays = self.rays.permute([0, 3, 1, 2, 4]) # [N, ro+rd, H, W, 3(+id)]

    def __len__(self):
        # return self.image_count * self.height * self.width
        return self.rays.shape[0]

    def __getitem__(self, i):
        return dict(rays = self.rays[i]) # [H, W, 3]

# def load_dataset(dataset_path, subsample=0, cam_id=False, device=torch.device("cpu")):

#     if not os.path.isdir(dataset_path):
#         raise ValueError("No such directory containing dataset:", dataset_path)

#     train_set = BatchNeRFDataset(dataset_path, subsample=subsample, split='train', cam_id=cam_id, device=device)
#     test_set = BatchNeRFDataset(dataset_path, subsample=subsample, split='test', cam_id=True, device=device)

#     exhibit_set = None
#     try:
#         exhibit_set = ExhibitNerfDataset(dataset_path, subsample=subsample, device=device)
#     except FileNotFoundError:
#         print("Warning: No exhibit set!")

#     return train_set, test_set, exhibit_set
if __name__ == "__main__":
    pass