import cv2
import os
import numpy as np
import glob
import tqdm
import torch
from torchvision import transforms
from PIL import Image
import concurrent.futures

class ToPatches(object):
    def __init__(self, patch_size=16):
        self.patch_size = patch_size

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to patches.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)  # Convert PIL Image to Tensor

        _, height, width = img.shape
        img = img.unsqueeze(0)
        num_patches_x = width // self.patch_size
        num_patches_y = height // self.patch_size

        patches = []
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                left = j * self.patch_size
                upper = i * self.patch_size
                patch = img[:, :, upper:upper + self.patch_size, left:left + self.patch_size]
                patches.append(patch)

        patches = torch.stack(patches, dim=1)
        # append the patches to 1600 with zeros
        if len(patches) < 1600:
            patches = torch.cat([patches, torch.zeros_like(patches[0]).repeat(1, 1600 - len(patches))], dim=1)
        elif len(patches) > 1600:
            patches = patches[:, :1600]
        return patches

def process_image(img_file):
    img = Image.open(img_file)
    patches = transform(img)
    return patches.shape[1]

img_dir = "D:\\Dataset\\AVA\\images"
img_files = glob.glob(os.path.join(img_dir, "*.jpg"))

transform = ToPatches()

# 使用ThreadPoolExecutor来并行处理
patch_num = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Map process_image function to each image file
    results = list(tqdm.tqdm(executor.map(process_image, img_files), total=len(img_files)))

patch_num.extend(results)

with open("patch_num.txt", "w") as f:
    for num in patch_num:
        f.write(str(num) + "\n")



