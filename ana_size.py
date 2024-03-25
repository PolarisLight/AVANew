import cv2
import os
import numpy as np
import glob
import tqdm

img_dir = "D:\\Dataset\\AVA\\images"
img_files = glob.glob(os.path.join(img_dir, "*.jpg"))

for img_file in tqdm.tqdm(img_files):
    img = cv2.imread(img_file)
    # if the image size is not 640*480 or 480*640, warning
    if img.shape[0] != 640 and img.shape[1] != 640:
        print(img_file, img.shape)

