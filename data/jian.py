import os
import cv2
import xlrd
import numpy as np
import re
import math
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def read_img(path):
    # read image by cv2 or from lmdb
    # return: Numpy float32, HWC, BGR, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)
def main():
    ori='/data/qld/image/10846.bmp'
    path='/data/qld/image/10847.bmp'
    path1 = '/data/qld/10847j.bmp'
    path_pjian='/data/qld/image/10848.bmp'
    path2 = '/data/qld/10848j.bmp'
    img = read_img(path)
    img1 = (img * 255.0).round()
    img = read_img(ori)
    ori = (img * 255.0).round()
    img = read_img(path_pjian)
    img2 = (img * 255.0).round()
    img1_jia = cv2.subtract(img1, ori)
    save_img(img1_jia,path1)
    img2_jia = cv2.subtract(img2, ori)
    save_img(img2_jia,path2)


if __name__ =='__main__':
    main()