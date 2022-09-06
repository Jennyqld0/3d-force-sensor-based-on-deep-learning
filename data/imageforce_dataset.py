import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
import re
import xlrd
import pandas as pd
from scipy.io import loadmat
#torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_tensor_type(torch.DoubleTensor)
class imgforcDataset(data.Dataset):
    '''
    Read image and force pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(imgforcDataset, self).__init__()
        self.opt = opt
        self.paths_image = None
        self.labels_file=pd.read_csv(self.opt['xlsroot'])
     #   self.LR_env = None  # environment for lmdb
        self.image_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_image = sorted([os.path.join(opt['dataroot_image'], line.rstrip('\n')) \
                        for line in f])
        else:  # read image list from lmdb or image files
            self.image_env, self.paths_image = util.get_image_paths(opt['data_type'], opt['dataroot_image'])

        assert self.paths_image, 'Error: HR path is empty.'
       # if self.paths_LR and self.paths_HR:
          #  assert len(self.paths_LR) == len(self.paths_HR), \
           #     'HR and LR datasets have different number of images - {}, {}.'.format(\
           #     len(self.paths_LR), len(self.paths_HR))

        self.random_scale_list = [1]

    def __getitem__(self, idx):
        # get image
   #     print(index)
        num=0
        fx=0.0
        fy=0.0
        fz=0.0
       # test_paths='/home/siat/文档/三维力数据/image/test_205/'
      #  pth=f'{self.labels_file.iloc[idx, 0]}_{self.labels_file.iloc[idx, 5]}.bmp'
        pth=f'{self.labels_file.iloc[idx, 0]}.bmp'
        img_path = os.path.join(self.opt['dataroot_image'],pth)
      #  img_path=self.labels_file.iloc[idx, 4]
      #  print(img_path)
        img=util.read_img(img_path)
        scale=1/4
    #    img = (img * 255.0).round()
        h = 960
     #   img = img[61:61 + h - 1, 371:371 + h - 1]
        labels=self.labels_file.iloc[idx, [1,2,3]]
        labels = np.array(labels, dtype=np.float32)
    #    fx= labels[0,0]
     #   fy= labels[0,1]
      #  fz= labels[0,2]
        

        if self.opt['color']:
            img = util.channel_convert(img.shape[2], self.opt['color'], [img])[0]
           # print(img.shape)

        if self.opt['phase'] == 'train':
            img= util.augment([img], self.opt['use_flip'], \
                self.opt['use_rot'])[0]
           # print(img.shape)

        np.expand_dims(img, axis=2)

        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]

        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
       # print(img.shape)
        img = util.imresize(img, scale, antialiasing=True)
   #     test_path = os.path.join(test_paths,pth)
     #   import torchvision.utils
      #  torchvision.utils.save_image(
       # (img * 255).round() / 255,test_path, nrow=1, padding=0, normalize=False)
     #   print(img.shape)

     #   fz = np.array
    #    fz = np.array(fz, dtype=np.float32)
      #  xs = np.array(xs, dtype=np.float32)
     #   ys = np.array(ys, dtype=np.float32)
     
       # print("=============")
      #  print(fz.dtype)
      #  print(img.dtype)
      #  print(image_path, fz)

        return {'img': img, 'force': labels, 'img_path': img_path}

    def __len__(self):
      #  print("=============")
        #print(len(self.paths_image))
      #  return len(self.paths_image)
        return len(self.labels_file)
