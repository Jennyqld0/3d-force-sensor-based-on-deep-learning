import os
import cv2
import xlrd
import numpy as np
import csv
import torch
import math
import pandas as pd
import random
import data.util as util
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
import math
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
  #  if img.ndim == 2:
    #    img = np.expand_dims(img, axis=2)
    # some images have 4 channels
  #  if img.shape[2] > 3:
  #      img = img[:, :, :3]
    return img

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)



def main():

    testxlsroot = "/data/qld/zhangqi/data/dabufenxishu/zengqiang/all.csv"
    test10csv="/data/qld/zhangqi/data/dabufenxishu/real/val_1000.csv"
    test10csv_2="/data/qld/zhangqi/data/dabufenxishu/real/train.csv"
   # valcsv="G:\\val15.csv"
  #  testcsv="G:\\test15.csv"

    with open(testxlsroot, 'r') as f:
        lenx = len(f.readlines())
   #     print(lenx)
    with open(testxlsroot, 'r') as f:
        reader =csv.reader(f)
        result = list(reader)
   #     print(result)

  #  print(lenx)
    tou=['xuhao','fx','fy','fz','image_path']   
  #  random.shuffle(result)
 #   print(result)
    len1=math.ceil(lenx * 0.1)
  #  len2=math.ceil(lenx * 0.85)
    restra=result[0:lenx]
    restra2=result[1000:lenx]
    #print(restra)
  #  resval=result[len1:len2]
   # restest=result[len2:lenx]
    for item in restra:
       img = cv2.imread(item[4])
       img = img * 1.0 / 255
       img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
       scale = 1 / 4
       rlt = util.imresize(img, scale, antialiasing=True) 
       import torchvision.utils
       torchvision.utils.save_image(
        (rlt * 255).round() / 255, item[4], nrow=1, padding=0, normalize=False)

 #   with open(test10csv, "w", newline='') as ff:
  #      f_csv = csv.writer(ff)
   #     f_csv.writerow(tou)
    #    for item in restra:
         #   print(item)
     #       f_csv.writerow(item)
    #with open(test10csv_2, "w", newline='') as ff:
     #   f_csv = csv.writer(ff)
      #  f_csv.writerow(tou)
       # for item in restra2:
         #   print(item)
        #    f_csv.writerow(item) 
  #  with open(valcsv, "w", newline='') as ff:
 #       f_csv = csv.writer(ff)
  #      for item in resval:
    #        f_csv.writerow(item)
  #  with open(testcsv, "w", newline='') as ff:
  #      f_csv = csv.writer(ff)
  #      for item in restest:
   #         f_csv.writerow(item)

      #  for i in range(lenx):
       #     print(i)
        #    row=result[i]
         #   if i<=math.ceil(lenx*0.7):
          #      with open(traincsv,"w",newline='') as ff:
           #         f_csv=csv.writer(ff)
            #        print(row)
             #       f_csv.writerow(row)
           # elif i>math.ceil(lenx*0.7) and i<=math.ceil(lenx*0.85):
            #    with open(valcsv,"w",newline='') as ff:
             #       f_csv=csv.writer(ff)
             #       f_csv.writerow(row)
           # else:
            #    with open(testcsv,"w",newline='') as ff:
             #       f_csv=csv.writer(ff)
              #      f_csv.writerow(row)




if __name__ =='__main__':
    main()
