import cv2
import numpy as np
import os

#
kernel_size = 11
data_in_path = 'lego/val'
data_out_path = 'lego_Guassblur/71/val/'

if __name__ == "__main__":
    imgs_T = os.listdir(data_in_path)
    for image_name in imgs_T:
        if image_name == '.ipynb_checkpoints':
            continue
        img_T = cv2.imread(data_in_path + image_name)
        #print(data_in_path + image_name)
        dst = cv2.GaussianBlur(img_T, (kernel_size, kernel_size), 0) 
        cv2.imwrite(data_out_path + image_name, dst)
        # cv2.waitKey()
