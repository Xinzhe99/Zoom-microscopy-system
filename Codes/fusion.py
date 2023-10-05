# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import time
from datetime import datetime
import glob
import os
import re
import cv2
import torch
from numba import jit
from numba.typed import List
from PyQt5.QtCore import QThread
import numpy as np
from torchvision import transforms
import cal_sf

@jit(nopython=True, cache=True)
def decisionmap_process(input_dm_np, k_size=None, use_fuzzy_op=None):
    img_height, img_width = input_dm_np.shape[0], input_dm_np.shape[1]
    padding_len = k_size // 2
    pad_img = np.zeros((img_height + 2 * padding_len, img_width + 2 * padding_len)).astype(np.int64)
    for i in range(pad_img.shape[1]):
        for j in range(pad_img.shape[0]):
            if i > padding_len - 1 and j > padding_len - 1 and i < pad_img.shape[1] - padding_len and j < \
                    pad_img.shape[0] - padding_len:
                pad_img[j][i] = int(input_dm_np[j - padding_len][i - padding_len])
            else:
                pad_img[j][i] = -1
    new_img = np.zeros((img_height, img_width)).astype(np.int64)
    for i in range(img_height):
        for j in range(img_width):
            # get original Value
            original_Value = pad_img[i + padding_len, j + padding_len]
            # get matrix
            moving_matrix = pad_img[i:i + 2 * padding_len + 1, j:j + 2 * padding_len + 1].flatten()
            # delete pidding value -1
            moving_matrix = moving_matrix[moving_matrix != -1]
            # get max min ,med,most_fre
            moving_most_fre = np.argmax(np.bincount(moving_matrix))
            if use_fuzzy_op == True:
                new_img[i][j] = int(np.median(moving_matrix))
            else:
                if original_Value == moving_most_fre:
                    new_img[i][j] = original_Value
                else:
                    new_img[i][j] = moving_most_fre
    return new_img

@jit(nopython=True, cache=True)
def Final_fusion(in_img_cv_list, in_decisionmap, height, width):  # Function is compiled and runs in machine code
    pic_fusion = np.zeros((height, width, 3), dtype=np.uint8)
    pic_fusion_height = pic_fusion.shape[0]
    pic_fusion_width = pic_fusion.shape[1]
    pic_fusion_channels = pic_fusion.shape[2]
    for row in range(pic_fusion_height):
        for col in range(pic_fusion_width):
            for channel in range(pic_fusion_channels):
                pic_fusion[row, col, channel] = in_img_cv_list[in_decisionmap[row, col]][row, col, channel]
    return pic_fusion

@jit(nopython=True,cache=True)
def Generate_decisionmap(sf_list,height,width):
    sf_numba_list = List(sf_list)

    sf_num_np = np.zeros(shape=((height, width)))
    decisionmap_np = np.zeros((height, width),dtype=np.int64)
    for i in range(len(sf_numba_list)):
        for a in range(sf_num_np.shape[0]):
            for b in range(sf_num_np.shape[1]):
                if sf_numba_list[i][a][b]>=sf_num_np[a][b]:
                    sf_num_np[a][b]=sf_numba_list[i][a][b]
                    decisionmap_np[a][b]=int(i)
    return decisionmap_np,sf_num_np

class Fusion_stack(QThread):

    def __init__(self, stack_path,result_path,source_format,save_format,Using_Optimised_Processing,image_scale,use_gpu):
        super().__init__()

        self.stack_path=stack_path
        self.result_path=result_path
        self.source_format=source_format
        self.save_format=save_format
        self.Using_Optimised_Processing=Using_Optimised_Processing
        self.image_scale=image_scale
        self.use_gpu=use_gpu
        self.fusion()

    def fusion(self):


        t1=time.time()
        glob_format='*'+self.source_format
        pic_sequence_list = glob.glob(os.path.join(self.stack_path,glob_format))
        pic_sequence_list.sort(
            key=lambda x: int(str(re.findall("\d+", x.split('/')[-1])[-1])))  # Sort by the number in the file name
        img_cv_list = [None] * (len(pic_sequence_list))
        img_list = [None] * (len(pic_sequence_list))
        sf_list = [None] * (len(pic_sequence_list))

        img_get_size=cv2.imread(pic_sequence_list[0])
        height ,width= img_get_size.shape[:2]
        width=int(width*self.image_scale)
        height=int(height*self.image_scale)
        print(width)
        print(height)

        if torch.cuda.is_available():
            print('GPU Mode Acitavted')
        else:
            print('CPU Mode Acitavted')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cal_model = cal_sf.cal_sf_by_net()

        for i in range(len(pic_sequence_list)):

            img_list[i] =cv2.imread(pic_sequence_list[i])
            img_list[i] =cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            img_list[i] = cv2.resize(img_list[i],((width, height)))

            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])
            img_list[i] = data_transforms(img_list[i]).unsqueeze(0).to(device)

            sf_list[i] = cal_model(img_list[i]).cpu().numpy()
            print('finish send no.{} pic into the net'.format(str(i)))

        t_current=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # generate initial decision map
        decisionmap_numpy, sf_numpy = Generate_decisionmap(sf_list,height,width)

        # save decision map numpy
        decisionmap_np_dst=os.path.join(self.result_path,t_current+'_ori.npy')
        np.save(decisionmap_np_dst, decisionmap_numpy)
        print('Finish gernerate initial decision map,and save in {}'.format(decisionmap_np_dst))


        # Optimization Decision Map
        if self.Using_Optimised_Processing == True:
            Post_processing_core_size = min(height, width) // 20 if (min(height, width) // 20) % 2 != 0 else (min(height,width) // 20) - 1
            decisionmap_np_afterprocess = decisionmap_process(decisionmap_numpy, k_size=Post_processing_core_size,
                                                              use_fuzzy_op=False)
            decisionmap_np_optimized_dst = os.path.join(self.result_path,t_current+'_opt.npy')
            np.save(decisionmap_np_optimized_dst, decisionmap_np_afterprocess)
            print('Finish gernerate Final decision map,and save in {}'.format(decisionmap_np_optimized_dst))
        else:
            decisionmap_np_afterprocess = decisionmap_numpy

        # fusion step1:create pic list which could be read by cv2
        for i in range(len(pic_sequence_list)):
            img_cv_list[i] = cv2.imread(pic_sequence_list[i])
            img_cv_list[i] = cv2.resize(img_cv_list[i], (width, height))

        # fusion step2:create final pic
        pic_fusion = Final_fusion(img_cv_list, decisionmap_np_afterprocess, height, width).astype(np.uint8)

        save_path = os.path.join(self.result_path,t_current+'.'+self.save_format)
        cv2.imwrite(save_path, pic_fusion)

        self.result_path_and_time='Fusion Done!,and save in {}, cost {} s'.format(save_path,str(round(float(time.time()-t1),2)))
        print(self.result_path_and_time)

