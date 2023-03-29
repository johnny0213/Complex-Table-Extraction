# Copyright (c) 2022 Johnny0213 Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cv2
import numpy as np
from math import *
import json
import sys 
sys.path.append("..") 
from utils.utils import rotate_bound1, nextlevel

PROJECT_BASE = r'D:\WORK\Text_Image_Orientation\\' 
DATA_BASE = PROJECT_BASE + r'data\\' 

if __name__ == "__main__":

    file_list = []
    file_list = nextlevel(DATA_BASE, file_list, [ '.jpg' , 'jpeg']) 
    print('file_list len=', len(file_list))
    #print('file_list[:2]=', file_list[:2])

    all_list = []
    for idx, img_path in enumerate(file_list):
       
        print('img_path=', img_path)
        img_name = img_path.split('\\')[-1]
        print('img_name=', img_name)
        if 'WM' in img_name or 'face' in img_name or 'sign' in img_name or 'xxxx' in img_name:
            print('\t\tSkipped!!!')
            continue

        img = cv2.imread(img_path)
        height = img.shape[0] #高度
        width = img.shape[1] #宽度

        print('height=', height, '\twidth=', width)

        if height >= width:
            new_width = 384
            new_height = int(height *384 / width)
        else:
            new_height = 384
            new_width = int(width *384 / height)

        print('resized height=', new_height, '\twidth=', new_width)
        '''
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path + '.resized_interarea.jpg', resized)

        resized = cv2.resize(img, (new_width, new_height), ) #default is shuangxianxing
        cv2.imwrite(img_path + '.resized_shuangxianxing.jpg', resized)
		'''
        resized0 = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC) #CUBIC is the best !!!

        cv2.imwrite(PROJECT_BASE + r'text_image_orientation\img_0\\' + img_name, resized0)
        all_list.append('img_0/' + img_name + ' 0')

        ################################################################
        # rotate_then_resize is the better than resize_then_rotate !!!
        rotate90 = rotate_bound1(img, 90)
        resized90 = cv2.resize(rotate90, (new_height, new_width), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(PROJECT_BASE + r'text_image_orientation\img_90\\' + img_name, resized90)
        all_list.append('img_90/' + img_name + ' 1')

        ################################################################
        rotate180 = rotate_bound1(img, 180)
        resized180 = cv2.resize(rotate180, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(PROJECT_BASE + r'text_image_orientation\img_180\\' + img_name, resized180)
        all_list.append('img_180/' + img_name + ' 2')

        ################################################################
        rotate270 = rotate_bound1(img, 270)
        resized270 = cv2.resize(rotate270, (new_height, new_width), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(PROJECT_BASE + r'text_image_orientation\img_270\\' + img_name, resized270)
        all_list.append('img_270/' + img_name + ' 3')


    f = open(PROJECT_BASE + 'all_list.txt' ,'w',encoding='utf-8') 
    f.write('\n'.join(all_list))
    f.close()

