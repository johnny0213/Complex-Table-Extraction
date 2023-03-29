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

import json
import cv2
import numpy as np
import math
import sys 
sys.path.append("..") 
from utils.utils import OCREngine, nextlevel

PROJECT_BASE = r'D:\WORK\Distortion_Correction\\'
DATA_BASE = PROJECT_BASE + r'data\\'
ocr_results_first_attempt = DATA_BASE + 'ocr_results_first_attempt.txt'  
ocr_results = DATA_BASE + 'ocr_results.txt'  

WINDOW_SIZE = 3
WINDOW_START = 0
WINDOW_END = WINDOW_START + WINDOW_SIZE


def calculate_slope(textbox):
    left_x0 = textbox['poly'][0][0]
    left_y0 = textbox['poly'][0][1]
    left_x1 = textbox['poly'][1][0]
    left_y1 = textbox['poly'][1][1]

    left_x_delta = left_x1 - left_x0 
    left_y_delta = left_y1 - left_y0
    slope = left_y_delta/left_x_delta 

    return slope, left_x_delta


def rotate_image(image, slope, center=None, scale=1.0): 
    # Calculate the angle to rotate
    width = 100
    height = slope * width
    c=math.sqrt(width**2 + height**2)
    angle=(math.asin(height/c))*180/math.pi
    print('angle = ', angle)

    (h, w) = image.shape[:2] 
    if center is None: 
        center = (w // 2, h // 2) 
    M = cv2.getRotationMatrix2D(center, angle, scale) 
    rotated = cv2.warpAffine(image, M, (w, h)) 
    return rotated


if __name__ == "__main__":

    ##################################################################################
    # 1. OCR first attempt
    file_list = []
    file_list = nextlevel(DATA_BASE, file_list, [ '.jpg' ])
    
    print('\n\nOCR first attempt. file_list len=', len(file_list))

    ocr_engine = OCREngine()
    with open(ocr_results_first_attempt, "w", encoding='utf-8') as fout:
        for idx, img_path in enumerate(file_list):

            print("process: [{}/{}], img_path= {}".format(idx, len(file_list), img_path))
            image_name_o = img_path.split('\\')[-1] 
            img = cv2.imread(img_path)

            try:
                result= ocr_engine.ocr(img)
            except Exception as e:
                print('OCR expcetion', e)
                continue

            fout.write(img_path + '\t' + json.dumps( { "ocr_result": result }, ensure_ascii=False ) + "\n")

    ##################################################################################
    # 2. Distortion Correction
    f = open(ocr_results_first_attempt ,'r',encoding='utf-8') 
    lines = [ v for v in f ]
    f.close()

    print('\n\nRotate images. line count=', len(lines))

    for line_count, line in enumerate(lines):
        image_path_name= line.split('\t')[0] 
        print('image_path_name=', image_path_name)
        image_name_o = line.split('\t')[0].split('\\')[-1] 
     
        ocr_info = json.loads(line.split('\t')[1])

        #print('ocr_info=', ocr_info)
        textboxes = ocr_info['ocr_result']
        for index, textbox in enumerate(textboxes):
            textbox['slope'], textbox['width'] = calculate_slope(textbox)

        # Split all textboxes into two halves(the upper half, the lower half) by the middle Y
        # Look for the middle Y:
        textboxes = sorted(textboxes, key=lambda r: (r["bbox"][1])) # sort all textboxes by Y, ascending order
        middle_Y = (textboxes[-1]["bbox"][1] - textboxes[0]["bbox"][1]) / 2
        print('middle textbox Y=', middle_Y)
        for idx, textbox in enumerate(textboxes):
            if textbox["bbox"][1] <= middle_Y:
                continue
            else:
                break
        print('middle textbox idx=', idx, '\ttextbox=', textboxes[idx])

        # Find the Top N textboxes in terms of width in the two halves respectively.
        textboxes_upper_half = sorted(textboxes[0:idx], key=lambda r: (r['width']), reverse=True) # sort all textboxes by width, descending order
        candidates = [ [textbox['text'], textbox['width'], textbox['slope']] for textbox in textboxes_upper_half[0:WINDOW_SIZE] ] 
        print('upper half Top ', WINDOW_SIZE, ' candidates:', candidates) #[WINDOW_START:WINDOW_END])
        textboxes_lower_half = sorted(textboxes[idx:-1], key=lambda r: (r['width']), reverse=True) # sort all textboxes by width, descending order
        candidates += [ [textbox['text'], textbox['width'], textbox['slope']] for textbox in textboxes_lower_half[0:WINDOW_SIZE] ] 
        print('upper and lower half Top ', WINDOW_SIZE, ' candidates:', candidates) #[WINDOW_START:WINDOW_END])

        # Calcuate the average slope of the textboxes
        slopes = np.array(candidates) # get the Top WINDOW_SIZE*2
        slopes = slopes[:, -1] # get the slopes of Top WINDOW_SIZE*2
        slopes = slopes.astype(np.float64) # cast string into float
        print('slopes=', slopes)
        average_slope = np.mean(slopes)
        print('upper and lower halves, slope average=', average_slope)

        # Rotate the image for the calculated angle
        image = cv2.imread(image_path_name)
        image = rotate_image(image, average_slope)
        cv2.imwrite( image_path_name + '_rotated.png', image )

    ##################################################################################
    # 3. OCR the corrected image, and get better OCR results
    file_list = []
    file_list = nextlevel(PROJECT_BASE, file_list, [ '.png' ]) 
    
    print('\n\nOCR again. file_list len=', len(file_list))
    with open(ocr_results, "w", encoding='utf-8') as fout:
        for idx, img_path in enumerate(file_list):

            print("process: [{}/{}], img_path= {}".format(idx, len(file_list), img_path))
            image_name_o = img_path.split('\\')[-1] 
            img = cv2.imread(img_path)

            try:
                result= ocr_engine.ocr(img)
            except Exception as e:
                print('OCR expcetion', e)
                continue

            fout.write(img_path + '\t' + json.dumps( { "ocr_result": result }, ensure_ascii=False ) + "\n")

