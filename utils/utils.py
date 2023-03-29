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

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
from paddleocr import PaddleOCR


# find all files with specified extensions in a folder and its subfolders recursively
def nextlevel(level1_p, file_list, target_file_types):
    if os.path.isdir(level1_p):
        # print(level1_p)
        level1_list = os.listdir(level1_p)
        for file1 in level1_list:
            level2_p = os.path.join(level1_p, file1)
            nextlevel(level2_p, file_list, target_file_types)
    if os.path.isfile(level1_p):
        file_path = level1_p
        if file_path not in file_list:
            if file_path[-4:] in target_file_types: # Caution: Only the last four letters in file extentions are used.
                file_list.append(file_path)
    return file_list

# rotate then resize an image
def rotate_bound1(image, angle): #https://www.jb51.net/article/144471.htm
    '''
     . 旋转图片
     . @param image    opencv读取后的图像
     . @param angle    (逆)旋转角度
    '''

    (h, w) = image.shape[:2]  # 返回(高,宽,色彩通道数),此处取前两个值返回
    # 抓取旋转矩阵(应用角度的负值顺时针旋转)。参数1为旋转中心点;参数2为旋转角度,正的值表示逆时针旋转;参数3为各向同性的比例因子
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0) #angle前面加了负号表示顺时针旋转,不加负号则为逆时针
    # 计算图像的新边界维数
    newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
    newH = int((h * np.abs(M[0, 0])) + (w * np.abs(M[0, 1])))
    # 调整旋转矩阵以考虑平移
    M[0, 2] += (newW - w) / 2
    M[1, 2] += (newH - h) / 2
    # 执行实际的旋转并返回图像
    return cv2.warpAffine(image, M, (newW, newH)) # borderValue 缺省，默认是黑色

# resize then rotate an image
def rotate_bound2(image, angle):    
    '''
     . 旋转图片
     . @param image    opencv读取后的图像
     . @param angle    (逆)旋转角度
    '''

    h, w = image.shape[:2]  # 返回(高,宽,色彩通道数),此处取前两个值返回
    newW = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
    newH = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    M[0, 2] += (newW - w) / 2
    M[1, 2] += (newH - h) / 2
    return cv2.warpAffine(image, M, (newW, newH), borderValue=(255, 255, 255))

def draw_ner_results(image,
                     ocr_results,
                     test_results_ERNIE_LAYOUT,
                     font_path="simfang.ttf",
                     font_size=18):
    color_map= {
        'PAGE_HEADER': (0,0,0),          #black
        'PROFILE_KEY': (255,255,0),      #yellow
        'PROFILE_VALUE': (128,128,0),    #olive
        'ITEM_NAME': (162, 106, 138),    #
        'ITEM_VALUE': (28, 99, 62),      #dark green
        'UNIT': (255,0,255),             #Magenta        
        'UNIT_PRICE': (128,0,128),       #Purple        
        'QUANTITY': (138,43,226),        #BlueViolet        
        'SPEC': (0,0,255),               #blue        
        'INSURANCE_CLASS': (0,255,0),    #lime        
        'INSURANCE_PERCENTAGE': (60,179,113),#SpringGreen       
        'OTHER': (128,128,128),          #gray        
        }

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    for text_id, (ocr_info, test_result) in enumerate(zip(ocr_results, test_results_ERNIE_LAYOUT)):
        ocr_info['pred'] = test_result[0] # This is the prediction label. 

        if test_result[0] not in color_map:
            print('ERROR: ', test_result[0], ' not in colr_map!!!')
            continue

        color = color_map[test_result[0]]
        text = "{}{}".format('[' + str(text_id) + ']', ocr_info["text"])

        draw_box_txt(ocr_info["bbox"], text, draw, font, font_size, color)

    img_new = Image.blend(image, img_new, 0.5)
    return np.array(img_new)

def draw_box_txt(bbox, text, draw, font, font_size, color):
    # draw ocr results outline
    bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
    draw.rectangle(bbox, fill=color)

    # draw ocr results
    start_y = max(0, bbox[0][1] - font_size)
    tw = font.getsize(text)[0]
    draw.rectangle(
        [(bbox[0][0] + 1, start_y), (bbox[0][0] + tw + 1, start_y + font_size)],
        fill=(0, 0, 255))
    draw.text((bbox[0][0] + 1, start_y), text, fill=(255, 255, 255), font=font)

def draw_lines(pairs, textboxes, image_path_name):
    LINE_COLOR2 = (0,0,128) #navy
    LINE_COLOR = (0,0,0)    #black
    
    img = cv2.imread(image_path_name)
    for idx, pair in enumerate(pairs):
        if pair[1] == -1:
            continue

        for textbox in textboxes:
            if textbox['id'] == pair[0]:
                ocr_info_head = textbox
                continue
            elif textbox ['id'] == pair[1]:
                ocr_info_tail = textbox
                continue

        center_head = (
            int( (ocr_info_head['bbox'][0] + ocr_info_head['bbox'][2]) // 2 ),
            int( (ocr_info_head['bbox'][1] + ocr_info_head['bbox'][3]) // 2 )
                )
        center_tail = (
            int( (ocr_info_tail['bbox'][0] + ocr_info_tail['bbox'][2]) // 2 ),
            int( (ocr_info_tail['bbox'][1] + ocr_info_tail['bbox'][3]) // 2 )
                )

        line_color = LINE_COLOR if (idx%2) == 0 else LINE_COLOR2
        cv2.line(img, center_head, center_tail, line_color, 2, cv2.LINE_AA)

    cv2.imwrite( image_path_name + '_LINES.png', img )


def trans_poly_to_bbox(poly):
    x1 = np.min([p[0] for p in poly])
    x2 = np.max([p[0] for p in poly])
    y1 = np.min([p[1] for p in poly])
    y2 = np.max([p[1] for p in poly])
    return [x1, y1, x2, y2]


def parse_ocr_info(ocr_result):
    ocr_info = []
    for res in ocr_result:
        ocr_info.append({
            "text": res[1][0],
            "bbox": trans_poly_to_bbox(res[0]),
            "poly": res[0],
        })
    return ocr_info


class OCREngine(object):
    def __init__(self):
        # init ocr_engine
        paddle_rec = r'D:\WORK\Re_OCR\inference\ch_PP-OCRv3_rec_distillation\Student'
        paddle_det = r'D:\WORK\Re_OCR\inference\ch_PP-OCRv3_det_student'
        paddle_cls = r'd:\WORK\Re_OCR\inference\cls' # Not used

        self.ocr_engine = PaddleOCR(cls=True, use_angle_cls=False, use_space_char=True, use_gpu=True,
                               det_max_side_lent=960,

                               rec_model_dir=paddle_rec,
                               cls_model_dir=paddle_cls,
                               det_model_dir=paddle_det,
                               rec_image_shape='3, 32, 320',  

                               # # DB parmas
                               det_db_thresh=0.1,  
                               det_db_box_thresh=0.,             
                               det_db_unclip_ratio=1.5,
                               det_limit_type="min", 
                               det_limit_side_len=736,

                               # # EAST parmas
                               det_east_score_thresh=0.4,
                               det_east_cover_thresh=0.,
                               det_east_nms_thresh=0.1,
                               
                               gpu_mem=1600, rec_batch_num=6, cls_batch_num=6)  

        
    def ocr(self, img):
        ocr_result = self.ocr_engine.ocr(img, cls=False)
        ###print('ocr_result=', ocr_result)
        ocr_info = parse_ocr_info(ocr_result)

        return ocr_info

