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
import json
import os

PROJECT_BASE = r'D:\WORK\NER\\'
DATA_BASE = PROJECT_BASE + r'data\\'

if __name__ == "__main__":

    '''
    pporclabel sample:
    UPPER_FOLDER/zh_train_0.jpg   
    [
        {
            "transcription": "汇丰晋信", 
            "points": [[104, 114], [530, 114], [530, 175], [104, 175]], 
            "difficult": false, 
            "key_cls": "OTHER"
            "id": 1, 
            "linking": []
        }, 

    ernie-layout NER sample:
    {"name": "UPPER_FOLDER/zh_train_0.jpg", 
      "text": ["汇", "丰", "晋", "信", ...],                                      
      "bbox": [[1956, 143, 2179, 196], [2065, 185, 2100, 213], ...],    
      "segment_bbox": [[789, 41, 880, 56], [827, 52, 875, 61], ...],   
      "segment_id": [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, ...],                     
      "image": "base64",
      "width": 2480, 
      "height": 3508, 
      "qas": [
            {'question': 'HEADER', 'answers': [{'text': '纾困申请表', 'answer_start': 7, 'answer_end': 12}, {'text': '仅限业主自住 物业', 'answer_start': 12, 'answer_end': 20}]}, 
            {'question': 'ANSWER', 'answers': [{'text': '夏艳辰', 'answer_start': 20, 'answer_end': 23}, 
                {'text': '湖南省怀化市市辖区', 'answer_start': 41, 'answer_end': 50}, 
                {'text': '15845878995', 'answer_start': 59, 'answer_end': 60},]}, 
            {'question': 'QUESTION', 'answers': [{'text': '姓名:', 'answer_start': 23, 'answer_end': 26}, 
                {'text': '出生日期:', 'answer_start': 26, 'answer_end': 31}, 
                {'text': '电子邮件:', 'answer_start': 31, 'answer_end': 36}, ]}
        ]
    }
    '''

    file = open(DATA_BASE + 'Label.txt', 'r', encoding='utf-8')
    lines = [ line for line in file ]
    file.close()
    print('original dataset len=', len(lines))

    with open(os.path.join(DATA_BASE, "train.json"), "w", encoding='utf-8') as fout:

        for idx, line in enumerate(lines):

            img_path = PROJECT_BASE + line.split('\t')[0].split('/')[1]
            print("process: [{}/{}], image file name: {}".format(idx, len(lines), img_path))

            ocr_result = json.loads( line.split('\t')[1] )
            #print('ocr_result=', ocr_result)
            image = cv2.imread(img_path)
            height = image.shape[0] #高度
            width = image.shape[1] #宽度

            word_index = 0
            for id, segment in enumerate(ocr_result): # A textbox in OCR is a segment in Ernie_layout_NER.
                points = segment['points']

                ###############################################################
                #1. Calculate a segment's bbox
                min_x = min(points[0][0], points[1][0], points[2][0], points[3][0], )
                min_y = min(points[0][1], points[1][1], points[2][1], points[3][1], )
                max_x = max(points[0][0], points[1][0], points[2][0], points[3][0], )
                max_y = max(points[0][1], points[1][1], points[2][1], points[3][1], )
                segment['bbox'] = [ min_x, min_y, max_x, max_y ]
                #print('segment[bbox]=', segment['bbox'])

                ###############################################################
                # 2. Calculate the start and end position of a segment in the single-word/character list
                segment['answer_start'] = word_index
                word_index += len(segment['transcription'])
                segment['answer_end'  ] = word_index

                ###############################################################
                # 3. Scale a segment's bbox to [1000, 1000]:
                segment['segment_bbox'] = [ 
                                            int(segment['bbox'][0]*1000/width ),
                                            int(segment['bbox'][1]*1000/height),
                                            int(segment['bbox'][2]*1000/width ),
                                            int(segment['bbox'][3]*1000/height),
                                            ]
                #print('segment[segment_bbox]=', segment['segment_bbox'])

                ###############################################################
                # 4. Set a segment's id
                segment['id'] = id

                ###############################################################
                # 5. Calculate every single word/character's bbox in a segment
                transcription_len = len(segment['transcription'])
                '''SAMPLE
                "points": [[164.0, 14.0], [614.0, 14.0], [614.0, 39.0], [164.0, 39.0]]

                x0,y0 [164.0, 14.0]                     x1,y1 [614.0, 14.0]
                x3,y3 [164.0, 39.0]                     x2,y2 [614.0, 39.0]
                '''
                delta_x_upper = (points[1][0] - points[0][0]) / transcription_len
                delta_y_upper = (points[1][1] - points[0][1]) / transcription_len
                delta_x_lower = (points[2][0] - points[3][0]) / transcription_len
                delta_y_lower = (points[2][1] - points[3][1]) / transcription_len
                # segment['words'] is a temporary attribute, just for calculating every single word/character's bbox
                segment['words'] = []
                for i in range(0, transcription_len):
                    segment['words'].append( 
                        {
                            'box': 
                                [ 
                                    int( points[0][0]+delta_x_upper*i),     int(points[0][1]+delta_y_upper*i), 
                                    int( points[3][0]+delta_x_lower*(i+1)), int(points[3][1]+delta_y_lower*(i+1) ), 
                                ], 
                            'text': segment['transcription'][i], 
                        } 
                    )
                #print('segment[words]=', segment['words'])

            result = {
                "name": img_path,
                "text":         [ word['text']             for segment in ocr_result for word in segment["words"] ], # Every single word/character's text
                "bbox":         [ word['box']              for segment in ocr_result for word in segment["words"] ],# Every single word/character's bbox
                "segment_bbox": [ segment['segment_bbox']  for segment in ocr_result for word in segment["words"] ],# Every OCR textbox's bbox, SCALED TO [1000, 1000]
                "segment_id":   [ segment['id']            for segment in ocr_result for word in segment["words"] ],# the id of the textbox that every single word/character belongs to
                "width": width, 
                "height": height, 
                "qas": [
                            {
                                "question": "PAGE_HEADER", 
                                "answers": [ { 'text': segment["transcription"], 'answer_start': segment['answer_start'], 'answer_end': segment['answer_end']} for segment in ocr_result if segment['key_cls']=='PAGE_HEADER' ]
                            },
                            {
                                "question": "PROFILE_KEY", 
                                "answers": [ { 'text': segment["transcription"], 'answer_start': segment['answer_start'], 'answer_end': segment['answer_end']} for segment in ocr_result if segment['key_cls']=='PROFILE_KEY' ]
                            },
                            {
                                "question": "PROFILE_VALUE", 
                                "answers": [ { 'text': segment["transcription"], 'answer_start': segment['answer_start'], 'answer_end': segment['answer_end']} for segment in ocr_result if segment['key_cls']=='PROFILE_VALUE' ]
                            },
                            {
                                "question": "ITEM_NAME", 
                                "answers": [ { 'text': segment["transcription"], 'answer_start': segment['answer_start'], 'answer_end': segment['answer_end']} for segment in ocr_result if segment['key_cls']=='ITEM_NAME' ]
                            },
                             {
                                "question": "ITEM_VALUE", 
                                "answers": [ { 'text': segment["transcription"], 'answer_start': segment['answer_start'], 'answer_end': segment['answer_end']} for segment in ocr_result if segment['key_cls']=='ITEM_VALUE' ]
                            },
                            {
                                "question": "SPEC", 
                                "answers": [ { 'text': segment["transcription"], 'answer_start': segment['answer_start'], 'answer_end': segment['answer_end']} for segment in ocr_result if segment['key_cls']=='SPEC' ]
                            },
                            {
                                "question": "UNIT", 
                                "answers": [ { 'text': segment["transcription"], 'answer_start': segment['answer_start'], 'answer_end': segment['answer_end']} for segment in ocr_result if segment['key_cls']=='UNIT' ]
                            },
                            {
                                "question": "UNIT_PRICE", 
                                "answers": [ { 'text': segment["transcription"], 'answer_start': segment['answer_start'], 'answer_end': segment['answer_end']} for segment in ocr_result if segment['key_cls']=='UNIT_PRICE' ]
                            },
                            {
                                "question": "QUANTITY", 
                                "answers": [ { 'text': segment["transcription"], 'answer_start': segment['answer_start'], 'answer_end': segment['answer_end']} for segment in ocr_result if segment['key_cls']=='QUANTITY' ]
                            },
                            {
                                "question": "INSURANCE_CLASS", 
                                "answers": [ { 'text': segment["transcription"], 'answer_start': segment['answer_start'], 'answer_end': segment['answer_end']} for segment in ocr_result if segment['key_cls']=='INSURANCE_CLASS' ]
                            },
                            {
                                "question": "INSURANCE_PERCENTAGE", 
                                "answers": [ { 'text': segment["transcription"], 'answer_start': segment['answer_start'], 'answer_end': segment['answer_end']} for segment in ocr_result if segment['key_cls']=='INSURANCE_PERCENTAGE' ]
                            },
                            {
                                "question": "OTHER", 
                                "answers": [ { 'text': segment["transcription"], 'answer_start': segment['answer_start'], 'answer_end': segment['answer_end']} for segment in ocr_result if segment['key_cls']=='OTHER' ]
                            },
                         ],
                "image": "N/A",
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
