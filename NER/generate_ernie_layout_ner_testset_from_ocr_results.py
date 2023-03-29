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
    ocr result sample:
    D:\WORK\NER\data\12f3492fb3ef492e9288171207846d7a_0.jpg {"ocr_result": 
        [
            {"text": "汇丰晋信", "bbox": [546.0, 461.0, 1126.0, 501.0], "poly": [[546.0, 461.0], [1126.0, 474.0], [1125.0, 501.0], [546.0, 488.0]]}, 
            {"text": "28", "bbox": [1361.0, 484.0, 1393.0, 510.0], "poly": [[1361.0, 484.0], [1393.0, 484.0], [1393.0, 510.0], [1361.0, 510.0]]}, 
            {"text": "0.25", "bbox": [1512.0, 487.0, 1573.0, 517.0], "poly": [[1514.0, 487.0], [1573.0, 491.0], [1571.0, 517.0], [1512.0, 514.0]]}, 
            ...
        ]
    }

    ernie-layout NER sample:
    {"name": "D:\WORK\NER\data\12f3492fb3ef492e9288171207846d7a_0.jpg", 
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

    file = open(DATA_BASE + 'ocr_results_test.txt', 'r', encoding='utf-8')
    lines = [ line for line in file ]
    file.close()
    print('original dataset len=', len(lines))

    with open(os.path.join(PROJECT_BASE, "test.json"), "w", encoding='utf-8') as fout:

        for idx, line in enumerate(lines):

            img_path = line.split('\t')[0]
            print("process: [{}/{}], image file name: {}".format( idx, len(lines), img_path) )
            img = cv2.imread(img_path)
            height = img.shape[0] #高度
            width = img.shape[1] #宽度
            
            ocr_result = json.loads( line.split('\t')[1] )
            ocr_result = ocr_result['ocr_result']
            #print('ocr_result=', ocr_result)

            word_index = 0
            for segment_index, segment in enumerate(ocr_result):

                ###############################################################
                # 1. Calculate the start and end position of a segment in the single-word/character list
                segment['answer_start'] = word_index
                word_index += len(segment['text'])
                segment['answer_end'  ] = word_index

                ###############################################################
                # 2. Scale a segment's bbox to [1000, 1000]:
                segment['segment_bbox'] = [ 
                                            int(segment['bbox'][0]*1000/width ),
                                            int(segment['bbox'][1]*1000/height),
                                            int(segment['bbox'][2]*1000/width ),
                                            int(segment['bbox'][3]*1000/height),
                                            ]
                #print('segment[segment_bbox]=', segment['segment_bbox'])

                ###############################################################
                # 3. Set a segment's id
                segment['id'] = id

                ###############################################################
                # 4. Calculate every single word/character's bbox in a segment
                text_len = len(segment['text'])
                points = segment['poly']
                '''SAMPLE
                "points": [[164.0, 14.0], [614.0, 14.0], [614.0, 39.0], [164.0, 39.0]]

                x0,y0 [164.0, 14.0]                     x1,y1 [614.0, 14.0]
                x3,y3 [164.0, 39.0]                     x2,y2 [614.0, 39.0]
                '''
                delta_x_upper = (points[1][0] - points[0][0]) / text_len
                delta_y_upper = (points[1][1] - points[0][1]) / text_len
                delta_x_lower = (points[2][0] - points[3][0]) / text_len
                delta_y_lower = (points[2][1] - points[3][1]) / text_len
                # segment['words'] is a temporary attribute, just for calculating every single word/character's bbox
                segment['words'] = []
                for i in range(0, text_len):
                    segment['words'].append( 
                        {
                            'box': 
                                [ 
                                    int(points[0][0]+delta_x_upper*i), int(points[0][1]+delta_y_upper*i), 
                                    int(points[3][0]+delta_x_lower*(i+1)), int(points[3][1]+delta_y_lower*(i+1)), 
                                ], 
                            'text': segment['text'][i], 
                        } 
                    )
                #print('segment[words]=', segment['words'])

            result = {
                "name": img_path,
                "text":         [ word                     for segment in ocr_result for word in segment["text"] ],
                "bbox":         [ word['box']              for segment in ocr_result for word in segment["words"] ],
                "segment_bbox": [ segment['segment_bbox']  for segment in ocr_result for word in segment["text"] ],
                "segment_id":   [ segment['id']            for segment in ocr_result for word in segment["text"] ],
                "width": width, 
                "height": height, 
                "qas": [
                            {
                                "question": "PAGE_HEADER", 
                                "answers": [  ]
                            },
                            {
                                "question": "PROFILE_KEY", 
                                "answers": [  ]
                            },
                            {
                                "question": "PROFILE_VALUE", 
                                "answers": [  ]
                            },
                            {
                                "question": "ITEM_NAME", 
                                "answers": [  ]
                            },
                             {
                                "question": "ITEM_VALUE", 
                                "answers": [  ]
                            },
                            {
                                "question": "SPEC", 
                                "answers": [  ]
                            },
                            {
                                "question": "UNIT", 
                                "answers": [  ]
                            },
                            {
                                "question": "UNIT_PRICE", 
                                "answers": [  ]
                            },
                            {
                                "question": "QUANTITY", 
                                "answers": [  ]
                            },
                            {
                                "question": "INSURANCE_CLASS", 
                                "answers": [  ]
                            },
                            {
                                "question": "INSURANCE_PERCENTAGE", 
                                "answers": [  ]
                            },
                            {
                                "question": "OTHER", 
                                "answers": [  ]
                            },
                        ],
                "image": "N/A",
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
