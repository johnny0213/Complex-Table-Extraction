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
import sys 
sys.path.append("..") 
from utils.utils import draw_ner_results

PROJECT_BASE = r'D:\WORK\NER\\'
DATA_BASE = PROJECT_BASE + r'data\\'

if __name__ == "__main__":

    ocr_results = DATA_BASE + r'ocr_results.txt'

    test_results = DATA_BASE + r'test_predictions.json'

    '''
    ocr_results sample:
    D:\WORK\NER\data\12f3492fb3ef492e9288171207846d7a_0.jpg 
    {"ocr_result": 
        [
            {"text": "复方磺胺甲晞壁片I山东新华1400mg：80mg*100", "bbox": [546.0, 461.0, 1126.0, 501.0], "poly": [[546.0, 461.0], [1126.0, 474.0], [1125.0, 501.0], [546.0, 488.0]]}, 
            {"text": "28", "bbox": [1361.0, 484.0, 1393.0, 510.0], "poly": [[1361.0, 484.0], [1393.0, 484.0], [1393.0, 510.0], [1361.0, 510.0]]}, 
            {"text": "0.25", "bbox": [1512.0, 487.0, 1573.0, 517.0], "poly": [[1514.0, 487.0], [1573.0, 491.0], [1571.0, 517.0], [1512.0, 514.0]]}, 
            ...
        ]
    }

    ernie-layout NER test results sample:
    {
    "D:\\WORK\NER\data\0c8c37092b874f4e8b789ec79e8fb12b_11.jpg_rotated.png": [
        [
            "OTHER",
            "血液内科一病",
            0.9998577733834585,
            [
                0,
                5
            ],
            "B-OTHER, I-OTHER, I-OTHER, I-OTHER, I-OTHER, I-OTHER"
        ],
        [
            "OTHER",
            "血液内科",
            0.9999558329582214,
            [
                6,
                9
            ],
            "B-OTHER, I-OTHER, I-OTHER, I-OTHER"
        ],
	}
    '''

    file = open(ocr_results, 'r', encoding='utf-8')
    lines = [ line for line in file ]
    file.close()
    print('ocr_results len=', len(lines))

    file = open(test_results, 'r', encoding='utf-8')
    test_results = json.load(file)
    file.close()

    test_results = [ v for k, v in test_results.items() ]
    print('test results len=', len(test_results))
    
    ocr_results_with_label = []
    for idx, (line, test_results_ERNIE_LAYOUT) in enumerate(zip(lines, test_results)):

        img_path = line.split('\t')[0]
        print('now visualize image=', img_path)
        save_img_path = img_path + "_ner.jpg"

        ocr_results = json.loads( line.split('\t')[1] )
        ocr_results = ocr_results['ocr_result']

        print('len(ocr_results)=', len(ocr_results))
        print('len(test_results_ERNIE_LAYOUT)=', len(test_results_ERNIE_LAYOUT))
        if len(ocr_results) != len(test_results_ERNIE_LAYOUT):
            print('\t\tlen(test_results_ERNIE_LAYOUT) is inconsistent with len(ocr_results)!!!')
        
        #test_results_ERNIE_LAYOUT must ALWAYS be cleaned up!
        print('Whether len(test_results_ERNIE_LAYOUT) is inconsistent with len(ocr_results) or not, test_results_ERNIE_LAYOUT MUST ALWAYS be cleaned up so as to align with ocr_results!!!')
        '''
        Ernie Layout test results sample:
        0 ['清单'，]
        1 ['总',]
        2 ['金',]
        3 ['额',]
        4 ['姓名',]

        ocr_results sample:
        0 ['清单'，]
        1 ['总金额',]
        2 ['姓名',]

        chars:
        ['清'，'单'，'总','金','额','姓','名',]

        test_result_char_indexes:
        [  0,   0,    1,   2,   3,  4,   4]

        ocr_result_char_indexes:
        [  0,   0,    1,   1,   1,  2,   2]
        '''
        
        test_result_char_indexes = []
        for idx, test_result in enumerate(test_results_ERNIE_LAYOUT):
            for char in test_result[1]:
                test_result_char_indexes.append(idx)
        ###print('test_result_char_indexes len=', len(test_result_char_indexes), '\t value=', test_result_char_indexes)

        ocr_result_char_indexes = []
        for idx, ocr_result in enumerate(ocr_results):
            for char in ocr_result['text'].strip(): #BUG FIX: strip()
                ocr_result_char_indexes.append(idx)
        ###print('ocr_result_char_indexes len=', len(ocr_result_char_indexes), '\t value=', ocr_result_char_indexes)

        try:
            test_results_ERNIE_LAYOUT_cleanup = []
            for i in range(0, len(ocr_results)):
                test_results_ERNIE_LAYOUT_cleanup.append( test_results_ERNIE_LAYOUT[ test_result_char_indexes[ ocr_result_char_indexes.index(i) ] ] )

            test_results_ERNIE_LAYOUT = test_results_ERNIE_LAYOUT_cleanup 
        except Exception as e:
            print('\t\tERROR: test_results_ERNIE_LAYOUT cleanup, ', e)
            print('\t\t\t\tThis image is SKIPPED!!!')
            continue # SKIP THIS IMAGE

        img = cv2.imread(img_path)
        height = img.shape[0] #高度
        width = img.shape[1] #宽度

        img_res = draw_ner_results(img, ocr_results, test_results_ERNIE_LAYOUT) #ocr_results['pred'] will be added  
        cv2.imwrite(save_img_path, img_res)

        ocr_results_with_label.append(img_path + '\t' + json.dumps({ 'ocr_result': ocr_results}, ensure_ascii=False ) + '\n')


    file = open(PROJECT_BASE + 'ocr_results_with_label.txt', 'w', encoding='utf-8')
    file.write(''.join(ocr_results_with_label))
    file.close()