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

PROJECT_BASE = r'D:\WORK\Text_Image_Classification\\'
DATA_BASE = PROJECT_BASE + r'data\\'

if __name__ == "__main__":

    '''
    label sample:
    {"text": "主诉：诊断xx癌2天 现病史：（病史叙述者：患者本人及家属可靠程度：可靠）患者55岁", "img": "D:/WORK/Text_Image_Classification/data/0ff6ce27-6267-4b65-a1c8-282c8410cd59.png", "label": "1"}
    '''
    file = open(DATA_BASE + 'labels.txt', 'r', encoding='utf-8')
    lines = [ line for line in file ]
    file.close()
    print('original dataset len=', len(lines))

    '''
    ocr result sample:
    D:/WORK/Text_Image_Classification/data/0ff6ce27-6267-4b65-a1c8-282c8410cd59.png {"ocr_result": [{"text": "3", "bbox": [820.0, 120.0, 852.0, 163.0], "poly": [[820.0, 120.0], [852.0, 120.0], [852.0, 163.0], [820.0, 163.0]]}, {"text": "65", "bbox": [705.0, 186.0, 785.0, 243.0], "poly": [[705.0, 186.0], [785.0, 186.0], [785.0, 243.0], [705.0, 243.0]]}, {"text": "CCTV6", "bbox": [177.0, 907.0, 334.0, 942.0], "poly": [[177.0, 907.0], [334.0, 907.0], [334.0, 942.0], [177.0, 942.0]]}, {"text": "电", "bbox": [218.0, 950.0, 248.0, 974.0], "poly": [[218.0, 950.0], [248.0, 950.0], [248.0, 974.0], [218.0, 974.0]]}, {"text": "58151", "bbox": [1308.0, 1390.0, 1342.0, 1397.0], "poly": [[1308.0, 1390.0], [1342.0, 1390.0], [1342.0, 1397.0], [1308.0, 1397.0]]}, {"text": "1", "bbox": [593.0, 1826.0, 617.0, 1858.0], "poly": [[596.0, 1826.0], [617.0, 1829.0], [614.0, 1858.0], [593.0, 1855.0]]}]}   0
    '''
    file = open(DATA_BASE + 'ocr_results.txt', 'r', encoding='utf-8')
    ocr = [ line.strip() for line in file ]
    file.close()
    print('ocr len=', len(ocr))

    file = open(DATA_BASE + 'cls.json', 'w', encoding='utf-8')
    for idx, (line, ocr_result) in enumerate(zip(lines, ocr)):
        line = json.loads(line)
        img_path = line['img']
        label = line['label'] # Here is the LABEL
        text = line['text']
        print("process: [{}/{}], image file name: {}".format(
            idx, len(lines), img_path))

        if len(text) < 20:
            print('\tlen(text) < 20, skipped!!!')
            continue

        img = cv2.imread(img_path)
        height = img.shape[0] #高度
        width = img.shape[1] #宽度

        ocr_result = json.loads(ocr_result.split('\t')[1])
        ocr_result = ocr_result['ocr_result']

        result = {
            "name": img_path,
            "text": [ textbox["text"] for textbox in ocr_result ],
            "bbox": [ textbox["bbox"] for textbox in ocr_result ],
            "segment_bbox": [ textbox["bbox"] for textbox in ocr_result ],
            "segment_id": [ idx for idx in range(0, len(ocr_result)) ],
            "width": width, 
            "height": height, 
            "qas": [{
                    "question": "What is the document type?", 
                    "question_id": -1, 
                    "answers": [{"text": label, "answer_start": -1, "answer_end": -1}]
                    }],
            "image": "N/A",
        }

        file.write(json.dumps(result, ensure_ascii=False) + "\n")

    file.close()

    