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
    ocr result sample:
    D:/data/881e8ff3-7799-495d-af33-0c9f51cdb7b4.png {"oer_result": [{"text": "3", "bbox": [820.0, 120.0, 852.0, 163.0], "poly": [[820.0, 120.0], [852.0, 120.0], [852.0, 163.0], [820.0, 163.0]]}, {"text": "65", "bbox": [705.0, 186.0, 785.0, 243.0], "poly": [[705.0, 186.0], [785.0, 186.0], [785.0, 243.0], [705.0, 243.0]]}, {"text": "CCTV6", "bbox": [177.0, 907.0, 334.0, 942.0], "poly": [[177.0, 907.0], [334.0, 907.0], [334.0, 942.0], [177.0, 942.0]]}, {"text": "电", "bbox": [218.0, 950.0, 248.0, 974.0], "poly": [[218.0, 950.0], [248.0, 950.0], [248.0, 974.0], [218.0, 974.0]]}, {"text": "58151", "bbox": [1308.0, 1390.0, 1342.0, 1397.0], "poly": [[1308.0, 1390.0], [1342.0, 1390.0], [1342.0, 1397.0], [1308.0, 1397.0]]}, {"text": "1", "bbox": [593.0, 1826.0, 617.0, 1858.0], "poly": [[596.0, 1826.0], [617.0, 1829.0], [614.0, 1858.0], [593.0, 1855.0]]}]}
    '''
    file = open(DATA_BASE + 'ocr_results.txt', 'r', encoding='utf-8')
    lines = [ line.strip() for line in file ]
    file.close()
    print('lines len=', len(lines))

    file = open(PROJECT_BASE + 'test.json', 'w', encoding='utf-8')
    for idx, line in enumerate(lines):

        img_path = line.split('\t')[0]

        img = cv2.imread(img_path)
        height = img.shape[0] #高度
        width = img.shape[1] #宽度

        ocr_result = json.loads(line.split('\t')[1])
        ocr_result = ocr_result['ocr_result']
        '''
        ocr_result sample:
         [
             {"text": "手术时间", "bbox": [691.0, 51.0, 1209.0, 132.0], "poly": [[695.0, 51.0], [1209.0, 90.0], [1206.0, 132.0], [691.0, 93.0]]}, 
             {"text": "手术医师签名", "bbox": [675.0, 112.0, 1236.0, 195.0], "poly": [[679.0, 112.0], [1236.0, 153.0], [1233.0, 195.0], [675.0, 154.0]]}, 
         ]
        '''
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
                    "answers": [{"text": "0", "answer_start": -1, "answer_end": -1}]
                    }],
            "image": "N/A",
        }

        #all_results += json.dumps(result, ensure_ascii=False) + "\n"
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

    file.close()