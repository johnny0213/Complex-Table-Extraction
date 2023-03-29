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
import sys 
sys.path.append("..") 
from utils.utils import OCREngine, nextlevel

PROJECT_BASE = r'D:\WORK\Re_OCR\\'
DATA_BASE = PROJECT_BASE + r'data\test\\'
ocr_results = DATA_BASE + 'ocr_results.txt'  

if __name__ == "__main__":

    file_list = []
    file_list = nextlevel(PROJECT_BASE, file_list, [ '.png' ]) 
    
    print('\n\nOCR. file_list len=', len(file_list))
    
    ocr_engine = OCREngine()
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

