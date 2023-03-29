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

PROJECT_BASE = r'D:\WORK\NER\\'
DATA_BASE = PROJECT_BASE + r'data\\'

'''
SOURCE sample:
D:\WORK\NER\data\12f3492fb3ef492e9288171207846d7a_0.jpg	{"ocr_result": 
[
	{"text": "复方磺胺甲晞壁片I", "bbox": [546.0, 461.0, 1126.0, 501.0], "poly": [[546.0, 461.0], [1126.0, 474.0], [1125.0, 501.0], [546.0, 488.0]]}, 
	{"text": "28", "bbox": [1361.0, 484.0, 1393.0, 510.0], "poly": [[1361.0, 484.0], [1393.0, 484.0], [1393.0, 510.0], [1361.0, 510.0]]}, 
	{"text": "0.25", "bbox": [1512.0, 487.0, 1573.0, 517.0], "poly": [[1514.0, 487.0], [1573.0, 491.0], [1571.0, 517.0], [1512.0, 514.0]]}, 
] 
}

TARGET sample:
UPPER_FOLDER/zh_train_0.jpg   
[
	{
		"transcription": "复方磺胺甲晞壁片I", 
		"points": [[104, 114], [530, 114], [530, 175], [104, 175]], 
		"difficult": false, 
		"id": 1, 
		"linking": []
	}, 
'''

if __name__ == "__main__":

	f = open(DATA_BASE + 'ocr_results.txt','r',encoding='utf-8') #ocr_results_with_label_ENLARGED.txt
	lines = [ v for v in f ]
	f.close()

	print('line count=', len(lines))

	label_output = ''
	fileState_output = ''
	for line in lines:
		image_path_name=line.split('\t')[0]
		print('image_path_name=', image_path_name)
		image_name_o = line.split('\t')[0].split('mm-imdb\\')[1] 

		ocr_result = json.loads(line.split('\t')[1])
		textboxes = ocr_result['ocr_result']
		print('textboxes len=', len(textboxes))

		outputs = []
		for index, textbox in enumerate(textboxes):
			output = { 
						"transcription": 	textbox["text"],
						"points": 			textbox["poly"], 
						"difficult": 		False, 
			}

			outputs.append(output)

		print('outputs len=', len(outputs))
		label_output += image_name_o + '\t' + json.dumps( outputs, ensure_ascii=False) + '\n' #ensure_ascii=False可以消除json包含中文的乱码问题
		fileState_output += image_path_name + '\t0' + '\n'

	f = open(DATA_BASE + 'Label.txt','w',encoding='utf-8')
	f.write(label_output)
	f.close()

	f = open(DATA_BASE + 'fileState.txt','w',encoding='utf-8')
	f.write(fileState_output)
	f.close()

