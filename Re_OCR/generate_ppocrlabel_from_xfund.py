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

PROJECT_BASE = r'D:\WORK\Re_OCR\\'
DATA_BASE = PROJECT_BASE + r'data\\'

'''
SOURCE sample:
zh_train_0.jpg	
{"height": 3508, "width": 2480, "ocr_info": 
	[
		{
			"text": "汇丰晋信", 
			"label": "other", 
			"bbox": [104, 114, 530, 175], 
			"id": 1, 
			"linking": [], 
			"words": 
				[
					{"box": [110, 117, 152, 175], "text": "汇"}, 
					{"box": [189, 117, 229, 177], "text": "丰"}, 
					{"box": [385, 117, 426, 177], "text": "晋"}, 
					{"box": [466, 116, 508, 177], "text": "信"}
				]
		}, 
	]
}

TARGET sample:
UPPER_FOLDER/zh_train_0.jpg   
[
	{
		"transcription": "汇丰晋信", 
		"points": [[104, 114], [530, 114], [530, 175], [104, 175]], 
		"difficult": false, 
		"key_cls": "other"
		"id": 1, 
		"linking": []
	}, 
'''

if __name__ == "__main__":

	f = open(DATA_BASE + '/XFUND/zh_val/' + 'xfun_normalize_val.json','r',encoding='utf-8') 
	lines = [ v for v in f ]
	f.close()

	print('line count=', len(lines))

	all_output = ''
	for line in lines:
		image_path_name= DATA_BASE + '/XFUND/zh_val/image/' + line.split('\t')[0].split('\\')[-1]  
		print('image_path_name=', image_path_name)
		image_name_o = 'image/' + line.split('\t')[0].split('\\')[-1] 

		textboxes = json.loads(line.split('\t')[1])
		textboxes = textboxes['ocr_info']
		print('textboxes len=', len(textboxes))

		outputs = []
		for index, textbox in enumerate(textboxes):
			x0 = textbox["bbox"][0]
			y0 = textbox["bbox"][1]
			x3 = textbox["bbox"][2]
			y3 = textbox["bbox"][3]
			output = { 
						"transcription": 	textbox["text"],
						"key_cls": 			textbox["label"],
						"bbox": 			textbox["bbox"],
						"points": 			[[x0, y0], [x3, y0], [x3, y3], [x0, y3],],
						"difficult": 		False, 
						"id": 				textbox["id"],
						"linking": 			textbox["linking"],
			}

			outputs.append(output)

		all_output += image_name_o + '\t' + json.dumps( outputs, ensure_ascii=False) + '\n' #ensure_ascii=False可以消除json包含中文的乱码问题

	f = open(DATA_BASE + 'Label.json','w',encoding='utf-8')
	f.write(all_output)
	f.close()


