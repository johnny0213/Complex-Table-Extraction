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
from copy import deepcopy
import sys
sys.path.append("..") 
from utils.utils import draw_lines

PROJECT_BASE = r'D:\WORK\NER\\'
DATA_BASE = PROJECT_BASE + 'data\\'
label_file = DATA_BASE + 'ocr_results_with_label.txt'  

IMAGE_HEIGHT_MAX = 2800 # Maximum height of an image, in pixels
X_THRESHOLD = 3         # A little bit X overlapping is allowed, in pixels
Y_THRESHOLD_MAX = 20    # A certain amount of Y overlapping is allowed, in pixels
WINDOW_SIZE = 5         # for calculating a textbox's slope_moving_average in a column 

# FOR DEBUGGING
DEBUG = False #True #
LEFT_TEXTBOX_ID = 90
RIGHT_TEXTBOX_ID = 88
IMAGE_NAME = '3ad0c443d128482993c4b6826ab3b565_3.jpg_rotated.png'

'''
COLUMNS sample:
{
    'ITEM_NAME': {
        'textboxes': [...], 
        'next_2_column_names': [UNIT, UNIT_PRICE], 
        'X0_mean': 338.0, 
        'X0_span': [0, 709.0285714285715]
    },
}
'''
COLUMNS = {}      

def init_columns():
    global COLUMNS 

    init_value = { 
                    'textboxes': [],            # all textboxes in this column
                    'next_2_column_names': [],  # this column's immediate right two columns
                    'X0_mean': -1,              # The average X0 of this column
                    'X0_span': []               # [X0_mean of the left column, X0_mean of the right column] is boundaries of this column
                }
    COLUMNS = { 
        'ITEM_NAME':        deepcopy(init_value),
        'ITEM_VALUE':       deepcopy(init_value),
        'UNIT':             deepcopy(init_value),
        'UNIT_PRICE':       deepcopy(init_value),
        'QUANTITY':         deepcopy(init_value),
        'SPEC':             deepcopy(init_value),
        'INSURANCE_CLASS':  deepcopy(init_value),
        'INSURANCE_PERCENTAGE': deepcopy(init_value),
        }

    return


def romove_outliers():
    global COLUMNS

    # get every column's X0_mean
    for column_name, column in COLUMNS.items():
        textboxes = column['textboxes']
        if len(textboxes) == 0 or len(textboxes) == 1 :
            continue

        X0_mean = np.mean( np.array([ textbox['bbox'][0] for textbox in textboxes ]) )
        print('\t\tcolumn_name=', column_name, '\tX0_mean=', X0_mean)
        column['X0_mean'] = X0_mean

    _COLUMNS = { key:value for key, value in COLUMNS.items() if value['X0_mean'] != -1 }
    _COLUMNS = sorted(_COLUMNS.items(), key=lambda x: x[1]['X0_mean']) # Now _COLUMNS is [(column_name, column)]

    # get every column's X0_span
    for idx, (column_name, column) in enumerate(_COLUMNS):
        sum = len(_COLUMNS)
        span_start = 0
        span_end = 10000
        if idx != 0:
            span_start = _COLUMNS[idx-1][1]['X0_mean']
        if idx != (sum-1):
            span_end = _COLUMNS[idx+1][1]['X0_mean']

        column['X0_span'] = [span_start, span_end]

    # remove outlying textboxes that are not within the X0_span of a column
    for (column_name, column) in _COLUMNS: 
        textboxes = []
        print('column_name=', column_name, '\tbefore cleanup len=', len(column['textboxes']), '\tcolumn[X0_span]=', column['X0_span'])
        for textbox in column['textboxes']: # 
            X0 = textbox['bbox'][0]
            if X0 > column['X0_span'][0] and X0 < column['X0_span'][1]:
                textboxes.append(textbox)

        column['textboxes'] = textboxes
        print('column_name=', column_name, '\tafter cleanup len=', len(column['textboxes']))

    COLUMNS = { key:value for (key, value) in _COLUMNS if len(value['textboxes']) > 0 }
    return


def restore_columns(textboxes):
    global COLUMNS
    init_columns() # They must be flushed before processing the next image

    # Group textboxes into columns as per textboxes' NER labels
    for idx, textbox in enumerate(textboxes):
        if textbox['pred'] in ['PAGE_HEADER', 'PROFILE_KEY', 'PROFILE_VALUE', 'OTHER'] :
            continue

        # If a textbox has one character only, OCR can not always draw its bbox properly, resulting in wrong orientation/slope. 
        # To mitigate this, the textbox before and after it are used to approximate its slope.
        if len(textbox['text']) == 1: 
            slope_before = textboxes[idx-1]['slope'] if idx>0 else textboxes[idx+1]['slope']
            slope_after  = textboxes[idx+1]['slope'] if (idx+1)<len(textboxes) else textboxes[idx-1]['slope']
            slope_new = (slope_before + slope_after) / 2
            print('\t\tRe-setting slope: OLD slope=', textbox['slope'], '\tNEW slope=', slope_new)
            textbox['slope'] = slope_new 

        column_name = textbox['pred']
        COLUMNS[column_name]['textboxes'].append(textbox)
    
    romove_outliers()

    COLUMN_NAMES = list( COLUMNS.keys() )
    for idx, (column_name, column) in enumerate(COLUMNS.items()):
        # get every column's immediate next 2 column names,
        sum = len(COLUMNS)
        if (idx+1) <= (sum-1):
            column['next_2_column_names'].append(COLUMN_NAMES[idx+1]) 
        if (idx+2) <= (sum-1):
            column['next_2_column_names'].append(COLUMN_NAMES[idx+2]) 
        print('column_name=', column_name, '\tnext_2_column_names=', column['next_2_column_names'])

        # There are always errors, either big or small, in a textbox's orientation/slope. 
        # To mitigate this, we use the moving average of the slopes of a textbox's nearby textboxes within a column. 
        slopes = np.array([ textbox['slope'] for textbox in column['textboxes'] ])
        slopes = np.append( slopes, np.ones(WINDOW_SIZE-1) * column['textboxes'][-1]['slope'] ) # append (N-1)*slope_last to the nparray
        window = np.ones(WINDOW_SIZE)
        slopes_moving_average = list( np.convolve(slopes, window, 'valid') / WINDOW_SIZE )
        for idx, textbox in enumerate(column['textboxes']): 
            textbox['slope_moving_average'] = slopes_moving_average[idx]

    return

'''
textbox sample:
[
    {
        "text": "医疗住院收资项据（电子）", 
        "bbox": [475.0, 7.0, 968.0, 41.0], 
        "poly": [[475.0, 7.0], [968.0, 10.0], [968.0, 41.0], [475.0, 38.0]], 
        "pred_id": 5, 
        "pred": "HEADER"
    }, 
    {
        "text": "0691181013", 
        "bbox": [967.0, 81.0, 1063.0, 98.0], 
        "poly": [[967.0, 81.0], [1063.0, 81.0], [1063.0, 98.0], [967.0, 98.0]], 
        "pred_id": 3, 
        "pred": "ANSWER"
    }, 
    ...
]
'''
'''
bbox sample:
X0,Y0
  -----------------------
 |                       |
 |                       |
 |                       |
  -----------------------
                        X1,Y1

poly sample:
            (poly[0][0], poly[0][1])     x0,y0  
                - _
               |    - _                                                                                                 
              |         - _
             |              - _                                                                                 
            |                   - _(poly[1][0], poly[1][1])  x1,y1
(poly[3][0], poly[3][1])           |    
x3,y3           - _               |                                         
                    - _          |                                      
                        - _     |
                            - _|
                               (poly[2][0], poly[2][1]) 
                               x2,y2                                                      

'''

def calculate_slope(textbox):
    left_x0 = textbox['poly'][0][0]
    left_y0 = textbox['poly'][0][1]
    left_x1 = textbox['poly'][1][0]
    left_y1 = textbox['poly'][1][1]

    left_x_delta = left_x1 - left_x0 
    left_y_delta = left_y1 - left_y0
    slope = left_y_delta/left_x_delta 

    return slope


# Search for every textbox's right side neighbour. If found, save them in pairs.
def find_right_side_textboxes(textboxes, Y_THRESHOLD):

    '''
    # pair: [left_textbox_id, right_textbox_id, used_flag]
    # used_flag will be used later in restore_lines(): False=unused, True=used
    pairs sample: 
    [
        [ 1, 2, False],
        [ 2, 3, False ],
        [ 3, 4, False ],
        [ 4, 5, False ],
        [ 5, -1, False ],
        ...
    ]
    '''
    pairs= [] 
    for column_name, column in COLUMNS.items(): # for every column in the table
        for left_textbox in column['textboxes']: # for every textbox in a column
            RIGHT_TEXTBOX_ID = -1 # -1 means that no right side textbox has been found.

            left_X1 = left_textbox['bbox'][2]
            left_x0 = left_textbox['poly'][0][0]
            left_y0 = left_textbox['poly'][0][1]
            left_x1 = left_textbox['poly'][1][0]
            left_y1 = left_textbox['poly'][1][1]

            slope = left_textbox['slope_moving_average']

            if DEBUG and left_textbox['id'] == LEFT_TEXTBOX_ID :
                print('left_textbox[slope]=', left_textbox['slope'], '\tleft_textbox[slope_moving_average]=', left_textbox['slope_moving_average'])

            # Candidate are only the immediate right two columns. Further right columns are not disregarded.
            candidates = []
            for column_name in column['next_2_column_names']:
                candidates += COLUMNS[ column_name ]['textboxes']

            if DEBUG and left_textbox['id'] == LEFT_TEXTBOX_ID :
                print('right candidate column names=', column['next_2_column_names'], '\tright candidate textboxes len=', len(candidates))
            
            for right_textbox in candidates: # for every candidate textbox

                if right_textbox['used'] == True : #This textbox has been used
                    continue

                right_X0 = right_textbox['bbox'][0]
                right_x0 = right_textbox['poly'][0][0]
                right_y0 = right_textbox['poly'][0][1]
                right_x1 = right_textbox['poly'][1][0]
                right_y1 = right_textbox['poly'][1][1]

                if DEBUG and left_textbox['id'] == LEFT_TEXTBOX_ID :
                    print('now try right_textbox id=', right_textbox['id'], '\tright_textbox[text]=', right_textbox['text'], '\tright_textbox[used]=', right_textbox['used']) 

                # all left side textboxes to the left_textbox are SKIPPED!!!
                x_distance = right_X0 - left_X1
                if  x_distance < -X_THRESHOLD : 
                    continue

                # Calculate the right_textbox's Y that should be if it is on the same line with left_textbox
                right_Y_estimation = (right_x0 - left_x0) * slope  
                # Get the right textbox's real Y 
                right_Y_real       = right_y0 - left_y0 
                right_Y_deviation  = abs(right_Y_real - right_Y_estimation)
                if DEBUG and left_textbox['id'] == LEFT_TEXTBOX_ID:
                    print('\t\tslope=', slope, '\tright_Y_real=', right_Y_real, '\tright_Y_estimation=', right_Y_estimation, '\tright_Y_deviation=', right_Y_deviation)
                # If the right textbox's deviation is within the Y_THRESHOLD, then it is deemed on the same line with the left textbox
                if right_Y_deviation <= Y_THRESHOLD : 
                    if DEBUG and left_textbox['id'] == LEFT_TEXTBOX_ID:
                        print('FOUND: left id=', left_textbox['id'], '\t\tright id=', right_textbox['id'])
                    
                    right_textbox['used'] = True
                    RIGHT_TEXTBOX_ID = right_textbox['id']
                    break # found the right textbox, exit the loop.

            pairs.append([left_textbox['id'], RIGHT_TEXTBOX_ID, False]) 

    return pairs


# Concatenate adjacent textboxes from left to right to restore a line
def restore_lines(pairs):
    '''
    lines sample:
    [
        [ 1, 2, 3, 4, 5, -1 ],  
        [ 291, 293, 292, 294, -1 ], 
        [ 10, 12, 13, 14, 15, 16, 19, 21, -1 ],
        [ 18, -1 ]
        ...
    ]
    '''
    lines = []

    for pair_i in pairs: 
        if pair_i[2] == True: # this pair has been used
            continue

        lines.append([ pair_i[0], pair_i[1] ]) # start a new line

        right = pair_i[1] #current_right
        while right != -1: # NOT the end of line
            for pair_j in pairs:
                if pair_j[2] != True and right == pair_j[0]: # Unused, and current_right == incoming_left
                    lines[-1].append(pair_j[1])
                    right = pair_j[1]
                    pair_j[2] = True # this pair has been used

    return lines


if __name__ == "__main__":

    f = open(label_file , 'r', encoding='utf-8') 
    lines = [ v for v in f ]
    f.close()

    print('line count=', len(lines))

    id_output = ''
    text_output = ''
    for line_count, line in enumerate(lines):
        image_path_name= line.split('\t')[0]
        print('\n\nnow process image_path_name=', image_path_name)
        image_name_o = line.split('\t')[0].split('\\')[-1] 

        if DEBUG and image_name_o != IMAGE_NAME:  
            continue
        
        image = cv2.imread(image_path_name)
        height = image.shape[0] #高度
        width = image.shape[1] #宽度

        # Y_THRESHOLD must be adjusted in proportion to the height of the image!!!
        Y_THRESHOLD = Y_THRESHOLD_MAX * (height / IMAGE_HEIGHT_MAX) 
        print('Image height=', height, '\tY_THRESHOLD=', Y_THRESHOLD)

        ocr_info = json.loads(line.split('\t')[1])

        # Add more attributes to textboxes
        textboxes = ocr_info['ocr_result']
        for index, textbox in enumerate(textboxes):
            textbox['id'] = index
            textbox['used'] = False # To be used later in find_right_side_textboxes(): False=unused, True=used
            textbox['slope'] = calculate_slope(textbox)
        
        if DEBUG:
            print('textboxes[', LEFT_TEXTBOX_ID, '][bbox]=', textboxes[LEFT_TEXTBOX_ID]['bbox'], '\ttextboxes[', LEFT_TEXTBOX_ID, '][poly]=', textboxes[LEFT_TEXTBOX_ID]['poly'], '\t text=', textboxes[LEFT_TEXTBOX_ID]['text'], '\tpred=', textboxes[LEFT_TEXTBOX_ID]['pred'])
            print('textboxes[', RIGHT_TEXTBOX_ID, '][bbox]=', textboxes[RIGHT_TEXTBOX_ID]['bbox'], '\ttextboxes[', RIGHT_TEXTBOX_ID, '][poly]=', textboxes[RIGHT_TEXTBOX_ID]['poly'], '\t text=', textboxes[RIGHT_TEXTBOX_ID]['text'], '\tpred=', textboxes[RIGHT_TEXTBOX_ID]['pred'])
        
        restore_columns(textboxes)

        pairs = find_right_side_textboxes(textboxes, Y_THRESHOLD)
        draw_lines(pairs, textboxes, image_path_name)

        lines = restore_lines(pairs)
        lines.sort() #.sort() has no return value!!! just changed list object itself.
        id_output += image_name_o + '\t' + json.dumps( lines, ensure_ascii=False) + '\n' 

        # Write the restored table into a tabular form
        '''
        output sample:
        0c8c37092b874f4e8b789ec79e8fb12b_7.jpg_rotated.png
        ITEM_NAME   UNIT_PRICE  QUANTITY    ITEM_VALUE  INSURANCE_CLASS
        费用名称    单价  数量  应收金额    
        异常红细胞形态检查   1.5 3   4.5 
        红细原沉降水刺定（ESR）（化器法）  10  1   10  
        '''
        COLUMN_NAMES = list( COLUMNS.keys() )
        text_output += '\n' + image_name_o + '\n'
        text_output += '\t'.join(COLUMN_NAMES) + '\n'
        for textline in lines:
            line = [ 'PLACE_HOLDER' ] * len(COLUMN_NAMES)
            for id in textline[:-1] :
                column_id = COLUMN_NAMES.index( textboxes[id]['pred'] )
                line[column_id] = textboxes[id]['text']

            line = '\t'.join(line).replace('PLACE_HOLDER', '')
            text_output += line + '\n'


    # Output IDs of the textboxes that are on the same lines
    f = open(PROJECT_BASE + 'restored_lines.txt','w',encoding='utf-8')
    f.write(id_output)
    f.close()

    # Output all restored tables in .tsv format
    f = open(PROJECT_BASE + 'restored_tables.tsv','w',encoding='utf-8')
    f.write(text_output)
    f.close()
