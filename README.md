In the health insurance business, a patient will collect all his or her medical records, which comprise a broad variety of documents and are usually taken photos with the patient's mobile phone, and submit them to an insurance company to claim medical expenses. The insurer will then process the document images, retrieve data, screen each expense item as per insurance terms, and tally the total medical expenses.
 
Automatic processing and retrieving data from medical document images with high accuracy(90%+) has long been a tough task for the industry. Last year we had done a series of experimental projects that were aimed to tackle the task and showed encouraging results. 
 
We have employed a pipeline approach instead of an end-to-end approach. Our pipeline comprises the following sub-tasks:
1. Image Orientation
2. OCR (for General Purpose)
3. Distortion Correction
4. Text-Image Classification
5. Re-OCR (for Bill-of-Expenses)
6. NER
7. Table Restoration
 
See utils/Figure 1.
 
A pipeline approach is tedious and less efficient than an end-to-end approach, whereas it gives us more control over each sub-task which can be independently optimized and be re-used in other projects. 
 
We have built the pipeline on the Baidu's Deep Learning stack, namely:
  PddalePaddle-gpu v2.2.2.post111,
  PaddleNLP v2.4,
  Ernie-layout,
  PaddleOCR v2.6.
 
1. Text_Image_Orientation
 
Our approach is to fine-tune the PaddleClas PULC text_image_orientation pretrained model to get our classifier. 
We tested our document images on the PaddleClas PULC text_image_orientation pretrained model, which gave a unsatisfactory result of 80% accuracy. Therefore we cannot use it out-of-the-box and will have to fine-tune it with our document images. 
 
1.1 Installation:
Please install PaddleClas(2022.6.15 Release) on your machine following its instructions.
 
1.2 Data Preparation:
We have collected 12181 document images in upright position, then rotated them to 90/180/270 degrees respectively, and finally resized them to a specific size:
Use ./preprocess_img.py to generate train and eval datasets.
 
1.3 Configs:
Use ./ppcls/configs/PULC/text_image_orientation/PPLCNet_x1_0.yaml.
Please note that batch_size and learning_rate must be adjusted in proportion to recommended values(batch_size=64, learning_rate=0.1). For example, we used batch_size=56, learning_rate=0.875.
Please note that -o Arch.pretrained=True means that the pretrained model will be used and downloaded to the local cache automatically.
 
1.4 Train:
Use ./train.bat.
After 40 epochs, we got eval accuracy of 0.9912.
 
1.5 Infer:
Use ./infer.bat.
Prediction results are shown in the command-line output.
On a small test dataset, we got accuracy around 0.99.
 
You can refer to the links below for more information:
https://github.com/PaddlePaddle/PaddleClas
https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/PULC/PULC_quickstart.md
https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_text_image_orientation.md
https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/PULC/PULC_model_list.md
https://blog.csdn.net/qq_15821487/article/details/124391651
https://blog.csdn.net/libo1004/article/details/118522864
https://www.sohu.com/a/561138525_827544
 
 
2. OCR (for General Purpose)
 
We have fine-tuned PaddleOCR v2.6 DET/REC pretrained models to get our general-purpose OCR models.
 
One lesson learned is that it is not recommended to use synthesized images in your train dataset. We synthesized a great amount of images with the tool TextRenderer, expecting to improve REC accuracy with more train data. However we found that the eval accuracy of the REC model plateaued after just several epochs, and the model seemed over-fitting to the synthesized data. Later on test results on real-world images were barely acceptable.
 
The underlying reason might be that our document images with printed texts are very complicated in terms of color, texture and font with random distortions, making it very difficult for image synthesization tools to emulate them. Synthesized images different from real-world ones will do more harm than good to the REC model.
 
Another lesson learned is that it is difficult for the DET model to detect single-character textboxes, i.e. textboxes comprising just one character. Even though they are detected, the DET model often gives bboxes with wrong orientation, for example, heavily tilted. The only solution is to annotate more single-character textboxes and then fine-tune the pretrained DET model.
 
We will discuss OCR further in section 5. Re-OCR (for Bill-of-Expenses).
 
 
3. Distortion_Correction
 
Even after image orientation, our document images are always distorted in one way or another, significantly damaging the performances of the downstream tasks such as DET, REC and NER. Image Distortion Correction is a very big yet hard topic. We once tried a CV-based DNN model but to no avail. Our simplified approach is to use textboxes' orientation in a document image as an indicator of distortion, and then rotate the image to a certain degree to get the textboxes upright.
 
We first tried to look for the top N in terms of width in all textboxes of an image, calculate their average slope, and then rotate the image to get it upright. However we found that in some cases an image was distorted so heavily that the upper half turned clockwise and the lower half turned counter-clockwise (or vice versa). Later we decided to take both halves into consideration, looking for the top N textboxes in the two halves respectively and then calculating their average slopes.
 
Please note that the following PaddleOCR parameters are very sensitive to and suitable for the document images we are concerned about, and do not make any changes to them unless you are very confident to do so:
 
  # # DB parmas (for DET)
  det_db_thresh=0.1,  
  det_db_box_thresh=0.,             
  det_db_unclip_ratio=1.5,
  det_limit_type="min", 
  det_limit_side_len=736,
 
  # # EAST parmas (for REC)
  det_east_score_thresh=0.4,
  det_east_cover_thresh=0.,
  det_east_nms_thresh=0.1,
 
  rec_image_shape='3, 32, 320', 
 
 
4. Text_Image_Classification
 
We have classified document images into 21 classes:
1 ID Cards,
2 Bank Cards,
3 Insurance Claim Forms,
4 Invoices,
5 Medical Record Coverpages,
6 Medical Orders,
7 Surgical Records,
8 Pathological Reports, 
9 Diagnostic Reports,
10 Admission Notes,
11 Exam & Test Reports,
12 Discharge Notes,
13 Bill of Expenses,
14 ...,
15 ..., 
16 ...,
17 ...,
18 ...,
19 ...,
20 ...,
0  Other
 
Each class may actually comprise several sub-classes, and there are more than 100 sub-classes in total. Differences between certain classes or sub-classes may be minor in terms of document appearance, layout or text. Therefore it is a tough task to classify them with high accuracy.
 
We have experimented with several approaches:
a. Text Classifier: Extract texts from the images and fine-tune BERT/Roberta to get our text classifier. The test accuray was around 0.85.
 
b. Multi-modal Text-Image Classifier: Extract visual features with an ImageEncoder(Resnet152), then fuse them with text features. See https://github.com/huggingface/transformers/tree/master/examples/research_projects/mm-imdb for more information. The test accuracy was around 0.90.
 
c. Multi-modal Text-Image Classifier: Fine-tune Ernie Layout pretrained model(ernie-layoutx-base-uncased) which fuses an image's visual, layout and text features. Test accuracy shows significant improvements on previous classifiers.
 
4.1 Installation:
Please install PaddleNLP v2.4.2 on your machine following its instructions.
Our work is adapted from Ernie-layout's RVL-CDIP sample.
 
4.2 Data Preparation:
We have collected 3849 document images of all 21 classes. 
First, run OCR for all the images, and save results in file ./data/ocr_results.txt.
Then, manually label all the images in file ./data/labels.txt.
Finally, run generate_cls_trainset.py to generate the train and eval datasets.
 
Please note that in cls.txt:
 
a. We use the attribute 'name', which is the full path-name of an image, to find and open an image, instead of the attribute 'image'(which is BASE64-encoded string of an image in Ernie-layout's RVL-CDIP sample).
 
b. We use 'bbox' in place of 'segment_bbox', which simplifies the data preparation while does no harm to the classification model. 
 
4.3 Train:
First, remove the compiled train and dev datasets in your compiled data cache. You can check the positions of the data cache and the compiled data cache in the command-line output when you run Ernie Layout rvl-cdip sample. On our Windows machine, it is like C:/Users/Administrator/.cache/huggingface/datasets/rvl_cdip_sampled/rvl_cdip_sampled/1.0.0/.
Then, copy the train and dev datasets to the data cache. On our Windows machine, it is like C:/Users/Administrator/.cache/huggingface/datasets/downloads/extracted/5af65cb9ee4ab39346983016db30b753ee3791e4303df2d445aacd16012293d1/rvl_cdip_sampled.
At last, run run_cls.bat.
After 16 epochs, we got eval accuracy of 0.9657.
 
4.4 Test:
First, run OCR on your test images, and save results to ocr_results.txt
Then run generate_cls_testset.py to generate the test dataset.
Then, remove the compiled test datasets in your compiled data cache. 
Then, copy the test datasets to the data cache. 
At last, run test_cls.bat.
Prediction results are in the file ./ernie-layoutx-base-uncased\models\rvl_cdip_sampled\test_predictions.json.
On a small test dataset, we got accuracy around 0.95.
 
Please note that we haven't used ./deploy/python/infer.py, which is recommended by the Ernie-layout's RVL-CDIP sample, but we think is not as straightforward as ours. You may try it for yourself as something like infer.py will have to be used in a production environment.
 
You can refer to the links below for more information:
https://github.com/PaddlePaddle/PaddleNLP
https://github.com/PaddlePaddle/PaddleNLP/model_zoo/ernie-layout
https://huggingface.co/datasets/rvl_cdip
https://adamharley.com/rvl-cdip/
 
 
5. Re_OCR (for Bill-of-Expenses)
 
Bills of Expenses are the most complicated and valuable documents among all the document image classes. In order to guarantee the best OCR accuracy, we have decided to fine-tune the pretrained PaddleOCR DET/REC models to get our OCR models exclusively for Bill-of-Expenses.
 
5.1 Installation:
Please install PaddleOCR v2.6 on your machine following its instructions.
 
5.2 Annotation:
Run "PPOCRLabel", the annotation tool provided by PaddleOCR.
We have annotated 328 images with 80892 textboxes in total.
 
Please note that PaddleOCR DET models are very sensitive to the bboxes you draw when you annotate. It is highly recommended that:
 
a. Guarantee the highest standards of annotation. Make sure annotators draw a bbox that precisely enclose a textbox, not too small or too big. The DET model will learn precisely what you teach it and draw exactly the same kind of bboxes in prediction. For instance, if a bbox is too small, the edge parts of a textbox may be cut off, leading to the REC model recognizing wrong texts in downstream tasks.
 
See utils/Figure 2.
 
b. Guarantee consistency of annotation among different annotators and batches. Make sure different annotators use exactly the same standards over different periods of time.
 
5.3 DET Config:
Refer to ./configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml
Please note that we don't use synthesized images in the train dataset. Instead, we use real-world images from 3rd-party datasets such as wildreceipts and XFUND, while set sampling ratios as 0.1 in training:
    label_file_list:
      - ./data/train/Label.txt
      - ./data/wildreceipt/wildreceipt_test.txt
      - ./data/wildreceipt/wildreceipt_train.txt
      - ./data/XFUND/zh_train/image/Label.txt
      - ./data/XFUND/zh_val/image/Label.txt
    ratio_list: [1.0, 0.1, 0.1, 0.1, 0.1]
 
We don't use images from the section 2 OCR(for General Purpose) dataset because its annotation is somewhat low in quality and inconsistent with this dataset. If you can find other high-quality 3rd-party datasets, you can use them to augment your train dataset and expect improved accuracy.
We use the script generate_ppocrlabel_from_xfund.py to convert XFUND dataset into a trainset.
 
5.4 DET Train:
PaddleOCR provides a script for training.
python tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml  
We got the best eval metric (hmean=0.9574) at epoch 294.
 
5.5 DET Infer:
python tools/export_model.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o Global.pretrained_model="./output/ch_PP-OCR_V3_det/best_accuracy"  Global.save_inference_dir=./inference/ch_PP-OCRv3_det_student/
python tools/infer/predict_det.py --det_algorithm="DB" --det_model_dir="./inference/ch_PP-OCRv3_det_student/" --image_dir="D:/WORK/Re_OCR/data/test/" --use_gpu=True --det_db_thresh=0.1 --det_db_box_thresh=0.0 --det_db_unclip_ratio=1.5   --det_limit_type="min" --det_limit_side_len=736
 
5.6 REC Config:
Refer to ./configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml
Again we don't use synthesized images in the train dataset. Instead, we use real-world images from 3rd-party datasets such as XFUND, while set sampling ratios as 0.5 in training:
    label_file_list:
      - ./data/train/rec_gt.txt
      - ./data/XFUND/zh_train/image/rec_gt.txt
      - ./data/XFUND/zh_val/image/rec_gt.txt
    ratio_list: [1.0, 0.5, 0.5]
 
5.7 REC Train:
python tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml 
We got the best eval metric (acc=0.9388) at epoch 115.
 
5.8 REC Infer:
python tools/export_model.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml -o Global.pretrained_model=./output/rec_ppocr_v3_distillation/best_accuracy  Global.save_inference_dir=./inference/ch_PP-OCRv3_rec_distillation/
 
python tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./your inference model" --rec_image_shape="3, 32, 320" --rec_char_dict_path="your text dict path"
 
5.9 DET+REC infer:
Use our scripts here to do batch OCR for all images in a folder:
python generate_OCR_results.py
Just in case the script runs successfully but no OCR results are not generated, please downgrade the packages below:
pip install paddleocr==2.4
pip install paddlenlp==2.2.2
The exception might be due to some legacy methods from older versions being used in the scripts.
 
You can refer to the links below for more information:
https://github.com/PaddlePaddle/PaddleOCR
https://github.com/paddlepaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/detection.md
https://github.com/paddlepaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/recognition.md
https://github.com/paddlepaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/finetune.md
https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/models_list_en.md
https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_en/ppocr_introduction_en.md
https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5/PPOCRLabel
https://blog.csdn.net/YY007H/article/details/120646395
https://blog.csdn.net/YY007H/article/details/120650155
 
 
6. NER
 
Bills of Expenses are structurally complicated tables usually with dozens of lines and hundreds of textboxes. The formats of Bill of Expenses vary from one hospital to another. There are hundreds of different formats in thousands of hospitals across the country. The objective is to retrieve all textboxes, and restore all columns and all lines, and then to output them in a tabular format. Later on the restored date will be used for further downstream tasks such as tallying the total medical expenses. 
 
Over years a lot of table recognition algorithms have emerged yet none has solved the task convincingly. The options are:
 
a. Template-based: Design templates for different document formats respectively, and retrieve data as per the templates. Which is labor-intensive and cumbersome, and is difficult to generalize.
 
b. CV-based: Use sophisticated CV models to recognize table structures, then retrieve data as per the table structure. Which is technically complicated.
 
c. OCR-based: Use OCR to get all the textboxes, then use rules or algorithms such as clustering to recognize columns and lines. Which entails a great amount of rules, and is difficult to maintain or generalize.
 
d. Multi-modal-based: Train NER/RE models by fusing a document image's visual, layout and text features, do NER/RE on the document, and retrieve data on top of the NER/RE results. Which is comparatively neat and shows promising results.
 
	d1. PPStructure LayoutXLM NER: We fine-tuned the pretrained LayoutXLM model. We tested on a small size dataset, and the performance was fairly good but worse than d3. 
 
	d2. PPStructure Table Recognition: We presume that the pretrained Table REC model will not suffice, hence fine-tuning is necessary. You may try it for yourself.
 
	d3. Ernie-layout NER: Fine-tune Ernie Layout pretrained model(ernie-layoutx-base-uncased) to get our NER model.
 
	d3. UIE-X: We haven't tried it. We don't know if the pretrained model suffices or fine-tuning is needed. You may try it for yourself.
 
We have chosen option d2. Our aim is to extract 12 classes of Named Entities, eight of which are in the table of a Bill-of-Expenses document:
	ITEM_NAME,
	ITEM_VALUE
	SPEC(Specification)
	UNIT
	UNIT_PRICE
	QUANTITY
	INSURANCE_CLASS
	INSURANCE_PERCENTAGE(Medical Insurance Reimbursement Percentage)
 
And three of which are in the profile of a Bill-of-Expenses document:
	PAGE_HEADER
	PROFILE_KEY
	PROFILE_VALUE
 
Plus one class for all textboxes that are of no interest to us:
	OTHER
 
See utils/Figure 3.
 
6.1 Installation:
Please install PaddleNLP v2.4.2 on your machine following its instructions.
Our work is adapted from Ernie-layout's XFUND sample. 
 
6.2 Data Preparation:
We have collected 303 Bill-of-Expenses images with about 75000 textboxes. 
First, run generate_OCR_results.py in section 5 Re_OCR to get OCR results for all the images, save the results in file ./data/ocr_results.txt.
Then, run generate_ppocrlabel_from_ocr_results.py to get PPOCRLabel annotation, save the annotation in file Label.txt and fileState.txt.
Then, run "PPOCRLabel --kie", to manually add NER annotation to the images.
At last, run generate_ernie_layout_ner_trainset_from_ppocrlabel.py to generate NER train (and dev) datasets, save them in train.json (and dev.json).
Please note that the Ernie Layout NER dataset requires a special format below:
A textbox in OCR is a segment in Ernie_layout_NER.
    "text": ["单", "个", "字", ...],								# EVERY SINGLE WORD/CHARACTER's text. Caution: This is NOT an OCR textbox.
    "bbox": [[1956, 143, 2179, 196], [2065, 185, 2100, 213],...], # EVERY SINGLE WORD/CHARACTER's bbox. Caution: This is NOT an OCR bbox.
    "segment_bbox": [[789, 41, 880, 56], [827, 52, 875, 61],... ],# EVERY TEXTBOX's bbox, SCALED TO [1000, 1000]. Caution: This IS an OCR bbox.
    "segment_id": [0, 1, 1, 2, 2, 2, 2, 3, ...],                 # the id of the TEXTBOX that EVERY WORD/CHARACTER belongs to
 
See utils/Figure 4.
 
6.3 Train:
First, remove the compiled train and dev datasets in your compiled data cache. You can check the positions of the data cache and the compiled data cache in the command-line output when you run Ernie Layout XFUND sample. On our Windows machine, it is like C:\Users\Administrator\.cache\huggingface\datasets\xfund_zh\xfund_zh\1.0.0.
Then, copy the train and dev datasets to the data cache. On our Windows machine, it is like C:\Users\Administrator\.cache\huggingface\datasets\downloads\extracted\91c713cefe82c5d282b49f94e27cff534c4b75ab9360b8891a3dbb9086cd2871\xfund_zh.
At last, run run_ner.bat. If you want to resume from a previous checkpoint, run run_ner_resume.bat.
After 26 epochs, we got eval f1 of 0.98. However, test f1 was lower than that, which means the model seemed overfitting to the train dataset. A larger train dataset with more images may help improve the accuracy.
 
6.4 Test:
First, run OCR on your test images, and save results to ocr_results.txt
Then run generate_ernie_layout_ner_testset_from_ocr_results.py to get the test dataset.
Then, remove the compiled test dataset in your compiled data cache. On our Windows machine, it is like C:\Users\Administrator\.cache\huggingface\datasets\xfund_zh\xfund_zh\1.0.0.
Then, copy the test dataset to the data cache. On our Windows machine, it is like C:\Users\Administrator\.cache\huggingface\datasets\downloads\extracted\91c713cefe82c5d282b49f94e27cff534c4b75ab9360b8891a3dbb9086cd2871\xfund_zh.
At last, run test_ner.bat.
Prediction results are in the file ./ernie-layoutx-base-uncased\models\xfund\test_predictions.json.
On a small test dataset, test f1 was around 0.90.
 
Please note that:
 
a. We haven't used ./deploy/python/infer.py, which is recommended by the Ernie-layout's RVL-CDIP sample, but we think is not as straightforward as ours. You may try it for yourself as something like infer.py will have to be used in a production environment. 
 
b. In case you use an earlier version of Ernie-layout, you may notice the inference results may differ from last time even with the same test data. Please refer to: https://github.com/PaddlePaddle/PaddleNLP/issues/4253 for a solution.
 
6.5 Visualize:
Draw the textboxes with different colors illustrating their classes respectively. 
 
Please note that:
 
a. NER results do not often align with OCR results, i.e. a textbox in OCR results may be split into two entities in NER results. Therefore NER results must be cleaned up to make it 100% align with OCR results.
 
b. In rare cases, some specific textboxes in OCR results, such as 'HOURS', may be missing in NER results for unknown reasons, which makes it impossible for us to align NER results with OCR result. Here we just simply report an exception and skip the whole document image. 
 
You may want to have a taste of the visualization samples we have produced in the ..\Table_Restoration\data folder.
 
You can refer to the links below for more information:
https://github.com/PaddlePaddle/PaddleNLP
https://github.com/PaddlePaddle/PaddleNLP/model_zoo/ernie-layout
https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/kie.md
https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.6/ppstructure/docs/PP-Structurev2_introduction.md
https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5/ppstructure/vqa
https://github.com/PaddlePaddle/PaddleNLP/issues/4253
https://github.com/doc-analysis/XFUND
 
 
7. Table_Restoration
 
In order to restore a table in a Bill of Expenses document, let's think step by step:
 
7.1 Restore columns: Determine what columns are in the table, and which textboxes are in the columns respectively.
 
Ernie-layout NER has shown us all the textboxes we are interested in, i.e. with the labels below:
	ITEM_NAME
	ITEM_VALUE
	SPEC
	UNIT
	UNIT_PRICE
	QUANTITY
	INSURANCE_CLASS
	INSURANCE_PERCENTAGE
 
and has eliminated those we are not interested in, i.e. with the labels below:
	PAGE_HEADER
	PROFILE_KEY
	PROFILE_VALUE
	OTHER
 
Therefore we just simply group the textboxes into columns as per textboxes' NER labels, then we get all the columns and their textboxes respectively.
 
7.2 Restore lines: Determine what textboxes are on the same line, which must include one and only one textbox with the label:
	ITEM_NAME
 
and include one or more textboxes with the labels:
	ITEM_VALUE
	SPEC
	UNIT
	UNIT_PRICE
	QUANTITY
	INSURANCE_CLASS
	INSURANCE_PERCENTAGE
 
Theoretically, DNN models may be used to recognize the lines in a table. We once tried PaddleOCR's LayoutXLM RE(Relation Extraction), unfortunately the test performance was not good. Except for the small size of our trainset, we thought it might be too difficult a task for a model to distinguish dozens of lines out of hundreds of textboxes in a crowded and noisy document image like ours. 
So we decided to use a simple-and-stupid rule-based approach, using a textbox's slope to search for its right-side textbox.
 
See utils/Figure 5.
 
a. Calculate each textbox's slope. In order to simplify the calculation, we just use the slope of the top line of a textbox to approximate the textbox's slope.
 
b. If a textbox has one character only, OCR can not always draw its poly/points properly, resulting in wrong orientation/slope. To mitigate this, the textbox before and after a single-character textbox are used to approximate its slope.
 
c. There are always errors, either big or small, in a textbox's orientation/slope. To mitigate this, we use the moving average of the slopes of a textbox's nearby textboxes in the same column. 
 
d. Get the candidates of right-side textboxes. Candidates are only in the immediate right two columns. Further right columns are not candidates.
 
e. Calculate the right_textbox's Y that should be if it is on the same line with left_textbox. Considering that the right-side textbox often deviates a little bit from its expected location, we use a parameter Y_THRESHOLD. If the right textbox's deviation is within the Y_THRESHOLD, then it is deemed on the same line with the left textbox.
 
Please note that the empirical value for Y_THRESHOLD is 20 for an document image with height of 2800 pixels, and must be adjusted linearly as per an image's actual height.
 
7.3 Output the restored lines and columns into a tabular form.
You may want to take a look at the output samples we have produced in the data folder.
 
 
 