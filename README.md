NIRPed-JointDetector (Implementation based on Tensorflow & Keras)
---
# Content
I. [NIRPed dataset](#数据创新)<br> 

II. [JointDetector](#联合检测方法创新)<br>

III. [Performance](#性能表现)<br>

IV. [References](#参考资料)<br> 

# I. NIRPed dataset <br>
There are training, validation and testing subset in NIRPed which doesn't need to be divided again. <br>
For being compatible with the existing framework, NIRPed's annotations are provided in the MS-COCO format (JSON). <br>
## A. Data of NIRPed <br>
PNG/JSON (Python)
Training images (60GB) /Training annotations (38MB)<br> 
Validation images (38GB) /Validation annotations (25MB)<br> 
Testing images (39GB) /Testing image information except annotations  (9MB)<br> 
###**Please use Google Chrome or Microsoft Edge to download the NIRPed dataset via: https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED%2FNIRPed**
  
## B. Data of miniNIRPed <br>
PNG/JSON (Python)
Training images (284MB)  /Training annotations (290KB) <br> 
Validation images (172MB)  /Validation annotations (183KB) <br> 
Testing images (177MB)   /Testing image information except annotations  (40KB) <br> 
###Please download the miniNIRPed dataset via: https://github.com/XiaobiaoDai/NIRPed/tree/JointDetector/data/miniNIRPed <br> 
###You can also use Google Chrome or Microsoft Edge to download the miniNIRPed dataset via: https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED%2FminiNIRPed
   
## C. License <br>
This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
That the dataset comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we do not accept any responsibility for errors or omissions.
That you include a reference to the NIRPed Dataset in any work that makes use of the dataset.
That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
You may not use the dataset or any derivative work for commercial purposes such as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
That all rights not expressly granted to you are reserved by us.

# II. JointDetector
NIRPed-JointDetector has been implemented based on Tensorflow & Keras in Python <br>
## A. Environment <br>
1. pythonn == 3.9 <br>
2. tensorflow-gpu == 2.9.0 <br>
3. keras == 2.9.0 <br>
4. Please refer to requirements.txt for more configuration. <br>

## B. Download <br>
### 1. Weights <br>
The required network weights (NIRPed_weights_resnet50.h) can be downloaded from our repository in GitHub. <br>
###Link: https://github.com/XiaobiaoDai/NIRPed/blob/master/model_data/NIRPed_weights_resnet50.h5 <br>
### 2. Data <br>
There are training, validation and testing subset in NIRPed which doesn't need to be divided again. <br>
###Link: https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED <br>

## C. How2train <br>
### 1. Training on NIRPed <br>
**Data preparation** <br>
   Before training, downloaded NIRPed training subset，and unzip images to the folder of "./data/NIRPed/images/train"; <br> 
   download COCO-format annotation train.json to the folder of "./data/NIRPed/labels". <br>
**Configuration** <br>
   Open "./keras_frcnn/config.py", modify self.train_img_dir to the training image path (./data/NIRPed/images/train); <br>
   Open "./keras_frcnn/config.py", modify self.train_anno to the training annotation path (./data/NIRPed/labels/train.json). <br>
**Begin training** <br>
   Run "train_JointDetector.py" to start training. <br>

### 2. Training on your own dataset <br>
**Data preparation**  <br>
   Collect the image and target distance information in the image, and make a COCO-format annotation file.  <br>
   Before training, put the png image files into the folder of "./data/yourDataset/images/train"; <br>
   put the annotation file train.json into the folder of "./data/yourDataset/labels". <br>   
**Configuration** <br>
   Open "./keras_frcnn/config.py", modify self.train_img_dir to the training image path (./data/yourDataset/images/train); <br>
   modify self.train_anno to the training annotation path (./data/yourDataset/labels/train.json); <br>
   modify self.class_mapping according to your tasks; <br>
   modify other parameters according to your tasks. <br>   
### 3. Begin training  <br>
   Run "train_JointDetector.py" to start training. During the training stage, weights will be saved in the folder of "./model_data". <br>

## D. How2predict <br>
### 1. Using our weights <br>
**Data preparation** <br>
   Before prediction, downloaded NIRPed validation or test subset，and unzip images to the folder of "./data/NIRPed/images/val" or "./data/NIRPed/images/test"; <br> 
   download COCO-format annotation val.json or test.json to the folder of "./data/NIRPed/labels"; <br>
   download optimized weight file (NIRPed_weights_resnet50.h) to "./model_data" from CSU cloud disk. <br>
**Configuration** <br>
   Open "./keras_frcnn/config.py", modify self.val_img_dir or self.test_img_dir to the image path ("./data/NIRPed/images/val" or "./data/NIRPed/images/test"); <br>
   modify self.model_path to the model path (./model_data/NIRPed_weights_resnet50.h5). <br>
   Open "Test_JointDetector.py", modify results_dir to the results-saving path ("./results_NIRPed"). <br>
**Begin prediction** <br>
   Run "Test_JointDetector.py" to start prediction. During the prediction stage, results will be saved in the folder of "./results_NIRPed". <br>

### 2. Using your own weights <br>
**Data preparation**  <br>
   After optimizing the weights on your own data, put the weights in the folder of "./model_data".  <br>
   Before prediction, put the png image files into the folder of "./data/yourDataset/images/val" or "./data/yourDataset/images/test"; <br>
   put the COCO-format annotation file val.json or test.json into the folder of "./data/yourDataset/labels". <br>   
**Configuration** <br>
   Open "./keras_frcnn/config.py", modify self.val_img_dir or self.test_img_dir to the image path ("./data/yourDataset/images/val" or "./data/yourDataset/images/test"); <br>
   modify self.val_anno or self.test_anno to the annotation paths ("./data/yourDataset/labels/val.json" or "./data/yourDataset/labels/test.json"); <br>
   modify self.class_mapping according to your tasks; <br>
   modify other parameters according to your tasks. <br>   
   Open "Test_JointDetector.py", modify results_dir to the results-saving path ("./results_yourDataset").** <br>
**Begin prediction** <br>
   Run "Test_JointDetector.py" to start prediction. During the prediction stage, results will be saved in the folder of "./results_yourDataset". <br>

## E. How2eval <br>
### 1. Evaluation on NIRPed validation or testing subset <br>
**Data preparation**  <br>
   After prediction, put the results in the folder of "./results_NIRPed";  <br>
   Before evaluation, put the png image files into the folder of "./data/NIRPed/images/val" or "./data/NIRPed/images/test"; <br>
   put the COCO-format annotation file val.json or test.json into the folder of "./data/NIRPed/labels". <br>  
**Configuration** <br>
   Open "./keras_frcnn/config.py", modify self.val_img_dir or self.test_img_dir to the image path ("./data/NIRPed/images/val" or "./data/NIRPed/images/test"). <br>
   Open "Evaluate_JointDetector.py", modify Detection_results_dir to the results-saving path ("./results_NIRPed/dt_results_val_B300_001"); <br>
   modify other parameters in the "Evaluate_JointDetector.py" according to your tasks. <br>   
**Begin evaluation** <br>
   Run Evaluate_JointDetector.py to start evaluation. During the prediction stage, results will be saved in the folder of "./results_NIRPed/dt_results_val_B300_001". <br>  

### 2. Evaluation on your own dataset (yourDataset) <br>
**Data preparation**  <br>
   After prediction, put the results in the folder of "./results_yourDataset";  <br>
   Before evaluation, put the png image files into the folder of "./data/yourDataset/images/val" or "./data/yourDataset/images/test"; <br>
   put the COCO-format annotation file val.json or test.json into the folder of "./data/yourDataset/labels". <br>  
**Configuration** <br>
   Open "./keras_frcnn/config.py", modify self.val_img_dir or self.test_img_dir to the image path ("./data/yourDataset/images/val" or "./data/yourDataset/images/test"). <br>
   Open "Evaluate_JointDetector.py", modify Detection_results_dir to the results-saving path ("./results_yourDataset/dt_results_val_B300_001"); <br>
   modify other parameters in the "Evaluate_JointDetector.py" according to your tasks. <br>   
**Begin evaluation** <br>
   Run Evaluate_JointDetector.py to start evaluation. During the prediction stage, results will be saved in the folder of "./results_yourDataset/dt_results_val_B300_001". <br>  
   
# III. Performance<br>
| train dataset | weight name | test dataset | input image size | MR-2 | AP@0.5 |  MAER |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: |
| NIRPed | [NIRPed_weights_resnet50.h](https://github.com/XiaobiaoDai/NIRPed/blob/master/model_data/NIRPed_weights_resnet50.h5) | NIRPed-val | 640*256 | **6.5** | **92.4** | **5.46**
| NightOwls | [NightOwls_weights_resnet50.h](https://github.com/XiaobiaoDai/NIRPed/blob/master/model_data/NightOwls_weights_resnet50.h5) | NightOwls-val | 640*256 | 17.2 | 77.7 | -
| ECP | [ECP_weights_resnet50.h](https://github.com/XiaobiaoDai/NIRPed/blob/master/model_data/ECP_weights_resnet50.h5) | ECP-val | 960*256 | 21.1 | 81.9 | -
| KAIST | [KAIST_weights_resnet50.h](https://github.com/XiaobiaoDai/NIRPed/blob/master/model_data/KAIST_weights_resnet50.h5) | KAIST-test | 640*256 | 37.3 | 69.8 | -
<br> 

# IV. References <br>
1. https://github.com/jinfagang/keras_frcnn <br>
2. https://github.com/chenyuntc/simple-faster-rcnn-pytorch <br>
