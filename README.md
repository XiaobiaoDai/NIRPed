# Faster-RCNN-based Pedestrian and its distance joint detection：Two-Stage行人及其距离联合检测模型在Keras中的实现
---

## 目录
1. [仓库更新 Top News](#仓库更新)
2. [性能情况 Performance](#性能情况)
3. [所需环境 Environment](#所需环境)
4. [文件下载 Download](#文件下载)
5. [预测步骤 How2predict](#预测步骤)
6. [训练步骤 How2train](#训练步骤)
7. [评估步骤 How2eval](#评估步骤)
8. [参考资料 Reference](#Reference)

## Top News
**`2022-12`**:**首次提交，支持行人及其距离联合检测、忽略区域自动剔除、训练样本自动选择。**   

## 性能情况
| train dataset | weight name | test dataset | input image size | MR-2 | AP@0.5 |  MAER |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: |
| NIRPed | [NIRPed_weights_resnet50.h](https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED) | NIRPed-val | 640*256 | 6.5 | 92.4 | 5.46
| NightOwls | [NightOwls_weights_resnet50.h](https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED) | NightOwls-val | 640*256 | 17.2 | 77.7 | -
| ECP | [ECP_weights_resnet50.h](https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED) | ECP-val | 640*256 | 21.1 | 81.9 | -
| KAIST | [KAIST_weights_resnet50.h](https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED) | KAIST-val | 640*256 | 37.3 | 69.8 | -


## 所需环境
tensorflow-gpu == 2.9.0;
keras == 2.9.0

## 文件下载
训练所需的NIRPed_weights_resnet50.h或者NightOwls_weights_resnet50.h以及主干的网络权重可以在百度云下载。  
NIRPed_weights_resnet50.h是resnet50为主干特征提取网络用到的;  
NightOwls_weights_resnet50.h是resnet50为主干特征提取网络用到的;  
中南大学云盘链接: https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED

NIRPed数据集下载地址如下，里面已经包括了训练集、验证集、测试集，无需再次划分：  
链接:https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED      

## 训练步骤
### a、训练NIRPed数据集
1. 数据集的准备
   **训练前需要下载好NIRPed的数据集，解压后png图像放在./data/NIRPed/images/train; coco格式的json注释路径为./data/NIRPed/labels/train.json。**
2. 数据集的处理 
   **在完成数据集的摆放之后，训练主程序train_JointDetector.py需要调用get_data_from_json.py获取训练用的json数据(./data/NIRPed/labels/train.json)。**
   **修改config.py里面的参数self.train_img_dir，指向训练图像存放路径(./data/NIRPed/images/train)。**
   **修改config.py里面的参数self.train_anno，指向训练训练注释存放路径(./data/NIRPed/labels/train.json)。**
3. 开始网络训练
   config.py的默认参数用于训练NIRPed数据集，直接运行train_JointDetector.py即可开始训练；
   完成修改后就可以运行Test_JointDetector.py进行检测了。   

### b、训练自己的数据集
1. 数据集的准备 
   **本文使用COCO格式进行训练，训练前需要自己制作好COCO格式数据集，** 
   训练前将标签文件train.json放在./data/NIRPed/labels文件夹中；
   训练前将图片文件*.png放在./data/NIRPed/images/train文件夹中；   
   在config.py文件里面，修改val_img_dir(test_img_dir)、val_anno(test_anno)对应训练好的文件，以及class_mapping对应的分类。
2. 数据集的处理
   在完成数据集的摆放之后，train_JointDetector.py需要调用get_data_from_json.py获得训练用的train.json。
   **修改config.py里面的参数self.train_anno，指向训练训练注释存放路径(./data/NIRPed/labels/train.json)。**
3. 开始网络训练
   **训练的参数较多，均在config.py中，大家可以在下载库后仔细看注释,做相应的修改。**
   修改完后就可以运行train_JointDetector.py开始训练了，在训练一个iteration后，权值会生成在./model_data文件夹中。  

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在中南云盘下载NIRPed_weights_resnet50.h，放入./model_data文件夹中。 
2. 训练结果预测需要用到Test_JointDetector.py文件。首先需要去Test_JointDetector.py里面修改model_path和results_dir;
   再修改config.py里面的参数self.val_img_dir或self.test_img_dir，指向预测图像存放路径。 
   **model_path指向训练好的权值文件，在./model_data文件夹里;**
   **results_dir为检测结果存放文件夹，在./results_NIRPed文件夹里;** 
   **config.py里面的参数self.val_img_dir或self.test_img_dir，指向训练图像存放路径./data/NIRPed/images/val或./data/NIRPed/images/test**
3. 完成修改后就可以运行Test_JointDetector.py进行检测了。

### b、使用自己训练的权重
1. 按照训练步骤训练。
2. 在config.py文件里面，修改model_path、val_img_dir(test_img_dir)、val_anno(test_anno)对应训练好的文件，以及class_mapping对应model_path的分类；
   **model_path对应./model_data文件夹下面的权值文件;**  
   **val_anno(test_anno)对应./data/dataset/labels文件夹下面的注释文件。**
3. 在Test_JointDetector.py里面进行设置RPN最大提案数量max_boxes(默认为300)和预测结果保存的置信度阈值score_threshold_cls(默认为0.001)。
   **score_threshold_cls取值较小，获得更多结果用于后续评估。**
4. 运行Test_JointDetector.py进行检测。 

## 评估步骤 
### a、评估NIRPed的验证集
1. 本文使用COCO格式进行评估。NIRPed已经划分好了验证集和测试集及其注释；
2. 在config.py里面修改model_path。**model_path指向训练好的权值文件，在./model_data文件夹里；**  
3. 运行Evaluate_JointDetector.py即可获得评估结果，评估结果会保存在./results_NIRPed文件夹中。

### b、评估自己的数据集
1. 本文使用COCO格式进行评估；  
2. 划分训练集、验证集和测试集，制作各子集COCO格式json文件；
3. 在config.py里面修改model_path。**model_path指向训练好的权值文件，在./model_data文件夹里；**  
4. 运行Evaluate_JointDetector.py即可获得评估结果，评估结果会保存在./results_dataset文件夹中。

## Reference
https://github.com/jinfagang/keras_frcnn
https://github.com/chenyuntc/simple-faster-rcnn-pytorch
