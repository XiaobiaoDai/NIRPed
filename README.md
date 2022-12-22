NIRPed: A Novel Benchmark for Nighttime Pedestrian and Its Distance Joint Detection<br>(Implementation based on Tensorflow & Keras)
---
# Content
一. [Paper Introduction](#论文介绍) <br> 
I. [Introduction](#简介)<br> 
II. [NIRPed dataset](#数据创新)<br> 
III. [Joint detection](#联合检测方法创新)<br>
IV. [Performance](#性能表现)<br> 
V. [Conclusion](#结论)<br>

二. [Implementation based on Tensorflow & Keras](#实施) <br> 
I. [Environment](#所需环境)<br> 
II. [Download](#文件下载)<br> 
III. [How2train](#训练步骤)<br> 
IV. [How2predict](#预测步骤)<br>
V. [How2eval](#评估步骤)<br> 

三. [References](#参考资料)<br> 
I. [Websites](#参考网址)<br> 
II. [Papers](#参考网址)<br> 

一. Paper Introduction
--- 

I. Introduction
 ![img.png](Table1.png)<br> 
 ![img.png](FigureS01.png)<br> 

This work‘s contributions are threefold:<br> 
(1) A large-scale, competitive and robust narrowband near- infrared nighttime pedestrian dataset containing target distances was constructed as a powerful supplement to NightOwls.<br> 
(2) Using LiDAR to obtain the precise distances to support distance detection and fine-grained model optimization.<br> 
(3) Extending the Faster-RCNN to achieve joint detection of pedestrian and its distance in one step.<br> 

#II.  Narrowband NIR Nighttime Pedestrian and Its Distance Joint Detection Dataset
##A. Data Collection Device
 ![img.png](Figure01.png)<br> 

##B. Data Collection and Target Distance Obtaining
 ![img.png](Figure2.png)<br> 
 ![img.png](Figure3.png)<br> 
 ![img.png](Figure4.png)<br> 
 ![img.png](FigureS2.png)<br> 

##C. Dataset Construction
![img.png](Table3.png)<br> 

##D. Statistics of Dataset Attributes
![img.png](Table4.png)<br> 
![img.png](Figure5_6.png)<br> 

## III.  Faster-RCNN-Based Joint Detection Method for Pedestrian and Its Distance Detection
 ![img.png](Figure07.png)<br> 

##IV. Performance
| train dataset | weight name | test dataset | input image size | MR-2 | AP@0.5 |  MAER |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: |
| NIRPed | [NIRPed_weights_resnet50.h](https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED) | NIRPed-val | 640*256 | **6.5** | **92.4** | **5.46**
| NightOwls | [NightOwls_weights_resnet50.h](https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED) | NightOwls-val | 640*256 | 17.2 | 77.7 | -
| ECP | [ECP_weights_resnet50.h](https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED) | ECP-val | 640*256 | 21.1 | 81.9 | -
| KAIST | [KAIST_weights_resnet50.h](https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED) | KAIST-test | 640*256 | 37.3 | 69.8 | -
<br> 

 ![img.png](Figure8.png)<br> 
 ![img.png](Figure9.png)<br> 
 ![img.png](Figure10.png)<br> 
 ![img.png](Table6.png)<br> 
##V. Conclusion
In order to cope with the huge challenges of nighttime driving scenes, we selected a narrowband near-infrared imaging system to collect high-quality images and matched LiADR to precisely obtain the distance information in the imaging area. We further constructed a nighttime pedestrian and its distance joint detection dataset NIRPed, which contains 142k narrowband near-infrared images and 236k annotations of pedestrian and other person- related objects. Among them, the pedestrian annotation amounts to 146k, which is 3 times of NightOwls—the largest visible-light nighttime pedestrian dataset at present. For achieving pedestrian detection and distance detection tasks synchronously, we proposed a joint detection method based on Faster-RCNN and monocular images. Our method has been proven to be advanta- geous in dealing with complex nighttime driving scenes, which will hopefully contribute to the PPS to effectively realize nighttime pedestrian detection and distance detection at low cost.
<br> 

二. Implementation based on Tensorflow & Keras
---

##I. Environment
pythonn == 3.9 <br>
tensorflow-gpu == 2.9.0 <br>
keras == 2.9.0 <br>
Please refer to requirements.txt for more configuration.  <br>
##II. Download
训练所需的NIRPed_weights_resnet50.h或者NightOwls_weights_resnet50.h以及主干的网络权重可以在百度云下载。  
NIRPed_weights_resnet50.h是resnet50为主干特征提取网络用到的;  
NightOwls_weights_resnet50.h是resnet50为主干特征提取网络用到的;  
中南大学云盘链接: https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED

NIRPed数据集下载地址如下，里面已经包括了训练集、验证集、测试集，无需再次划分：  
链接:https://pan.csu.edu.cn/#/link/3F35F56A95E21A7D2BDE30B3A431936B?path=NIR_PED      

##III. How2train
### a、Training on NIRPed
1. 数据集的准备
   **训练前需要下载好NIRPed的数据集，解压后png图像放在./data/NIRPed/images/train; coco格式的json注释路径为./data/NIRPed/labels/train.json。**
2. 数据集的处理 
   **在完成数据集的摆放之后，训练主程序train_JointDetector.py需要调用get_data_from_json.py获取训练用的json数据(./data/NIRPed/labels/train.json)。**
   **修改config.py里面的参数self.train_img_dir，指向训练图像存放路径(./data/NIRPed/images/train)。**
   **修改config.py里面的参数self.train_anno，指向训练训练注释存放路径(./data/NIRPed/labels/train.json)。**
3. 开始网络训练
   config.py的默认参数用于训练NIRPed数据集，直接运行train_JointDetector.py即可开始训练；
   完成修改后就可以运行Test_JointDetector.py进行检测了。   

### b、Training on your own dataset
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

##IV. How2predict
### a、Use our weights
1. 下载完库后解压，在中南云盘下载NIRPed_weights_resnet50.h，放入./model_data文件夹中。 
2. 训练结果预测需要用到Test_JointDetector.py文件。首先需要去Test_JointDetector.py里面修改model_path和results_dir;
   再修改config.py里面的参数self.val_img_dir或self.test_img_dir，指向预测图像存放路径。 
   **model_path指向训练好的权值文件，在./model_data文件夹里;**
   **results_dir为检测结果存放文件夹，在./results_NIRPed文件夹里;** 
   **config.py里面的参数self.val_img_dir或self.test_img_dir，指向训练图像存放路径./data/NIRPed/images/val或./data/NIRPed/images/test**
3. 完成修改后就可以运行Test_JointDetector.py进行检测了。

### b、Use your own weights
1. 按照训练步骤训练。
2. 在config.py文件里面，修改model_path、val_img_dir(test_img_dir)、val_anno(test_anno)对应训练好的文件，以及class_mapping对应model_path的分类；
   **model_path对应./model_data文件夹下面的权值文件;**  
   **val_anno(test_anno)对应./data/dataset/labels文件夹下面的注释文件。**
3. 在Test_JointDetector.py里面进行设置RPN最大提案数量max_boxes(默认为300)和预测结果保存的置信度阈值score_threshold_cls(默认为0.001)。
   **score_threshold_cls取值较小，获得更多结果用于后续评估。**
4. 运行Test_JointDetector.py进行检测。 

##V. How2eval
### a、Evaluation on NIRPed validation subset
1. 本文使用COCO格式进行评估。NIRPed已经划分好了验证集和测试集及其注释；
2. 在config.py里面修改model_path。**model_path指向训练好的权值文件，在./model_data文件夹里；**  
3. 运行Evaluate_JointDetector.py即可获得评估结果，评估结果会保存在./results_NIRPed文件夹中。

### b、Evaluation on your own dataset
1. 本文使用COCO格式进行评估；  
2. 划分训练集、验证集和测试集，制作各子集COCO格式json文件；
3. 在config.py里面修改model_path。**model_path指向训练好的权值文件，在./model_data文件夹里；**  
4. 运行Evaluate_JointDetector.py即可获得评估结果，评估结果会保存在./results_dataset文件夹中。

三. References
---
###I. websites <br>
1. https://github.com/jinfagang/keras_frcnn <br>
2. https://github.com/chenyuntc/simple-faster-rcnn-pytorch <br>
###II. papers <br>
[1]	    WHO, “Global Status Report on Road Safety,” World Health Organization, Geneva, Switzerland, Rep. WH018, Jun. 17, 2018. <br> 
[2]	    X. Zhang et al., “Deep Convolutional Neural Networks for All-Day Pedestrian Detection,” in Proc. ICISA, 2017, pp. 171-178. <br> 
[3]	    P. Dollár et al., “Pedestrian detection: an evaluation of the state of the art,” IEEE TPAMI, vol. 34, no.4, pp. 743-761, Apr. 2012. <br> 
[4]	    T. Gandhi et al., “Pedestrian Protection Systems: Issues, Survey, and Challenges," IEEE TITS, vol. 8, no.3, pp. 413-430, Sep. 2007.<br> 
[5]	    A. A. Ali et al., “Distance estimation and vehicle position detection based on monocular camera,” in Proc. AIC-MITCSA, 2016, pp. 1-4. <br> 
[6]	    J. Wu et al., “Temporal-Context Enhanced Detection of Heavily Occluded Pedestrians,” in Proc. CVPR, 2020, pp. 13427-13436. <br> 
[7]	    M. Braun et al., “The EuroCity Persons Dataset: A Novel Benchmark for Person Detection in Traffic Scenes,” IEEE TPAMI, vol. 41, no.8, pp. 1844-1861, 2019. <br> 
[8]	    S. Zhang et al., “CityPersons: A Diverse Dataset for Pedestrian Detection,” in Proc. CVPR, 2017, pp. 4457-4465. <br> 
[9]	    L. Zhang et al., “Is Faster R-CNN Doing Well for Pedestrian Detection?” in Proc. ECCV, 2016, pp. 443-457. <br> 
[10]	D. Matti et al., “Combining lidar space clustering and convolutional neural networks for pedestrian detection,” in Proc. IEEE AVSS, 2017, pp. 1-6.<br> 
[11]	S. Zhang et al., “Occluded Pedestrian Detection Through Guided Attention in CNNs,” in Proc. CVPR, 2018, pp. 6995-7003. <br> 
[12]	M. Kilicarslan and J. Y. Zheng, “DeepStep: Direct Detection of Walking Pedestrian From Motion by a Vehicle Camera,” IEEE Transactions on Intelligent Vehicles, vol. 7, no. 2, pp. 1-11, Jun. 2022.<br> 
[13]	D. Bao et al., “Vehicle distance detection based on monocular vision,” in Proc. PIC, 2016, pp. 187-191.<br> 
[14]	A. Bharade et al., “Statistical approach for distance estimation using Inverse Perspective Mapping on embedded platform,” in Proc. IEEE India Conf., 2014, pp. 1-5.<br> 
[15]	P. Wongsaree et al., “Distance Detection Technique Using Enhancing Inverse Perspective Mapping,” in Proc. Int. Conf. on Comput. And Comm. Syst., 2018, pp. 217-221.<br> 
[16]	R. Adamshuk et al., “On the applicability of inverse perspective mapping for the forward distance estimation based on the HSV colormap,” in Proc. IEEE Int. Conf. on Industr. Technology, 2017, pp. 1036-1041.<br> 
[17]	T. Jia et al., “Depth measurement based on omni-directional camera and pixel offset,” in Proc. WCICI, 2014, pp. 5416-5419.<br> 
[18]	Z. Rumin et al., “A Real-Time Obstacle Detection Algorithm for the Visually Impaired Using Binocular Camera,” in Proc. Int. Conf. in Commun. Signal Process. And Syst., 2017, pp. 1412-1419. <br> 
[19]	S. Nedevschi et al., “Stereo-Based Pedestrian Detection for Collision- Avoidance Applications,” IEEE TITS, vol. 10, no.3, pp. 380-391, Sep. 2009.<br> 
[20]	M. Rezaei et al., "Robust Vehicle Detection and Distance Estimation Under Challenging Lighting Conditions," IEEE TITS, vol. 16, no.5, pp. 2723-2743, Mar. 2015.<br> 
[21]	X. Dai et al., “Multi-task faster R-CNN for nighttime pedestrian detection and distance estimation,” Infrared Physics & Technology, vol. 115, p. 103694, Jun. 2021. <br> 
[22]	X. Dai et al., “Near infrared nighttime road pedestrian recognition based on convolutional neural network,” Infrared Physics & Technology, vol. 97, pp. 25-32, Mar. 2019. <br> 
[23]	L. Zhang et al. (2019). “The Cross-Modality Disparity Problem in Multispectral Pedestrian Detection.” [Online]. Available: https://arXiv. org/abs/1901.02645 <br> 
[24]	C. Li et al., “Illumination-aware faster R-CNN for robust multispectral pedestrian detection,” Pattern Recognit., vol. 85, pp. 161-171, Jan. 2019. <br> 
[25]	S. K. Biswas et al., “Linear support tensor machine with LSK channels: pedestrian detection in thermal infrared images,” IEEE TIP, vol. 26, no.9, pp. 4229-4242, May 2017. <br> 
[26]	L. Lu et al., “Infrared Pedestrian Segmentation Through Background Likelihood and Object-Biased Saliency,” IEEE TITS, vol. 19, no.9, pp. 2826-2844, Nov. 2017. <br> 
[27]	M. A. Galarza-Bravo et al., “Pedestrian Detection at Night Based on Faster R-CNN and Far Infrared Images,” in Proc. Int. Conf. on Intell. Robotics and Appl., 2018, pp. 335-345. <br> 
[28]	L. Neumann et al., “NightOwls: A Pedestrians at Night Dataset,” in Proc. ACCV, 2018, pp. 691-705. <br> 
[29]	J. Liu et al. (2016). “Multispectral Deep Neural Networks for Pedestrian Detection,” [Online]. Available: http://arxiv.org/abs/1611.02644 <br> 
[30]	S. Hwang et al., “Multispectral pedestrian detection: Benchmark dataset and baseline,” in Proc. CVPR, 2015, pp. 1037-1045. <br> 
[31]	Z. Zhao et al., “Object Detection with Deep Learning: A Review,” IEEE TNNLS., vol. 30, no.11, pp. 3212-3232, Jan. 2019.<br> 
[32]	R. Girshick et al., “Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation,” in Proc. CVPR, 2014, pp. 580-587.<br> 
[33]	R. Girshick, “Fast R-CNN,” in Proc. ICCV, 2015, pp. 1440-1448. <br> 
[34]	S. Ren et al., “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” IEEE TPAMI, vol. 39, no.6, pp. 1137-1149, Jun. 2016.<br> 
[35]	J. Redmon et al., “You Only Look Once: Unified, Real-Time Object Detection,” in Proc. CVPR, 2015, pp. 779-788.<br> 
[36]	J. Leng et al., "Robust Obstacle Detection and Recognition for Driver Assistance Systems," IEEE TITS, vol. 21, no.4, pp. 1560-1571, Apr. 2020.<br> 
[37]	T. Lin et al., “Microsoft COCO: Common Objects in Context,” in Proc. ECCV, 2014, pp. 740-755.<br> 
[38]	K. He et al., “Deep Residual Learning for Image Recognition,” in Proc. CVPR, 2016, pp. 770-778.<br> 
[39]	Y. Tian et al., "Depth estimation for advancing intelligent transport systems based on self-improving pyramid stereo network," IET Intell. Transp. Syst., vol. 14, no.5, pp. 338-345, Jul. 2020.<br> 
[40]	S. Liang et al., “A Novel Sparse Geometric 3-D LiDAR Odometry Approach,” IEEE Syst. J., vol. 15, no.1, pp. 1390-1400, Mar. 2020.<br> 
[41]	H. Gao et al., “Object classification using CNN-based fusion of vision and LIDAR in autonomous vehicle environment,” IEEE TII, vol. 14, no. 9, pp. 4224-4231, Apr. 2018.<br> 
[42]	Z. Rumin et al., "A Real-Time Obstacle Detection Algorithm for the Visually Impaired Using Binocular Camera," in Proc. Int. Conf. in Comm., Signal Process., and Syst., 2017, pp. 1412-1419.<br> 
[43]	S. Joung et al., "Unsupervised Stereo Matching Using Confidential Corresp- ondence Consistency," IEEE TITS, vol. 21, no.5, pp. 2190-2203, May 2020.<br> 
[44]	A. Geiger et al., “Automatic camera and range sensor calibration using a single shot,” in Proc. ICRA, 2012, pp. 3936-3943. <br> 
[45]	Intelligent transport systems-Forward vehicle collision warning systems- Performance requirements and test procedures, ISO15623, I. T. Systems, Switzerland, 2013.<br> 
