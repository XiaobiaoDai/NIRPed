# coding: utf-8

#对代码所需要的参数进行配置
from keras import backend as K
import numpy as np
import sys,os
class Config:
    def __init__(self): #调用此类是初始化运行函数或方法s
        self.verbose = True
        # TODO: the only file should to be change for other data to train
        self.network = 'Resnet50NIR1' #frcnn网络名称
        self.learning_rate = 1e-5

        # setting for data augmentation 设置数据是否增强
        self.use_hsv = False     # 饱和度和亮度增强
        self.use_horizontal_flips = True  #默认进行水平翻转增强#TODO:对训练影响大？
        self.use_vertical_flips = False   #默认不进行垂直翻转增强
        self.use_rotate_angle = False  # 小角度旋转
        self.use_translation = True  # 水平垂直平移
        self.rot_90 = False          #默认不进行顺时针旋转角度增强

        self.balanced_classes = False #类别数量是否平衡：默认否WeightsResnet50NIR1RGB64_1024_128_2o5
        self.use_attention_imgs = True
        # TODO:用k-means方法求得3*3=9个先验框(注意：先年框的个数必须为自然是的平方,否则后续计算无法进行)。
        self.batch_size_rpn = 256
        self.batch_size_cls = 32

        # size to resize the smallest side of the image 缩放后的图像短边像素
        self.im_rows = 256 #近红外图像的短边360->352->320->256
        self.im_cols = 640 #近红外图像的短边1280->640
        self.im_rows_show = 512 #近红外图像的短边360->352->320->256
        self.im_cols_show = 1280 #近红外图像的短边1280->640

        #image channel-wise mean to subtract 图像三通道均值？？？？如何计算得来？需要对所有训练样本求RGB均值
        # self.num_test_imgs = 40405
        self.num_test_imgs = 39972
        self.anchor_box_scales = [18, 28, 38, 60, 100]  # 根据实拍图像尺寸1024*640，resized后变成640*256，NightOwls 1.7x  统计 distance<80
        self.anchor_box_ratios = [[1, 2.0 / 2.5 / 0.41]]  # 在resized图上的锚框短长边放大系数 NightOwls 0.41    2.44  1.95
        self.img_channel_mean = [73.52, 68.18, 72.75] #夜间近红外图像RGB三通道均值 num_images=141792
        self.Dis0 = 30  # -np.log(gta_real[bbox_num, 4]/C.Dis0)近红外距离归一化标准距离distance_ped_mean=33.97178154169947
        self.Dis_max = 80  # -np.log(gta_real[bbox_num, 4]/C.Dis0)近红外距离归一化标准距离distance_ped_mean=33.97178154169947
        self.Dis_threshold = self.Dis_max  #用来训练网络的标记的最远距离。
        self.Occ_threshold = 0.35  # 遮挡系数阈值0.35
        self.class_mapping = {'Ped': 0, 'bg': 1}  # 网络输出节点与类映射字典:#{'pedestrian': 0, 'bg': 1}
        self.classes_count = {'Ped': 64167, 'bg': 282}  # 网络输出节点与类映射字典:{'Cyc': 0, 'Mot': 1, 'Ped': 2, 'bg': 3}

        self.best_loss_min = 0.02  # 分类时最大交并比阈值self.classifier_max_overlap = 0.5
        self.best_loss_max = self.best_loss_min + 0.01  # 分类时最大交并比阈值self.classifier_max_overlap = 0.5

        self.img_scaling_factor = 1.0 #图像缩放因子

        # number of ROIs at once.  classifier net 每批次检测RoI数量为32个
        self.num_rois = 32

        # stride at the RPN (this depends on the network configuration)RPN步长
        self.rpn_stride = 8 #在VGG网络在缩放后的图片上的rpn步长为16,我们的也为16

        # scaling the stdev 缩放标准差???？？？
        self.std_scaling = 4.0  #分类方差缩放？
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]#边界框回归标准差系数

        # overlaps for RPN RPN的交并比阈值
        self.rpn_min_overlap = 0.3 #RPN分类时最小交并比阈值
        self.rpn_max_overlap = 0.7 #RPN分类时最大交并比阈值

        # overlaps for classifier ROIs ROI分类的交并比阈值
        self.classifier_min_overlap = 0.45 #分类时最小交并比阈值self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.50 #分类时最大交并比阈值self.classifier_max_overlap = 0.5

        self.use_bg_imgs = True
        self.model_dir = '.\model'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model_path = os.path.join(self.model_dir, 'Weights%sRGB64_1024_128_2o5.h5' % (self.network))  # 基础的NIRnet模型路径,已训练好
        self.model_path0 = os.path.join(self.model_dir, 'Weights%sRGB64_1024_128_2o5' % (self.network))   # 基础的NIRnet模型路径,已训练好
        self.model_pathe = os.path.join(self.model_dir, 'Weights%sRGB64_1024_128_2o5e.h5' % (self.network))

        self.base_net_weights = self.model_path
        self.num_epochs = 5000   #训练回合(次数)100*8727/200 = 4363.5
        self.length_epoch = 1000   #训练批次(回合)图片数
        # TODO: this field is set to simple_label txt, which in very simple format like:此字段设置为simple_label.txt，其格式非常简单，如：
        # TODO: /path/image_2/000000.png,712.40,143.00,810.73,307.92,Pedestrian, see kitti_simple_label.txt for detail

        self.train_file = r'.\data\miniNIRPed\labels\train_mini.json'
        self.val_file = r'.\data\miniNIRPed\labels\val_mini.json'
        self.test_file = r'.\data\miniNIRPed\labels\test_mini.json'
        self.show_imgs_directory = r'.\data\Show'

        self.training_loss_file = os.path.join(self.model_dir, 'Training_loss.json')
        self.Loss0 = 0.05  # 训练总损失目标值