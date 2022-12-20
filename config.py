# coding: utf-8

#对代码所需要的参数进行配置
from keras import backend as K
import numpy as np
import sys,os
class Config:
    def __init__(self): #调用此类是初始化运行函数或方法
        self.verbose = True
        # TODO: the only file should to be change for other data to train
        #self.network = 'VGG16NIR'   #frcnn网络名称
        #self.network = 'Resnet18NIR' #frcnn网络名称

        # self.network = 'Resnet50VIS1' #frcnn网络名称NightOwls
        self.network = 'Resnet50NIR1' #frcnn网络名称
        # self.quality_or_IoU = 'quality'
        self.quality_or_IoU = 'Iou'

        if self.network == 'Resnet50VIS1':
            self.learning_rate = 1e-5
        elif self.network == 'Resnet50NIR1':
            self.learning_rate = 1e-5

        # self.data_train = 'overall'
        self.data_train = 'train'
        # self.data_train = 'val'

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
        #self.anchor_boxes_FRCNN = [[8, 68], [11, 93], [15, 124], [19, 169], [26, 123], [27, 213], [36, 291], [57, 382],[105, 408]]
        #self.anchor_boxes_FRCNN = [[15, 37], [21, 51], [27, 68], [36, 93], [47, 68], [49, 117], [66, 160], [104, 210], [192, 224]]
        # anchor box scales 锚框短边像素大小
        #self.anchor_box_scales = [128, 256, 512]
        #self.anchor_box_scales = [30, 60, 120]  # 根据实拍图像尺寸1280*960和1280*720，缩放后变成640*480和640*360，设置在resized图上的锚框短边尺寸
        #self.anchor_box_ratios = [[1, 1], [1, 1.8], [1, 2.6]] #在resized图上的锚框短长边放大系数
        #self.anchor_box_scales = [32, 64, 128]  # 根据实拍图像尺寸1280*960和1280*720，缩放后变成640*480和640*360，设置在resized图上的锚框短边尺寸
        # self.anchor_box_scales = [24, 36, 54, 81, 122]  # 根据实拍图像尺寸1280*960和1280*720，缩放后变成640*352，设置在resized图上的锚框短边尺寸 NightOwls 1.3x
        # 设置在resized图上的锚框短边尺寸 NightOwls 1.6x # num_Anchors=42770   [[12,17], [18,30], [29,49], [50,84], [95,148]]
        # anchor box ratios 在原始图上的锚框长/短边比例
        #self.anchor_box_ratios = [[1, 1], [2, 1], [1, 2]]
        self.batch_size_rpn = 256
        self.batch_size_cls = 32

        # size to resize the smallest side of the image 缩放后的图像短边像素
        self.im_rows = 256 #近红外图像的短边360->352->320->256
        self.im_cols = 640 #近红外图像的短边1280->640
        self.im_rows_show = 512 #近红外图像的短边360->352->320->256
        self.im_cols_show = 1280 #近红外图像的短边1280->640
        #self.im_size = 360 #近红外图像的短边360->352
        #self.im_size = 512 #热红外图像的短边

        #image channel-wise mean to subtract 图像三通道均值？？？？如何计算得来？需要对所有训练样本求RGB均值
        #self.img_channel_mean = [121.726, 10.523, 131.566]  # 20181220拍摄的热红外图像RGB三通道均值
        if self.network in ['Resnet50VIS0', 'Resnet50VIS1']:
            self.num_test_imgs = 51848
            # self.num_test_imgs = 6373

            self.Datasets = 'NightOwls'  # 模型种类，区分可见光和近红外
            self.anchor_box_scales = [12, 20, 35, 60, 100]  # 根据实拍图像尺寸1024*640，resized后变成640*256，NightOwls 1.7x
            self.anchor_box_ratios = [[1, 1.6/2.5/0.41]] #在resized图上的锚框短长边放大系数 NightOwls 0.41    1.56
            self.img_channel_mean = [64.54, 77.68, 77.05] #NightOwls训练和验证子集图像RGB三通道均值num_images=181912
            self.Dis_threshold = 50  # 用来训练网络的标记的最远距离。
            self.Dis0 = 30  # -np.log(gta_real[bbox_num, 4]/C.Dis0)近红外距离归一化标准距离distance_ped_mean=33.97178154169947
            self.Dis_max = 50  # -np.log(gta_real[bbox_num, 4]/C.Dis0)近红外距离归一化标准距离

            # self.class_mapping = None #网络输出节点与类映射字典:{'Pedestrian'      : 0, 'bg': 1}
            self.class_mapping = {'Ped': 0, 'bg': 1}  # 网络输出节点与类映射字典:#{'pedestrian': 0, 'bg': 1}
            # self.class_mapping = {'pedestrian': 0,  'bg': 1} #网络输出节点与类映射字典:#{'pedestrian': 0, 'bg': 1}
            self.classes_count = {'pedestrian': 42969, 'bg': 0}
            # 网络输出节点与类映射字典:# classes_count={'pedestrian': 42770, 'bicycledriver': 7017, 'motorbikedriver': 438, 'ignore': 23113, 'bg': 0}
            # 对NightOwls训练集进行全面清理后V2： {'Ped': 42923, 'Peo': 0, 'Bic': 7066, 'Mot': 441, 'Ign': 20525, 'bg': 0}
            # 对NightOwls训练集再次进行全面清理后V3： {'Ped': 43049, 'Peo': 0, 'Bic': 7070, 'Mot': 441, 'Ign': 20976, 'bg': 0}
            # 对NIR_Ped训练集{'Ped': 86746, 'Peo': 1811, 'Bic': 2481, 'Mot': 7814, 'Ign': 6063, 'bg': 163}
            '''{'Ped': 131838, 'Peo': 4362, 'Bic': 4919, 'Mot': 21600, 'Ign': 25020, 'bg': 185} Instructions for updating:{'Ped': 0, 'bg': 1}
            Num classes (including Ign and bg) = 6    Number of training images 61711'''
            self.best_loss_min = 0.02  # 分类时最大交并比阈值self.classifier_max_overlap = 0.5
            self.best_loss_max = self.best_loss_min + 0.01  # 分类时最大交并比阈值self.classifier_max_overlap = 0.5
        elif self.network in ['Resnet50NIR0', 'Resnet50NIR1']:
            # self.num_test_imgs = 40405
            self.num_test_imgs = 39972
            # self.num_test_imgs = 6618
            self.Datasets = 'NIR_Ped'  # 模型种类，区分可见光和近红外
            self.anchor_box_scales = [18, 28, 38, 60, 100]  # 根据实拍图像尺寸1024*640，resized后变成640*256，NightOwls 1.7x  统计 distance<80
            # Anchors_FRCNN_Ours =[[18,33], [27,49], [39,72], [58,111], [98,179], ] # num_Anchors=122149
            # [[12,17]1.41, [18,30]1.67, [29,49]1.69, [50,84]1.68, [95,148]1.56]
            self.anchor_box_ratios = [[1, 2.0 / 2.5 / 0.41]]  # 在resized图上的锚框短长边放大系数 NightOwls 0.41    2.44  1.95
            self.img_channel_mean = [73.52, 68.18, 72.75] #夜间近红外图像RGB三通道均值 num_images=141792
            # self.img_channel_mean = [0, 0, 0] #夜间近红外图像RGB三通道均值num_count = 29761
            # self.img_channel_mean = [138.42480960451329, 130.34162923107286, 137.21212691844136] #白天近红外图像RGB三通道均值num_count = 10105
            self.Dis0 = 30  # -np.log(gta_real[bbox_num, 4]/C.Dis0)近红外距离归一化标准距离distance_ped_mean=33.97178154169947
            self.Dis_threshold = 80  #用来训练网络的标记的最远距离。
            self.Occ_threshold = 0.35  # 遮挡系数阈值0.35
            # self.class_mapping = None #网络输出节点与类映射字典:{'Pedestrian'      : 0, 'bg': 1}
            self.class_mapping = {'Ped': 0, 'bg': 1}  # 网络输出节点与类映射字典:#{'pedestrian': 0, 'bg': 1}
            # self.class_mapping = {'pedestrian': 0,  'bg': 1} #网络输出节点与类映射字典:#{'pedestrian': 0, 'bg': 1}
            self.classes_count = {'Ped': 64167, 'bg': 282}  # 网络输出节点与类映射字典:{'Cyc': 0, 'Mot': 1, 'Ped': 2, 'bg': 3}

            self.best_loss_min = 0.02  # 分类时最大交并比阈值self.classifier_max_overlap = 0.5
            self.best_loss_max = self.best_loss_min + 0.01  # 分类时最大交并比阈值self.classifier_max_overlap = 0.5
        else:
            print('Notion: unknown training dataset!')
            sys.exit(0)  # 干净利落地退出系统


        #self.img_channel_mean = [8.1] #灰度图像只有一个通道，计算了1000张图片的均值假设为8.1
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
        self.classifier_min_overlap = 0.30 #分类时最小交并比阈值self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.50 #分类时最大交并比阈值self.classifier_max_overlap = 0.5

        # placeholder for the class mapping, automatically generated by the parser 类映射的占位符，由解析器自动生成

        # location of pretrained weights for the base network 基础网络的预训练权重的位置
        # weight files can be found at:权重文件可在以下位置找到：
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5

        #TODO：'_original'是指_Fast_RCNN， 它 和 _Faster_RCNN  和 self.classifier_min_overlap = 0.5 是绑定在一起的。
        # self.mode = '_Fast_RCNN'
        # self.mode = '_Faster_RCNN'
        self.mode = '25_50'
        # self.mode = ''
        if self.mode in ['_original', '_Fast_RCNN', '_Faster_RCNN']:
            self.rpn_max_overlap = 0.70  # RPN分类时最大交并比阈值
            self.classifier_min_overlap = 0.50  # 分类时最小交并比阈值self.classifier_min_overlap = 0.1
        elif self.mode == '25_50':
            self.rpn_max_overlap = 0.6  # RPN分类时最大交并比阈值
            if self.Datasets == 'NightOwls':
                self.classifier_min_overlap = 0.45  # 分类时最小交并比阈值self.classifier_min_overlap = 0.1
            else:
                self.classifier_min_overlap = 0.50  # 分类时最小交并比阈值self.classifier_min_overlap = 0.1
        else:
            # self.rpn_max_overlap = 0.75  # RPN分类时最大交并比阈值
            # self.rpn_max_overlap = 0.70  # RPN分类时最大交并比阈值
            # self.rpn_max_overlap = 0.65  # RPN分类时最大交并比阈值
            self.rpn_max_overlap = 0.60  # RPN分类时最大交并比阈值
            # self.rpn_max_overlap = 0.55  # RPN分类时最大交并比阈值
            # self.rpn_max_overlap = 0.50  # RPN分类时最大交并比阈值

            self.classifier_min_overlap = 0.50  # 分类时最小交并比阈值self.classifier_min_overlap = 0.1
            # self.classifier_min_overlap = 0.47  # 分类时最小交并比阈值self.classifier_min_overlap = 0.1
            # self.classifier_min_overlap = 0.45  # 分类时最小交并比阈值self.classifier_min_overlap = 0.1
            # self.classifier_min_overlap = 0.43  # 分类时最小交并比阈值self.classifier_min_overlap = 0.1
            # self.classifier_min_overlap = 0.40  # 分类时最小交并比阈值self.classifier_min_overlap = 0.1
            # self.classifier_min_overlap = 0.35  # 分类时最小交并比阈值self.classifier_min_overlap = 0.1
            # self.classifier_min_overlap = 0.30  # 分类时最小交并比阈值self.classifier_min_overlap = 0.1
            # self.classifier_min_overlap = 0.25  # 分类时最小交并比阈值self.classifier_min_overlap = 0.1

        # self.use_bg_imgs = False
        self.use_bg_imgs = True
        #TODO：_Fast_RCNN 和 _Faster_RCNN  和 self.classifier_min_overlap = 0.5 是绑定在一起的。
        if self.mode == '25_50':
            # self.model_dir = '.\model2021_RPNT%d_T%s_%s' % (100*self.rpn_max_overlap, self.mode, self.data_train)
            self.model_dir = '.\%s_RpnT%d_ClsT%s_%s_%s' % (self.Datasets, 100*self.rpn_max_overlap, self.mode, self.data_train, self.quality_or_IoU)
        else:
            # self.model_dir = '.\model2021_RPNT%d_T%d%s_%s' % (100*self.rpn_max_overlap, 100*self.classifier_min_overlap, self.mode, self.data_train)
            self.model_dir = '.\%s_RpnT%d_ClsT%d%s_%s_%s' % (self.Datasets, 100*self.rpn_max_overlap, 100*self.classifier_min_overlap, self.mode, self.data_train, self.quality_or_IoU)

        # model_dir = './model_T%d' % (100*self.classifier_min_overlap)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if len(self.img_channel_mean) == 3:  #如果是RGB图像
            self.model_path = os.path.join(self.model_dir, 'Weights%sRGB64_1024_128_2o5.h5' % (self.network))  # 基础的NIRnet模型路径,已训练好
            self.model_path0 = os.path.join(self.model_dir, 'Weights%sRGB64_1024_128_2o5' % (self.network))   # 基础的NIRnet模型路径,已训练好
            self.model_pathe = os.path.join(self.model_dir, 'Weights%sRGB64_1024_128_2o5e.h5' % (self.network))

        elif len(self.img_channel_mean) == 1:  # 如果是Gray图像
            self.model_path = './model/Weights{}NIR32_512_64_2o5.h5'.format(self.network)   # 基础的NIRnet模型路径,已训练好
            self.model_path0 = './model/Weights{}NIR32_512_64_2o5'.format(self.network)   # 基础的NIRnet模型路径,已训练好
            self.model_pathe = './model/Weights{}NIR32_512_64_2o5e.h5'.format(self.network)

        self.base_net_weights = self.model_path
        self.data_dir = '.data/' #数据指向：当前目录的data文件夹
        self.num_epochs = 5000   #训练回合(次数)100*8727/200 = 4363.5
        self.length_epoch = 1000   #训练批次(回合)图片数
        # TODO: this field is set to simple_label txt, which in very simple format like:此字段设置为simple_label.txt，其格式非常简单，如：
        # TODO: /path/image_2/000000.png,712.40,143.00,810.73,307.92,Pedestrian, see kitti_simple_label.txt for detail
        self.simple_label_file = 'GT2018_2019MoreV3Ok.txt' #简单图像及其RoI标记 标签文本
        if self.network == 'Resnet50VIS0':
            self.annotation_file = r'H:\Datasets\NightOwls\Python_version\nightowls_training.json'
            self.training_loss_file = r'H:\Datasets\NightOwls\Python_version\nightowls_training_lossVIS0.json'
            self.image_directoryVIS = r'H:\Datasets\NightOwls\Python_version\nightowls_training'
        elif self.network == 'Resnet50VIS1':
            self.all_file = r'H:\Datasets\NightOwls\Python_version\datasets_NightOwls\nightowls_training_val_Merge90.json'
            self.train_file = r'H:\Datasets\NightOwls\Python_version\datasets_NightOwls\nightowls_trainingMerge.json'
            if self.num_test_imgs == 51848:
                self.val_file = r'H:\Datasets\NightOwls\Python_version\datasets_NightOwls\nightowls_validation_improved.json'
            elif  self.num_test_imgs == 6373:
                self.val_file = r'H:\Datasets\NightOwls\Python_version\datasets_NightOwls\nightowls_validation_improved_exist_ped.json'
            self.test_file = r'H:\Datasets\NightOwls\Python_version\datasets_NightOwls\nightowls_test_imageids.json'
            # self.annotation_file_Valid_PedOk = r'H:\Datasets\NightOwls\Python_version\datasets_NightOwls\nightowls_trainingMV6_Valid_PedOk.json'

            self.image_directoryVIS = r'H:\Datasets\NightOwls\Python_version\nightowls_training'
            self.image_directoryVIS_val = r'H:\Datasets\NightOwls\Python_version\nightowls_validation'
            # self.training_loss_file = r'G:\Datasets\NightOwls\Python_version\nightowls_training_lossVIS1.json'

            if self.data_train == 'overall':
                self.annos_file = self.all_file
            elif self.data_train == 'train':
                self.annos_file = self.train_file

            self.attention_imgs_directory = r'E:\Daixb\Img_dt_results\Visualization_NightOwls40m'
            self.show_imgs_directory = r'H:\Datasets\NightOwls\Python_version\Show'
            self.image_directoryVIS = r'H:\Datasets\NightOwls\Python_version\nightowls_training'
            self.training_image_directory = r'H:\Datasets\NightOwls\Python_version\nightowls_training'
            self.val_image_directory = r'H:\Datasets\NightOwls\Python_version\nightowls_validation'
            self.use_attention_imgs = True
            self.use_bg_imgs = True


        elif self.network == 'Resnet50NIR0':
            self.annotation_file = r'D:\Datasets\annotation\datasets_NIR\train_data_NIR1.json'
            self.training_loss_file = r'D:\Datasets\annotation\datasets_NIR\training_lossNIR0.json'
        elif self.network == 'Resnet50NIR1':
            #self.annotation_file = r'E:\Daixb\Data_Pre_PostProcessing\Annos_NIR_Valid_Ped\Annos_Data\train_data_NIR12_Merge.json'
            #self.annotation_file = r'E:\Daixb\Data_Pre_PostProcessing\Annos_NIR_Valid_Ped\Annos_Data\all_data_NIR13_Merge.json'
            # self.annotation_file = r'D:\Datasets\annotation\datasets_NIR\all_data_NIR90_Night_Merge.json'
            # self.annotation_file = r'H:\Datasets\NIR_Pedestrian\NIR_Ped_all.json'
            # self.annos_file = r'I:\Datasets\annotation\datasets_NIR2018_2021\NIR_Ped_all_AllDay20210321.json'
            self.all_file = r'H:\Datasets\NIR_Ped2021\annos_Datasets2018_2021AllDay\NIR_Ped_AllDay_Modify6_SceOK_Night_allV0.json'
            # self.annotation_file = r'H:\Datasets\NIR_Pedestrian\NIR_Ped_all2021_Night_Merge1.json'
            # self.annotation_file = r'H:\Datasets\NIR_Pedestrian\NIR_Ped_all_Night20210318_Merge.json'

            # self.train_file = r'H:\Datasets\NIR_Ped2021\annos_Datasets2018_2021AllDay\NIR_Ped_AllDay_Modify6_SceOK_Night_trainV0.json'
            self.train_file = r'D:\Datasets\NIR_Ped2021\annos_Datasets2018_2021Night\NIR_Ped2021_trainV6_Spate_Speed_Night.json'

            if self.num_test_imgs == 39972:
                self.val_file = r'H:\Datasets\NIR_Ped2021\annos_Datasets2018_2021AllDay\NIR_Ped_AllDay_Modify6_SceOK_Night_valV0.json'
            elif self.num_test_imgs == 6618:
                self.val_file = r'H:\Datasets\NIR_Ped2021\annos_Datasets2018_2021AllDay\NIR_Ped_AllDay_Modify6_SceOK_Night_valV0_exist_ped.json'
                # self.val_file = r'H:\Datasets\NIR_Ped2021\annos_Datasets2018_2021AllDay\NIR_Ped_AllDay_Modify6_SceOK_Night_valV0_small.json'

            self.test_file = r'H:\Datasets\NIR_Ped2021\annos_Datasets2018_2021AllDay\NIR_Ped_AllDay_Modify6_SceOK_Night_testV0.json'

            if self.data_train == 'overall':
                self.annos_file = self.all_file
            elif self.data_train == 'train':
                self.annos_file = self.train_file
            # self.train_file = r'H:\Datasets\NIR_Pedestrian\NIR_Ped_train_Night20210318_Merge.json'
            # self.val_file = r'I:\Datasets\annotation\datasets_NIR2018_2021\NIR_Ped_val_AllDay20210320_merge.json'
            # self.test_file = r'I:\Datasets\annotation\datasets_NIR2018_2021\NIR_Ped_test_AllDay20210320_merge.json'
            # self.val_file = r'H:\Datasets\NIR_Pedestrian\NIR_Ped_val_Night20210318_Merge.json'
            # self.test_file = r'H:\Datasets\NIR_Pedestrian\NIR_Ped_test_Night20210318_Merge.json'
            #self.training_loss_file = r'D:\Datasets\annotation\datasets_NIR\training_lossNIR5.json'

            self.attention_imgs_directory = r'E:\Daixb\Img_dt_results\Visualization_NIR_Pedval90_Night'
            #self.attention_imgs_directory = r'E:\Daixb\Data_Pre_PostProcessing\Annos_NIR_Valid_Ped'
            # self.show_imgs_directory = r'E:\Ped_Detection_NIR_Distance\PATTERN_RECOGNITION\Illustration\Show'
            # self.show_imgs_directory = r'E:\Daixb\Img_dt_results\Evaluation_results2021\Show'
            self.show_imgs_directory = r'E:\Faster_RCNN\Img_dt_results\Evaluation_results2021\Show'
            # self.show_imgs_directory = r'G:\Faster_RCNN\Img_dt_results\Evaluation_results2022\Show'
            '''attention_imgs_list=['Data20181219200348_010000','Data20181219200348_040000','Data20181219200350_070000', 'Data20181220192439_040000', 'Data20181220192500_010000', 'Data20181220193117_010000',
            'Data20181220193534_040000', 'Data20181220193756_010000', 'Data20181220194000_040000',  'Data20181220194122_030000', 'Data20181220203431_030000', 'Data20181220204307_030000', 'Data20181220204750_030000', 
            'Data20181220205233_030000', 'Data20181220210646_030000', 'Data20190113185714_090000',  'Data20190113185956_020000', 'Data20190113190356_060000', 'Data20190113191658_030000', 'Data20190113192238_060000', 
            'Data20190113192932_030000', 'Data20190113195748_090000', 'Data20190113200454_080000',  'Data20190113201216_090000', 'Data20190113201234_030000', 'Data20190325204013_016894', 'Data20190325204013_941615', 
            'Data20190325204256_658315', 'Data20190325204922_764680', 'Data20190325205118_131475',  'Data20190325205441_145679', 'Data20190326190956_698003', 'Data20190326191132_362155', 'Data20190326191202_009595', 
            'Data20190326191202_922472', 'Data20190326191517_402710', 'Data20190326191531_434266',  'Data20190326191702_266360', 'Data20190326191751_082660', 'Data20190326191924_314180', 'Data20190326192544_825718', 
            'Data20190326193234_632522', 'Data20190326193621_546316', 'Data20190326193716_696971',  'Data20190503192839_214202', 'Data20190503192841_829674', 'Data20190503192850_195236', 'Data20190503193106_171620', 
            'Data20190503193655_400563', 'Data20190503193855_214786', 'Data20190503193921_646400',  'Data20190503193922_036345', 'Data20190503194031_443235', 'Data20190503194126_990864', 'Data20190503194149_626400', 
            'Data20190503194210_987780', 'Data20190503201503_440167', 'Data20190503201504_021074', 
            'Data20190503201514_248990', 'Data20190503201657_625951', 'Data20190508194624_238984', 'Data20190508194828_833326', 'Data20190508195341_224143', 'Data20190508195344_011205', 'Data20190508195506_427675', 
            'Data20190508195529_586374', 'Data20190508195532_567615', 'Data20190508195537_837030', 'Data20190508195938_998420', 'Data20190508195942_574684', 'Data20190508195946_005943', 'Data20190508200021_043592', 
            'Data20190508200057_595354', 'Data20190508200212_812952', 'Data20190508200232_425433', 'Data20190508200424_636763', 'Data20190508200608_023579', 'Data20190508200720_008026', 'Data20190508200720_990096', 
            'Data20190508200726_192491', 'Data20190508200745_622957', 'Data20190508201606_014887', 'Data20190508201609_512155', 'Data20190508201720_367536', 'Data20190508201720_561551', 'Data20190508201722_785721', 
            'Data20190508202335_393888', 'Data20190508202515_761158', 'Data20190508202548_970672', 'Data20190508203008_392678', 'Data20190508203831_384722', 'Data20190508203836_396535', 'Data20190508203836_511564', 
            'Data20190508203836_615552', 'Data20190508203840_993368', 'Data20190508203844_016590', 'Data20190508203846_393197', 'Data20190508203848_824216', 'Data20190508203904_175216', 'Data20190508204003_625496', 
            'Data20190508204136_741984', 'Data20190508204207_019110', 'Data20190508204210_424952', 'Data20190508204210_841099', 'Data20190508204211_041130', 'Data20190508204216_776388', 'Data20190508204225_394834', 
            'Data20190508204240_590973', 'Data20190508204334_234383', 'Data20190508204337_117669', 'Data20190508204451_017767', 'Data20190508204513_787731', 'Data20190508204517_820419', 'Data20190508204523_823041', 
            'Data20190508204938_901319', 'Data20190508210930_782731', 'Data20190508211623_151915', 'Data20190508211623_360933', 'Data20190508211623_745959', 'Data20190508211626_557486', 'Data20190508211626_971518', 
            'Data20190508211628_969667', 'Data20200406183056_611998N850F12', 'Data20200406183057_442211N850F12', 'Data20200425183316_037235N850F12', 'Data20200521191807_323937N850F12', 'Data20200521192115_543171N850F12',
             'Data20200521192116_234713N850F12', 'Data20200521192118_730387N850F12', 'Data20200521192119_793974N850F12', 'Data20200521192211_481258N850F12', 'Data20200521192315_747927N850F12', 
             'Data20200521192316_094445N850F12', 'Data20200521192317_170024N850F12', 'Data20200521192317_530052N850F12', 'Data20200521192318_241104N850F12', 'Data20200521192320_006235N850F12', 
             'Data20200521192520_343525N850F12', 'Data20200521192603_578697N850F12', 'Data20200521192745_548200N850F12', 'Data20200521192745_879716N850F12', 'Data20200521192845_036077N850F12', 
             'Data20200521193049_499636N850F12', 'Data20200521193050_946739N850F12', 'Data20200521193054_796540N850F12', 'Data20200524200653_991997N850F12', 'Data20200525213646_587840N850F12', 
             'Data20200525214923_846580N850F12', 'Data20200525214926_587025N850F16', 'Data20200525214930_768084N850F12', 'Data20200528195246_028374N850F12', 'Data20200528195247_024946N850F12', 
             'Data20200528213556_161795N850F12', 'Data20200529203004_065387N850F12', 'Data20200529203225_511953N850F12', 'Data20200529203253_332032N850F12', 'Data20200529203326_002972N850F12', 
             'Data20200529203503_971293N850F12', 'Data20200529203507_146029N850F12', 'Data20200529203632_876936N850F12', 'Data20200529203907_251470N850F12', 'Data20200529204501_497436N850F12', 
             'Data20200529204611_499666N850F12', 'Data20200529204714_149848N850F12', 'Data20200529204735_359932N850F12', 'Data20200529205229_449766N850F12', 'Data20200529210843_809278N850F12',
            'Data20200607194327_710665N850F12',  'Data20200624201750_525208N850F12', 'Data20200624201751_568282N850F12', 'Data20200624202807_727540N850F12', 'Data20200624202824_811294N850F12',
             'Data20200624202833_785954N850F12','Data20200624202834_470504N850F12', 'Data20200624202931_983730N850F12', 'Data20200624202934_080382N850F12', 'Data20200624202947_355855N850F12',
            'Data20200624203051_122542N850F12', 'Data20200624203051_476068N850F12', 'Data20200624203130_894197N850F12', 'Data20200624203135_221518N850F12', 'Data20200624203135_568541N850F12',
             'Data20200624203159_017264N850F12',
             'Data20200624203208_282444N850F12', 'Data20200624203208_999500N850F12', 'Data20200624203209_355521N850F12', 'Data20200624203209_759551N850F12', 'Data20200624203210_534617N850F12', 
             'Data20200624203210_876135N850F12', 'Data20200624203211_993714N850F12', 'Data20200624203214_465399N850F12', 'Data20200624203216_214025N850F12', 'Data20200624203216_559051N850F12', 
             'Data20200624203220_025321N850F12', 'Data20200624203220_687353N850F12', 'Data20200624203222_042957N850F12', 'Data20200624203223_822084N850F12', 'Data20200624203231_796170N850F12', 
             'Data20200624203238_420659N850F12', 'Data20200624203238_751680N850F12', 'Data20200624203243_323018N850F12', 'Data20200624203243_667043N850F12', 'Data20200624203252_558193N850F12', 
             'Data20200624203326_248170N850F12', 'Data20200624203327_288748N850F12', 'Data20200624203329_738427N850F12', 'Data20200624203336_953455N850F12', 'Data20200624203351_004986N850F12', 
             'Data20200624203409_154321N850F12', 'Data20200624203409_495844N850F12', 'Data20200624203410_548424N850F12', 'Data20200624203411_232471N850F12', 'Data20200624203412_602576N850F12', 
             'Data20200624203449_819688N850F12', 'Data20200624203450_895264N850F12', 'Data20200624203506_649430N850F12', 'Data20200624203516_182125N850F12', 'Data20200624203516_536649N850F12', 
             'Data20200630195032_491073N850F12', 'Data20200630195057_940947N850F12', 'Data20200630195158_003865N850F12', 'Data20200630195158_383961N850F12', 'Data20200630195211_287956N850F12', 
             'Data20200630195229_598454N850F12', 'Data20200630195252_523371N850F12', 'Data20200630195253_273562N850F12', 'Data20200630195317_061169N850F12', 'Data20200630195339_046515N850F12', 
             'Data20200630195340_560359N850F12', 'Data20200630195858_142920N850F12', 'Data20200630195859_366765N850F12', 'Data20200630195901_440802N850F12', 'Data20180702194335_461240N850F12', 
             'Data20180702194335_799271N850F12', 'Data20180702194336_877848N850F12', 'Data20180702194337_945933N850F12', 'Data20180702194346_560582N850F12', 'Data20180702194347_963171N850F12', 
             'Data20180702194348_659725N850F12', 'Data20180702194349_362776N850F12', 'Data20180702195328_807896N850F12', 'Data20180702195329_491939N850F12', 'Data20180702195350_520497N850F12', 
             'Data20180702195352_170120N850F12', 'Data20180710193805_531310N850F12', 'Data20180710193811_691271N850F12', 'Data20180710193821_372988N850F12']
'''
            # TODO: Exception: 'a' cannot be empty unless no samples are taken in image:
            self.empty_imgIds = ['Data20200425183316_037235N850F12',
                'Data20190508201609_512155',
                'Data20200521192520_343525N850F12', 'Data20200521192745_548200N850F12', 'Data20200521193049_499636N850F12', 'Data20200521193050_946739N850F12',
                  'Data20200529203907_251470N850F12', 'Data20200529204501_497436N850F12',
                  'Data20200607194327_710665N850F12',
                 'Data20200624203351_004986N850F12',
                  'Data20200630195340_560359N850F12', 'Data20200630195858_142920N850F12',
                 'Data20180702194335_461240N850F12', 'Data20180702194335_799271N850F12', 'Data20180702194337_945933N850F12', 'Data20180702194346_560582N850F12', 'Data20180702194348_659725N850F12',
                 'Data20180702194349_362776N850F12', 'Data20180702195328_807896N850F12',
                'Data20180710193811_691271N850F12']
        else:
            print('Notion: unknown training dataset!')
            sys.exit(0)  # 干净利落地退出系统

        self.training_loss_file = os.path.join(self.model_dir, 'Training_loss_%s.json' % self.Datasets)
        self.IoUs_Cls_original_list_file = os.path.join(self.model_dir, 'IoUs_Cls_original_list_%s.json' % self.Datasets)
        self.IoUs_Cls_list_file = os.path.join(self.model_dir, 'IoUs_Cls_list_%s.json' % self.Datasets)
        self.IoUs_RPN_original_list_file = os.path.join(self.model_dir, 'IoUs_RPN_original_list_%s.json' % self.Datasets)
        self.IoUs_RPN_list_file = os.path.join(self.model_dir, 'IoUs_RPN_list_%s.json' % self.Datasets)

        self.Loss0 = 0.05  # 训练总损失目标值