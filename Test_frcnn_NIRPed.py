from __future__ import division
import os, pdb
import cv2, json, sys
import tensorflow as tf
import numpy as np
import pickle
import time, datetime
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import keras_frcnn.roi_helpers_Ln as roi_helpers
from keras_frcnn import Resnet50RGB64_1024_128_2o5stride8 as nn  # 从keras_frcnn模块包中的resnet_NIR.py文件中定义的所有方法(或函数)导入为nn
import argparse  # argparse模块可以轻松编写用户友好的命令行界面。 程序定义了它需要的参数，argparse将弄清楚如何解析sys.argv中的参数。
import shutil
from keras_frcnn.coco import COCO

from keras_frcnn import config
cfg = config.Config()  # 实例化config.py文件中的类Config，存储到变量cfg中
cfg.use_horizontal_flips = False
cfg.use_vertical_flips = False
cfg.rot_90 = False

max_boxes = 300
IoU_threshold_rpn = 0.70
score_threshold_rpn = 0.5
IoU_threshold_cls = 0.50
score_threshold_cls = 0.001
subset = 'val'
# subset = 'test'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定第一个GPU参与运算。
gpu_cfg = tf.compat.v1.ConfigProto()
gpu_cfg.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用GPU90%的显存
session = tf.compat.v1.Session(config=gpu_cfg)

cfg.model_path = os.path.join(cfg.model_dir, 'WeightsResnet50NIR1RGB64_1024_128_2o5.h5')

input_shape_img = (None, None, 3)  # 灰度图像只有一个channel
input_shape_features = (None, None, 1024)  # (32, 40, 1024)

class_mapping = cfg.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}

img_input = Input(shape=input_shape_img)  # input_shape_img = (None, None, 1)
roi_input = Input(shape=(cfg.num_rois, 4))  # roi_input=(32, 4)
feature_map_input = Input(shape=input_shape_features)  # (32, 40, 1024)

shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers 定义基于基础层构建的RPN
num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)  # 计算特征图上一个点的锚框数
rpn_layers = nn.rpn(shared_layers, num_anchors)  # rpn_layers = [x_class, x_regr, base_layers=shared_layers]
classifier = nn.classifier(feature_map_input, roi_input, cfg.num_rois, nb_classes=len(class_mapping), trainable=True)
model_rpn = Model(img_input, rpn_layers)  # 模型输入为img_input, 输出为rpn_layers=[x_class, x_regr, base_layers=shared_layers]
model_classifier_only = Model([feature_map_input, roi_input], classifier)
model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(cfg.model_path))
model_rpn.load_weights(cfg.model_path, by_name=True)  # 根据网络层名称，加载训练好的网络权值
model_classifier.load_weights(cfg.model_path, by_name=True)

# model_rpn.compile(optimizer='Adagrad', loss='mse')  #此处不需要对网络进行训练了，损失函数与训练时的不一样!
# model_classifier.compile(optimizer='Adagrad', loss='mse')  # 编译分类预测网络，与训练网络不同
model_rpn.compile(optimizer='sgd', loss='mse')  #此处不需要对网络进行训练了，损失函数与训练时的不一样!
model_classifier.compile(optimizer='sgd', loss='mse')  # 编译分类预测网络，与训练网络不同

Pr_path = "./results_miniNIRPed/dt_results_%s_B%d_%s" % (subset, max_boxes, str(score_threshold_cls)[2:])
if not os.path.exists(Pr_path):  # if it exist already
    # shutil.rmtree(Pr_path)     # reset the results directory 重置结果目录,也就是删除现有的文件夹images-optional
    os.makedirs(Pr_path)  # 在程序当前目录下创建一个新的文件夹名为results_files_path = "results"

time_cost_file = os.path.join(Pr_path, 'time_cost_list_{}.json'.format(subset))
if os.path.exists(time_cost_file):
    file_obj = open(time_cost_file, 'r')
    time_cost_list = json.load(file_obj)
    file_obj.close()
    num_imgs_initial = len(time_cost_list)
else:
    time_cost_list = []
    num_imgs_initial = 0

bounding_boxes_file = os.path.join(Pr_path, 'DtResults_{}.json'.format(subset))
if os.path.exists(bounding_boxes_file):
    file_obj = open(bounding_boxes_file, 'r')
    bounding_boxes = json.load(file_obj)
    file_obj.close()
else:
    bounding_boxes = []

time_now_day = datetime.datetime.now().strftime('%Y%m%d')

def format_img_size(img):  # 对原始图像进行前处理：缩放。
    """ formats the image size based on config """  # 根据配置格式化图像大小
    resized_height = int(cfg.im_rows)
    resized_width = int(cfg.im_cols)
    (height, width, _) = img.shape #原始图像img=(720,1280),故：height=720, width=1280
    ratio = [resized_height / height, resized_width/width] # 缩放比例ratio = 512 / 720 = 0.5
    img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)  # 对原始图像按新的尺寸进行缩放
    return img, ratio  # 返回缩放后的图像img=(512,640)，以及缩放比例ratio = 0.5

def format_img_channels(img):  # 对原始图像img=(720,1280)色彩通道进行前处理：去均值。
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= cfg.img_channel_mean[0]
    img[:, :, 1] -= cfg.img_channel_mean[1]
    img[:, :, 2] -= cfg.img_channel_mean[2]
    img /= cfg.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)  # 将图片的维度变成了(1,channels=1,512,640)
    return img

def format_img(img):  #对原始图像进行前处理。
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img)  # 对原始图像img=(720,1280)进行前处理：缩放，返回缩放后的图像img=(512,640)
    img = format_img_channels(img)  # 对缩放后的图像img=(512,640)色彩通道进行前处理：去均值，且维度变成了(1,channels=1,512,640)
    return img, ratio  # 返回前处理后的图像img=(1,channels=1,512,640)，以及缩放比例ratio = 0.5

# Method to transform the coordinates of the bounding box to its original size 将边界框的坐标转换为其原始大小的方法
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio[1]))
    real_y1 = int(round(y1 // ratio[0]))
    real_x2 = int(round(x2 // ratio[1]))
    real_y2 = int(round(y2 // ratio[0]))

    return real_x1, real_y1, real_x2, real_y2

def predict_single_image(img_path):  # 预测单张图片中的目标
    st0 = time.time()  # 开始计时
    img = cv2.imread(img_path)
    #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 原始图像img=(720,1280)
    if img is None:  # 如果未读入图像数据
        print('reading image failed.')
        exit(0)  # exit(0)：意味着一个干净的出口，没有任何错误/问题  exit(1)：这意味着存在一些问题/错误/问题，这就是程序退出的原因。
    X_reImg, ratio = format_img(img)  # 返回前处理后的图像X_reImg：X_reImg=(Samples=1,channels=1,rows=512,cols=640)，以及缩放比例ratio = 0.5

    if K.image_data_format() == "channels_last":
        X_reImg = np.transpose(X_reImg, (0, 2, 3, 1))  # 转换图像X_reImg维度：X_reImg=(Samples=1,rows=512,cols=640,channels=3)

    img_id = os.path.basename(img_path).split(".", 1)[0]
    time_pre_rpn = time.time()
    # get the feature maps and output from the RPN 从RPN获取特征图和输出
    [Pcls_rpn, Pregr_rpn, Xbase_layers] = model_rpn.predict(X_reImg, verbose=0)  # RPN分类预测Pcls_rpn=(1, 32, 40, 10),RPN边界框回归坐标预测Pregr_rpn=(1, 32, 40, 40)，
    time_rpn_gpu = time.time()
    result = roi_helpers.rpn_to_roi(Pcls_rpn, Pregr_rpn, cfg, K.image_data_format(), overlap_thresh=IoU_threshold_rpn, max_boxes=max_boxes)#TODO：RPN的交并比阈值overlap_thresh=0.7 max_boxes=300

    result[:, 2] -= result[:, 0]
    result[:, 3] -= result[:, 1]
    #TODO: result即为RPN提案的前300个大概率RoIs,但坐标已变换到特征图上的：RoI300_rpn = (300, (4 + 1)= (x1, y1, x2, y2))

    time_pre_cls = time.time()
    # apply the spatial pyramid pooling to the proposed regions 将空间金字塔池应用于区域提案
    boxes = dict()  # 创建一个新的字典用于存放边界框回归坐标以及分类概率
    for jk in range(result.shape[0] // cfg.num_rois + 1):  # 循环jk=0,1,2,3,...,9   300 // 32 + 1 = 10
        rois = np.expand_dims(result[cfg.num_rois * jk:cfg.num_rois * (jk + 1), :], axis=0)  # 分10组rois(最后一组只有12个)，并扩维成(Samples=1，num_rois=32, (4+1))
        if rois.shape[1] == 0:  # 防止最后一组没有数据
            break
        if jk == result.shape[0] // cfg.num_rois:  # =300//32=9 如果是最后一组，填充第1维12个成32个
            curr_shape = rois.shape
            target_shape = (curr_shape[0], cfg.num_rois, curr_shape[2])
            rois_padded = np.zeros(target_shape).astype(rois.dtype)
            rois_padded[:, :curr_shape[1], :] = rois
            rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]  # 用最后一组的第0个RoI坐标填充第1维后面32-12=20个RoIs
            rois = rois_padded
        [Pcls_cls, Pregr_cls] = model_classifier_only.predict([Xbase_layers, rois], verbose=0)  # 分组预测，加快计算速度。注意：预测只输出分类预测和坐标回归预测两项
        #返回：最后锚框分类：Pcls_cls = out_class=(Samples=1, num_rois=32, nb_classes=2)；#假定分2类[行人得分，背景得分]
        #最后锚框坐标回归：Pregr_cls = out_regr=(Samples=1, num_rois=32, 4 * (nb_classes - 1)=4)#假定分2类(回归坐标不包括背景bg)
        #最后锚框坐标回归：Pregr_cls = out_regr=(Samples=1, num_rois=32, (4+1) * (nb_classes - 1)=(4+1))#假定分2类(回归坐标不包括背景bg)

        Pcls_cls[0, :, 1] -= 1 - 2 * score_threshold_cls
        for ii in range(Pcls_cls.shape[1]):  # 遍历当前组32个RoIs对目标行人预测概率
            if np.argmax(Pcls_cls[0, ii, :]) == (Pcls_cls.shape[2] - 1): #TODO:最大置信度为背景，则跳过本次循环。
                continue  # 跳过当前循环的后续语句，继续循环。

            cls_num = np.argmax(Pcls_cls[0, ii, :])  # 取出分类预测概率最大值的索引
            if cls_num not in boxes.keys():  # 如果分类预测概率最大值的索引不在边界框boxes的键中。
                boxes[cls_num] = []  # 增加一个键cls_num对应的类

            (x, y, w, h) = rois[0, ii, :]  # 取出当前组当前RoI在特征图上的坐标
            try:
                (tx, ty, tw, th, td) = Pregr_cls[0, ii, 5 * cls_num:5 * (cls_num + 1)]  # 取出当前组当前RoI的分类cls_num的坐标回归系数
                # cfg.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]#边界框回归标准差系数
                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]
                Dis_cls = round(cfg.Dis_mean*np.exp(-td), 2)
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                # 执行坐标回归计算：返回修正后的特征图上RoI左上角点坐标及宽度圆整值(像素点)
            except Exception as e:  # 抛出异常但不中断程序，继续执行except后续语句
                print(e)
                pass  # 空语句，是为了保持程序结构的完整性。pass不做任何事情，一般用做占位语句。
            boxes[cls_num].append([cfg.rpn_stride * x, cfg.rpn_stride * y, cfg.rpn_stride * (x + w), cfg.rpn_stride * (y + h), Dis_cls, np.max(Pcls_cls[0, ii, :])])
            #将RoI坐标从特征图返回到预处理的前处理后图上cls_num:(x, y, w, h,  Dis_rpn, Dis_cls, Pcls )

    time_cls_gpu = time.time()

    for cls_num, box in boxes.items():  #遍历边界框boxes的键(分类)：值(边界框坐标及概率)=(x, y, w, h, P)
        boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=IoU_threshold_cls)  # 最后一组的第0个RoI坐标填充的20个RoIs会被抑制掉
        # 返回类cls_num通过快速非最大抑制过的边界框boxes=[x1, y1, x2, y2, Dis_C, prob]
        boxes[cls_num] = boxes_nms  # 更新类cls_num的边界框
        class_name = class_mapping[cls_num]
        for Bb in boxes_nms:
            Bb[0], Bb[1], Bb[2], Bb[3] = get_real_coordinates(ratio, Bb[0], Bb[1], Bb[2], Bb[3])  # 从前处理图返回原始图像，获取真实坐标值：ratio = 0.5
            bounding_boxes.append({'class_name': class_name, 'confidence': str(np.round(Bb[-1], decimals=6)), 'file_id': img_id, 'bbox': list(Bb[0:4]), 'Dis': Bb[4]})
    time_post_cls = time.time()
    time_all = time_post_cls-st0
    time_post_cls = time_post_cls - time_cls_gpu
    time_cls_gpu = time_cls_gpu - time_pre_cls
    time_pre_cls = time_pre_cls - time_rpn_gpu
    time_rpn_gpu = time_rpn_gpu - time_pre_rpn
    time_pre_rpn = time_pre_rpn - st0

    time_cost = [round(time_pre_rpn, 6), round(time_rpn_gpu, 6), round(time_pre_cls, 6), round(time_cls_gpu, 6), round(time_post_cls, 6), round(time_all, 6)]
    time_cost_list.append(time_cost)
    # print('[time_pre_rpn,time_rpn_gpu,time_pre_cls,st_cls_gpu/st_cls,time_post_cls, time_all]=[%6f,%6f,%6f,%.6f/%.6f,%6f,%6f]' % (
    # time_pre_rpn, time_rpn_gpu, time_pre_cls, st_cls_gpu, time_cls_gpu, time_post_cls, time_all))
    return None

def predict(test_images_json):  # 预测args_指定的图片
    cocoGt = COCO(test_images_json)
    imgIds = sorted(cocoGt.getImgIds())  # imgIds=[100013, 100024, 100063, 100065,...]
    num_imgs_all = len(imgIds)
    for index_image in range(num_imgs_all):
        if index_image < num_imgs_initial-1:
            continue
        image = cocoGt.loadImgs(ids=imgIds[index_image])[0]
        img_name = image['file_name']     # file_path='\\58c58285bc26013700140940.png'
        # test_img_path = os.path.join('E:\\Datasets\\NIRPed2021\\NIRPed\\images\\{}\\{}'.format(subset, img_name))
        # test_img_path = '.\\data\\NIRPed\\images\\{}\\{}'.format(subset, img_name)
        test_img_path = '.\\data\\miniNIRPed\\images\\{}\\{}'.format(subset, img_name)

        if test_img_path == None:
            print('Notion:{} do not exist.'.format(test_img_path))
            continue
        else:
            predict_single_image(test_img_path)

        if index_image % 50 == 0:
            print('N={}/{}:{} exist.'.format(index_image + 1, num_imgs_all, test_img_path))  # 打印预测图像的名称

            bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)  # sort detection-result by decreasing confidence

            outfile = open(os.path.join(Pr_path,  'DtResults_%s.json' %(subset)), 'w')
            json.dump(bounding_boxes, outfile)
            outfile.close()
            outfile = open(os.path.join(Pr_path, 'time_cost_list_%s.json' %(subset)), 'w')
            json.dump(time_cost_list, outfile)
            outfile.close()

    bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)  # sort detection-result by decreasing confidence

    outfile = open(os.path.join(Pr_path, 'DtResults_{}.json'.format(subset)), 'w')
    json.dump(bounding_boxes, outfile)
    outfile.close()

    mean_time_cost = np.round(np.mean(time_cost_list[1:], axis=0), 6)
    time_cost_list[0] = mean_time_cost.tolist()
    outfile = open(os.path.join(Pr_path, 'time_cost_list_{}.json'.format(subset)), 'w')
    json.dump(time_cost_list, outfile)
    outfile.close()

if __name__ == '__main__':
    if subset == 'val':
        print('Detecte subset {}'.format(cfg.val_file))
        predict(cfg.val_file)
    elif subset == 'test':
        print('Detecte subset {}'.format(cfg.test_file))
        predict(cfg.test_file)