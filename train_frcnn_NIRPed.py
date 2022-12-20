""" this code will train on NIRPed data set。 """
from __future__ import division  # 导入python未来支持的语言特征division(精确除法)，当我们没有在程序中导入该特征时，"/"操作符执行的是
import pdb, glob, re
import os, time, sys,  shutil  # os模块提供了非常丰富的方法用来处理文件和目录。
import numpy as np
np.set_printoptions(precision=6, threshold=np.inf, edgeitems=10, linewidth=260, suppress=True)
import tensorflow as tf
from keras import backend as K  # 我们的backend 是 TensorFlow
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
from keras.layers import Input  # Input(shape=None,batch_shape=None,name=None,dtype=K.floatx(),sparse=False,
from keras.models import Model
from keras_frcnn import config, data_generators_new  # 从keras_frcnn模块包中导入config.py和data_generators文件以及其中的函数、类及方法
from keras_frcnn import losses_distance as losses_fn  # 从keras_frcnn模块包中导入losses.py文件以及其中的函数、类及方法为losses_fn
import keras_frcnn.roi_helpers_Ln as roi_helpers
from get_data_from_json import get_data  # 从keras_frcnn模块包中的simple_parser.py文件中定义的解析记事本标注数据的方法(或函数)get_data
from keras.utils.vis_utils import plot_model
from keras_frcnn.Visualize_RPN_Cls import Visual_RPN_Train, Visual_RPN_Predict, Visual_Cls_Train, Visual_Cls_Predict
import json
import matplotlib.pyplot as plt
from select_samples import select_samples_Multi_task
from model_classifier_predict import model_classifier_predict
from keras_frcnn import Resnet50RGB64_1024_128_2o5stride8 as nn

font = {'family': 'SimSun', 'weight': 'normal', 'size': 18}
font_size = 16
tick_size = 14
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用简体仿宋显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def train_NIR():  # 定义基于Kitti数据集的网络训练函数
	# config for data argument
	cfg = config.Config()  # 实例化config.py文件中的类Config，存储到变量cfg中
	cfg.base_net_weights = cfg.model_path
	train_dataset, train_loss, classes_count, class_mapping = get_data(cfg)

	cfg.class_mapping = class_mapping # 将字典class_mapping赋给类cfg的变量class_mapping
	cfg.classes_count = classes_count  # 将字典class_mapping赋给类cfg的变量class_mapping
	print('Training images per class:')
	print(classes_count)
	print(class_mapping)
	print('Num classes (including Ign and bg) = {}'.format(len(classes_count)))  # 打印出：目标分类个数 = len(classes_count)
	print('Number of training images {}'.format(len(train_dataset)))  # 打印训练图片(样本)的张数，即：列表train_imgs的长度

	show_imgs_list = []
	if os.path.exists(cfg.show_imgs_directory):
		show_imgs_list = glob.glob(cfg.show_imgs_directory + '/*.png')
		p = re.compile('m.png|g.png')  # 正则化
		show_imgs_list = [os.path.basename(img) for img in show_imgs_list if not p.findall(img)]

	data_gen_train = data_generators_new.get_anchor_gt(train_dataset, classes_count, cfg, nn.get_img_output_length, K.image_data_format(), mode='train')  # 产生用于训练的真实锚框(ground truth)数据：

	if K.image_data_format() == 'channels_first':  # 获取当前的维度顺序：th则是(channels=1, Samples=1, conv_dim2, conv_dim3)
		input_shape_img = (len(cfg.img_channel_mean), None, None)  # 如果是RGB图像len(cfg.img_channel_mean)=3
	else:  # 获取当前的维度顺序：tf假定(Samples=1, conv_dim2, conv_dim3, channels=1)
		input_shape_img = (None, None, len(cfg.img_channel_mean))  # 如果是Gray图像len(cfg.img_channel_mean)=1

	img_input = Input(shape=input_shape_img)  # 返回图像输入的张量img_input.shape=(Samples=1, 256, 640, channels=1),元组型.实例化一个keras张量
	roi_input = Input(shape=(None, 4))  # (4+1) 返回RoI输入的张量roi_input.shape=(Samples=1, num_rois=32, [x,y,w,h,Dis]),元组型.

	# define the base network (resnet here, can be VGG, Inception, etc)
	shared_layers = nn.nn_base(img_input, trainable=True)

	# define the RPN, built on the base layers定义基于基础层构建的RPN
	# TODO:用k-means方法求得3*3=9个先验框
	num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)  # 计算特征图上每个像素点上的锚框个数num_anchors=5*1=5

	[Pcls_rpn, Pregr_rpn, Xbase_layers] = nn.rpn(shared_layers, num_anchors)  # 构建RPN网络的函数，

	nb_classes = len(class_mapping)  #nb_classes=2
	classifier = nn.classifier(shared_layers, roi_input, cfg.num_rois, nb_classes=nb_classes, trainable=True)  # 继续分类
	model_rpn = Model(img_input, [Pcls_rpn, Pregr_rpn])
	model_classifier = Model([img_input, roi_input], classifier)  # 建立分类网络模型

	# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
	model_all = Model([img_input, roi_input], [Pcls_rpn, Pregr_rpn] + classifier)  # 建立网络总模型：Pcls_rpn.shape!=classifier[0].shape????????????
	try:
		print('loading weights from {}'.format(cfg.base_net_weights))
		model_rpn.load_weights(cfg.model_path, by_name=True)  # 根据层名称导入RPN和基础网络权值
		model_classifier.load_weights(cfg.model_path, by_name=True)  # 根据 层名称导入检测网络权值
		num_epochs0 = len(train_loss)
		train_loss_array = np.around(np.array(train_loss), decimals=6)
		if num_epochs0 == 0:
			best_loss = 0.5
			print('Start training from zero')
		else:
			best_loss = min(train_loss_array[:, 4])  # numpy里的无穷大1.7976931348623157e+308
			curr_loss = train_loss_array[-1, 4]
			print('best_loss=%.2f%%' % np.array(100*best_loss))

	except Exception as e:
		print(e)
		num_epochs0 = 0
		best_loss = 0.5
		curr_loss = best_loss
		train_loss_array = np.around(np.array(train_loss), decimals=6)
		print('Could not load pretrained model weights. Weights can be found in the keras application folder https://github.com/fchollet/keras/tree/master/keras/applications')


	learning_rate = cfg.learning_rate

	learning_rate_rpn = learning_rate
	learning_rate_cls = learning_rate
	learning_rate_all = learning_rate
	print('initial learning_rate = {}'.format(learning_rate))
	optimizer = Adam(lr=learning_rate, epsilon=0.1*learning_rate)  # 选用Adam优化器：梯度下降改进版
	optimizer_classifier = Adam(lr=learning_rate, epsilon=0.1*learning_rate)  # 分类选用Adam优化器

	model_rpn.compile(optimizer=optimizer, loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])  # 编译出损失需要加入距离损失信息

	model_classifier.compile(optimizer=optimizer_classifier, loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(nb_classes - 1)], metrics={'dense_class_{}'.format(nb_classes): 'accuracy'})
	print(model_classifier.summary())
	model_all.compile(optimizer='sgd', loss='mae')  # 随机梯度下降算法SGDA=stochastic gradient descent algorithm
	model_all_architectures = os.path.join(cfg.model_dir, 'NIRPed_%s.png' % ( os.path.basename(os.path.normpath(cfg.model_path0))))
	if not os.path.exists(model_all_architectures):
		print(model_all.summary())
		plot_model(model_all, to_file=model_all_architectures, show_shapes=True)

	epoch_length = cfg.length_epoch  # 1000(即：1000个batch，每个batch为一张图片)迭代算算一个epoch, 求一次损失平均值
	num_epochs1 = int(cfg.num_epochs) - num_epochs0  # 训练回合,在config中设置为3000
	iter_num = 0  # 迭代次数开始计数iter_num
	losses = np.zeros((epoch_length, (4 + 1)))
	rpn_accuracy_rpn_monitor = []  # 将RPN精度监视器rpn_accuracy_rpn_monitor置成空列表
	rpn_accuracy_for_epoch = []  # 每一回合RPN的平均精度
	start_time = time.time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）

	print('\033[1;30;43m Starting training,initial classifier_min_overlap={}\033[0m'.format(cfg.classifier_min_overlap))
	training_cls = 0
	for epoch_num1 in range(num_epochs1):  # 创建一个整数列表：range(0, 5, 1)=[0, 1, 2,……,2998, 2999]
		epoch_num = epoch_num1 + num_epochs0
		num_epochs = num_epochs1 + num_epochs0
		print('\033[1;35m Epoch {}/{}\033[0m'.format(epoch_num + 1, num_epochs))  # 打印：Epoch epoch_num + 1/num_epochs

		if learning_rate_rpn < 1e-10 and learning_rate_cls < 1e-10 and learning_rate_all < 1e-10:
			curr_loss_round = '%.3g' % curr_loss  # curr_loss0 = np.round(curr_loss, decimals=4)
			best_model_path = '{}L{}Dis{}Ped{}.h5'.format(cfg.model_path0, str(curr_loss_round)[2:], cfg.Dis0, classes_count[training_cls])
			model_all.save_weights(best_model_path)
			model_all.save_weights(cfg.model_path)
			break
		if (best_loss < 0.05 or epoch_num > 1000) and cfg.use_bg_imgs == False:
			cfg.use_bg_imgs = True

		times_increase_max = int(len(train_dataset) / cfg.length_epoch)

		if epoch_num > 2*times_increase_max and int(epoch_num % times_increase_max) == 0:
			if np.mean(train_loss_array[-int(times_increase_max):, :2]) >= 0.975 * np.mean(train_loss_array[-int(2*times_increase_max):-int(times_increase_max), :2]):  # train_loss_array
				learning_rate_rpn = 0.1 * learning_rate_rpn
				print('\033[1;30;43m Current learning_rate_rpn={} and times_increase={:d}/{:d}\033[0m'.format(learning_rate_rpn, epoch_num % times_increase_max, times_increase_max))
				# TODO: set new lr:K.set_value(model_rpn.optimizer.lr, learning_rate_rpn) # model_rpn.optimizer.lr = learning_rate_rpn
				K.set_value(model_rpn.optimizer.lr, np.float32(learning_rate_rpn))
				model_rpn.optimizer.epsilon = 0.1 * learning_rate_rpn  # TODO: K.set_value(model_rpn.optimizer.epsilon, 0.1*learning_rate_rpn)
			if np.mean(train_loss_array[-int(times_increase_max):, 2:4]) >= 0.975 * np.mean(train_loss_array[-int(2*times_increase_max):-int(times_increase_max), 2:4]):  # train_loss_array
				learning_rate_cls = 0.1 * learning_rate_cls
				print('\033[1;30;43m Current learning_rate_cls={} and times_increase={:d}/{:d}\033[0m'.format(learning_rate_cls, epoch_num % times_increase_max, times_increase_max))
				# TODO: set new lr:K.set_value(model_rpn.optimizer.lr, learning_rate_cls) # model_rpn.optimizer.lr = learning_rate_cls
				K.set_value(model_rpn.optimizer.lr, np.float32(learning_rate_cls))
				model_rpn.optimizer.epsilon = 0.1 * learning_rate_cls  # TODO: K.set_value(model_rpn.optimizer.epsilon, 0.1*learning_rate_cls)
			if np.mean(train_loss_array[-int(times_increase_max):, 4]) >= 0.975 * np.mean(train_loss_array[-int(2 * times_increase_max):-int(times_increase_max), 4]):  # train_loss_array
				learning_rate_all = 0.1 * learning_rate_cls
				print('\033[1;30;43m Current learning_rate_all={} and times_increase={:d}/{:d}\033[0m'.format(learning_rate_all, int(epoch_num % times_increase_max), times_increase_max))
				# TODO: set new lr:K.set_value(model_rpn.optimizer.lr, learning_rate_all) # model_rpn.optimizer.lr = learning_rate_all
				K.set_value(model_all.optimizer.lr, np.float32(learning_rate_all))
				model_all.optimizer.epsilon = 0.1 * learning_rate_all

		if epoch_num > 10 and epoch_num % 5 == 0:
			if np.mean(train_loss_array[-5:, 2:4]) <= 0.95 * np.mean(train_loss_array[-10:-5, 2:4]):  # train_loss_array
				cfg.classifier_min_overlap = min(cfg.classifier_max_overlap, cfg.classifier_min_overlap + 0.01)
				# print('\033[34;42m Increase classifier_min_overlap to {} \033[0m'.format(cfg.classifier_min_overlap))
				print('\033[30;41m Increase classifier_min_overlap to {} \033[0m'.format(cfg.classifier_min_overlap))
			elif np.mean(train_loss_array[-5:, 2:4]) >= 1.05 * np.mean(train_loss_array[-10:-5, 2:4]):
				cfg.classifier_min_overlap = max(0.25, cfg.classifier_min_overlap - 0.01)
				print('\033[30;41m Decrease classifier_min_overlap to {}\033[0m'.format(cfg.classifier_min_overlap))
			else:
				print('\033[30;41m Keep classifier_min_overlap for {}\033[0m'.format(cfg.classifier_min_overlap))
		if epoch_num > 10:
			if np.random.randint(0, 5) == 0:
				cfg.rpn_is_Ok = False
			else:
				if train_loss_array[-1, 0] >= 0.005 or train_loss_array[-1, 1] >= 0.005:
					cfg.rpn_is_Ok = False
				else:
					cfg.rpn_is_Ok = True
		else:
			cfg.rpn_is_Ok = False

		exist_obj_imgs_count = 0
		Classifier_ignored_image_list = []
		while True:  # 开始根据训练图像样本进行无限循环
			try:  # 尝试try后至下一个except之间的语句，并检测是否异常，无异常忽略后面的except语句，异常就返回异常类型，并执行except后的语句
				# 每回合结束时，检查RPN精度监视列表的长度，如果迭代len(rpn_accuracy_rpn_monitor)=1000， 且 cfg.verbose = True（显示详细信息为真）
				if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
					mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)  # 求取重叠边界框个数的均值
					mean_overlapping_bboxes_exist_obj_imgs = float(sum(rpn_accuracy_rpn_monitor)) / exist_obj_imgs_count # 求取重叠边界框个数的均值
					print('Average number of overlapping bounding boxes from RPN = {} ({}) for {} previous iterations'.format(mean_overlapping_bboxes, mean_overlapping_bboxes_exist_obj_imgs, epoch_length))
					rpn_accuracy_rpn_monitor = []  # 将rpn_accuracy_rpn_monitor再次置成空列表

					# 对前epoch_length=1000次迭代，RPN = mean_overlapping_bboxe的重叠边界框的均值
					if mean_overlapping_bboxes == 0:  # RPN没有生成与真实框重叠的边界框。 检查RPN设置或继续训练。
						print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

				X_reImg, Ycls_rpn, Yregr_rpn, img_data, X_reImg0, IoUs_RPN, IoUs_RPN_original = next(data_gen_train)  # get_anchor_gt函数通过 next() 不断返回用于训练的一张图像样本数据，内存占用始终为常数。
				# TODO:***判断图像中是否存在训练目标***判断图像中是否存在训练目标***判断图像中是否存在训练目标***判断图像中是否存在训练目标
				GTs = img_data['bboxes']
				exist_obj = False
				for GT in GTs:
					cls_GT = GT['class']
					if cls_GT in ['Pedestrian', 'pedestrian', 'Ped', 'ped']:
						exist_obj = True
						break

				if exist_obj:
					training_cls = cls_GT
					exist_obj_imgs_count += 1
				else:
					if iter_num % 500 == 0 and training_cls != 0:
						if cfg.use_bg_imgs:
							print('No object ({}) in the image:{}--For training detection network at iter_num={}, is used to train network!'.format(training_cls, img_data['filepath'], iter_num))
						else:
							print('No object ({}) in the image:{}--For training detection network at iter_num={}, is not used to train network!'.format(training_cls, img_data['filepath'], iter_num))
							continue #TODO:没有任何目标的图片先不参加训练。
					else:
						if not cfg.use_bg_imgs:
							continue

				try:
					loss_rpn = model_rpn.train_on_batch(X_reImg, [Ycls_rpn,Yregr_rpn])  # X_reImg.shape=(Samples=1, rows=256, cols=640, Channels=1)
				except:
					pdb.set_trace()

				[Pcls_rpn, Pregr_rpn] = model_rpn.predict_on_batch(X_reImg)  # 注意：预测只输出分类预测和坐标回归预测两项
				BBdt300_rpn = roi_helpers.rpn_to_roi(Pcls_rpn, Pregr_rpn, cfg, K.image_data_format(), use_regr=True, overlap_thresh=0.9, max_boxes=300)
				X_BBs68, Ycls_BBs68, Yregr_BBs68, IouS, IoU_RoIs_original, gta_feature_map, ignore_feature_map = roi_helpers.calc_iou(BBdt300_rpn, img_data, cfg, class_mapping)

				if X_BBs68 is None:  # 如果优选RoI与图像中的标记框交并比都小于0.1，则将rpn_accuracy_rpn_monitor尾追加为0，rpn_accuracy_for_epoch尾追加为0
					rpn_accuracy_rpn_monitor.append(0)
					rpn_accuracy_for_epoch.append(0)
					if len(rpn_accuracy_rpn_monitor) == epoch_length-1:
						model_all.save_weights(cfg.model_pathe)

					img_name = os.path.basename(img_data['filepath'])
					Classifier_ignored_image_list.append(img_name)
					if len(Classifier_ignored_image_list) % 200 == 0:
						print("Detection network training phase is ignored for the %dth image (1000 images in total):'%s'" % (len(Classifier_ignored_image_list), img_name))

					continue

				sel_samples, count_pos_samples = select_samples_Multi_task(np.array(IouS), cfg)

				rpn_accuracy_rpn_monitor.append(count_pos_samples)  # rpn_accuracy_rpn_monitor等于正样本数
				rpn_accuracy_for_epoch.append(count_pos_samples)  # rpn_accuracy_for_epoch等于正样本数
				try:
					loss_class = model_classifier.train_on_batch([X_reImg, X_BBs68[:, sel_samples, 0:4]], [Ycls_BBs68[:, sel_samples, :], Yregr_BBs68[:, sel_samples, :]])
				except:
					pdb.set_trace()

				if show_imgs_list != []:
					img_name = os.path.basename(img_data['filepath'])
					if img_name in show_imgs_list:
						if not cfg.rpn_is_Ok:
							Visual_RPN_Train(X_reImg0, Ycls_rpn, img_data, cfg) #TODO:可视化训练RPN的正负锚框。
						Visual_RPN_Predict(X_reImg0, BBdt300_rpn, img_data, cfg) #TODO：取overlap_thresh=0.9999，预测概率大于0.5 的作为RPN 网络对目标存在的预测框。
						Visual_Cls_Train(X_reImg0, X_BBs68[:, sel_samples, 0:4], Ycls_BBs68[:, sel_samples, :], img_data, gta_feature_map, ignore_feature_map, cfg)  #TODO:可视化训练Classifier网络的正、负样本框（由RPN提供并回归过）。
						boxes_dt = model_classifier_predict(model_classifier, X_reImg, BBdt300_rpn, roi_helpers, class_mapping, img_data, cfg)
						Visual_Cls_Predict(X_reImg0, boxes_dt, img_data, cfg)  #TODO:可视化Classifier网络回归过的预测结果。

				if iter_num >= epoch_length:
					loss_rpn_cls = np.mean(losses[:, 0])
					loss_rpn_regr = np.mean(losses[:, 1])  # 包含距离损失
					loss_class_cls = np.mean(losses[:, 2])
					loss_class_regr = np.mean(losses[:, 3]) # 包含距离损失
					class_acc = np.mean(losses[:, 4])  # 分类精度

					mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)

					rpn_accuracy_for_epoch = []
					Classifier_ignored_image_list = []
					if cfg.verbose:
						print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
						print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
						print('Loss RPN classifier: {}'.format(loss_rpn_cls))
						print('Loss RPN regression: {}'.format(loss_rpn_regr))  # 包括距离损失
						print('Loss Detector classifier: {}'.format(loss_class_cls))
						print('Loss Detector regression: {}'.format(loss_class_regr))  # 包括距离损失
						time_cost = time.time() - start_time
						print('Elapsed time: {}s'.format(time_cost))

					curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr  # 计算整个网络总损失
					iter_num = 0
					start_time = time.time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
					if curr_loss < best_loss:  # 当前损失小于最好损失，则
						if cfg.verbose:
							print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
						best_loss = curr_loss
						if curr_loss < 7 * cfg.Loss0:
							curr_loss_round = '%.3g' % curr_loss  # curr_loss0 = np.round(curr_loss, decimals=4)
							best_model_path = '{}T{:.0f}L{}.h5'.format(cfg.model_path0, 100*cfg.classifier_min_overlap, str(curr_loss_round)[2:])
							model_all.save_weights(best_model_path)
							model_all.save_weights(cfg.model_path)
					else:
						print('Total loss increased from {} to {} at {:d}/{:d} times, saving weights'.format(best_loss, curr_loss, int(epoch_num % times_increase_max), times_increase_max))
						model_all.save_weights(cfg.model_path)
						if curr_loss < cfg.Loss0*1.0:   #cfg.Loss0=0.05
							curr_loss_rounde = '%.3g' % curr_loss  # curr_loss0 = np.round(curr_loss, decimals=4)
							best_model_pathe = '{}T{:.0f}L{}e.h5'.format(cfg.model_path0, 100*cfg.classifier_min_overlap, str(curr_loss_rounde)[2:])
							model_all.save_weights(best_model_pathe)
						elif curr_loss < best_loss*1.05:
							curr_loss_rounde = '%.3g' % curr_loss  # curr_loss0 = np.round(curr_loss, decimals=4)
							best_model_pathe = '{}T{:.0f}L{}e.h5'.format(cfg.model_path0, 100*cfg.classifier_min_overlap, str(curr_loss_rounde)[2:])
							model_all.save_weights(best_model_pathe)

					train_loss.append([loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, curr_loss, class_acc, mean_overlapping_bboxes_exist_obj_imgs, time_cost, cfg.classifier_min_overlap])

					train_loss_array = np.around(np.array(train_loss), decimals=6)

					file_obj = open(cfg.training_loss_file, 'w')
					json.dump(train_loss_array.tolist(), file_obj)
					file_obj.close()

					plt.figure(1, figsize=(8, 4))
					# plt.xlabel('iterations (x%d)' % cfg.length_epoch, font)
					plt.xlabel('iterations', font)
					plt.ylim(0, 6)#plt.xlim(0, 100)
					plt.ylabel('training loss (%)', font)
					# plt.xlabel('Epoches')
					plt.plot(range(train_loss_array.shape[0]), np.array(100 * train_loss_array[:, 0]), 'k', label=' RPN cls loss')
					plt.plot(range(train_loss_array.shape[0]), np.array(100 * train_loss_array[:, 1]), 'g', label=' RPN regr loss')
					plt.plot(range(train_loss_array.shape[0]), np.array(100 * train_loss_array[:, 2]), 'b', label=' class cls loss')
					plt.plot(range(train_loss_array.shape[0]), np.array(100 * train_loss_array[:, 3]), 'm', label=' class regr loss')
					plt.plot(range(train_loss_array.shape[0]), np.array(100 * train_loss_array[:, 4]), 'r', label='Total loss')
					plt.legend(loc="upper right")
					plt.tick_params(labelsize=tick_size)  # 刻度字体大小13
					fig = plt.gcf()  # gcf - get current figure 得到当前图 # plt.show()  # plt.draw()
					fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.15)
					train_loss_fig_path = os.path.join(cfg.model_dir, 'train_loss.png')
					fig.savefig(train_loss_fig_path)
					plt.clf()

					plt.figure(2, figsize=(6, 4))
					plt.xlabel('iterations (x%d)' % cfg.length_epoch, font)
					plt.ylabel('mean number of BBs from RPN overlapping GT boxes', font)
					plt.plot(range(train_loss_array.shape[0]), np.array(train_loss_array[:, 6]), 'k', label='Mean number of BBs')
					plt.legend(loc="upper left")
					plt.tick_params(labelsize=tick_size)  # 刻度字体大小13
					fig = plt.gcf()  # gcf - get current figure 得到当前图 # plt.show()  # plt.draw()
					fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.15)
					Mean_number_of_BBs_fig_path = os.path.join(cfg.model_dir, 'Mean_number_of_BBs.png')
					fig.savefig(Mean_number_of_BBs_fig_path)
					plt.clf()
					break #完成1000张图片训练，退出当前while循环
				else:
					losses[iter_num, 0] = loss_rpn[1]  # RPN回归损失,loss_rpn=[0.0, 0.0, 0.0]
					losses[iter_num, 1] = loss_rpn[2]
					losses[iter_num, 2] = loss_class[1]
					losses[iter_num, 3] = loss_class[2]
					losses[iter_num, 4] = loss_class[3]

					iter_num += 1  # 迭代计数iter_num递增

			except Exception as e:
				#errormessage = '{}'.format(e)
				s = sys.exc_info()
				print("Exception: Error '%s' happened on line %d with image:'%s'" % (s[1], s[2].tb_lineno, img_data['filepath']))
				model_all.save_weights(cfg.model_pathe)# save model保存权重
				#model_all.save_weights(cfg.model_path)
				# pdb.set_trace()
				continue
	print('Training complete, exiting.')  # 训练完成并退出

if __name__ == '__main__':
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定第0个GPU:TitanRTX参与运算。
	gpu_cfg = tf.compat.v1.ConfigProto()
	gpu_cfg.gpu_options.allow_growth = False
	gpu_cfg.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用GPU90%的显存
	session = tf.compat.v1.Session(config=gpu_cfg)
	train_NIR()