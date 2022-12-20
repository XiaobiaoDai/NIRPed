import cv2, pdb
import numpy as np
import copy

#数据边缘填充，默认是否增强逻辑参数augment=True。img_data={ 'filepath': 'C:/WPy-3661/notebooks/keras_frcnn-master/images/000001.png', 'width': 1242, 'height': 375, 'bboxes': [{'class': 'Truck', 'x1': 599, 'x2': 629, 'y1': 156, 'y2': 189}, {'class': 'Car', 'x1': 387, 'x2': 423, 'y1': 181, 'y2': 203}, {'class': 'Cyclist', 'x1': 676, 'x2': 688, 'y1': 163, 'y2': 193}, {'class': 'DontCare', 'x1': 503, 'x2': 590, 'y1': 169, 'y2': 190}, {'class': 'DontCare', 'x1': 511, 'x2': 527, 'y1': 174, 'y2': 187}, {'class': 'DontCare', 'x1': 532, 'x2': 542, 'y1': 176, 'y2': 185}, {'class': 'DontCare', 'x1': 559, 'x2': 575, 'y1': 175, 'y2': 183}], 'imageset': 'trainval'}
def augment(img_data, config, augment=True):
# img_data=img_data_aug={'filepath': 'H:\\Daixb\\Experimetation\\Experiment_Road\\Near_Infrared20180320\\Calibration\\2018-03-20_19_01_55m.bmp',
#  'width': 1280, 'height': 1024, 'bboxes': [{'class': 'Ped', 'x1': 769, 'x2': 865, 'y1': 416, 'y2': 730, 'Dis': 55}], 'imageset': 'trainval'}
	assert 'filepath' in img_data #assert的语法格式：assert expression 它的等价语句为：
	#if not expression:   ——>raise AssertionError,确保img_data中的数据是完整无缺的
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)#深拷贝：拷贝img_data为img_data_aug,他们互相独立

	try:
		if len(config.img_channel_mean) == 3:  # 如果是RGB图像
			img = cv2.imread(img_data_aug['filepath'])  #img.shape=(720, 1280, 3)
		elif len(config.img_channel_mean) == 1:  # 如果是Gray图像
			img = cv2.imread(img_data_aug['filepath'], cv2.IMREAD_GRAYSCALE) #利用 cv2库按img_data_aug['filepath']路径读入灰度图像1024*1280，而对于彩色图像为1024*1280*3

		(rows0, cols0) = img.shape[:2]  #(rows0, cols0) = (720, 1280)
	except:
		pdb.set_trace()  	#I:\Datasets\VLP16_NIR2_2020CS\Data20200624201N850F12\Data20200624201642_833234N850F12.png
	img = cv2.resize(img, (2*config.im_cols, 2*config.im_rows), interpolation=cv2.INTER_CUBIC) #img.shape=(512, 1280, 3)
	#pdb.set_trace()
	for bbox in img_data_aug['bboxes']:  # 对img_data_aug的边界框坐标进行相应处理，对img_data_aug['bboxes']进行直接修改????
		x1 = float(bbox['x1'])/cols0
		x2 = float(bbox['x2'])/cols0
		y1 = float(bbox['y1'])/rows0
		y2 = float(bbox['y2'])/rows0
		bbox['x1'] = max(int(x1 * 2*config.im_cols), 0)
		bbox['x2'] = min(int(x2 * 2*config.im_cols), 2*config.im_cols-1)
		bbox['y1'] = max(int(y1 * 2*config.im_rows), 0)
		bbox['y2'] = min(int(y2 * 2*config.im_rows), 2*config.im_rows-1)
		bbox['area'] = (bbox['x2']-bbox['x1'])*(bbox['y2']-bbox['y1'])
	if augment:#是否增强逻辑参数augment=True
		(rows, cols) = img.shape[:2]#把图像的第1维(行)和第2维(列)分别赋给变量(rows, cols)=(512, 1280)

		if config.use_hsv and np.random.randint(0, 2) == 0:#如果使用饱和度和亮度变换增强 和从[0,1,2]中随机选取的数非0
			img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # RGB转换为HSV
			# image[:, :, 0] = np.power(image[:, :, 0], 1)
			coefficient_sv = np.random.randint(95, 100)
			img[:, :, 1] = np.power(img[:, :, 1], coefficient_sv/100)
			coefficient_sv = np.random.randint(95, 100)
			img[:, :, 2] = np.power(img[:, :, 2], coefficient_sv/100)
			img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)  # HSV转换为RGB

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:#如果使用水平翻转 和从[0,1]中随机选取的数为0
			img = cv2.flip(img, 1)#利用cv2.flip(img, 1)：1 	水平翻转；0	垂直翻转 -1 	水平垂直翻转
			for bbox in img_data_aug['bboxes']:#对img_data_aug的边界框坐标进行相应处理，对img_data_aug['bboxes']进行直接修改????
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:#进行垂直翻转及对边界框坐标进行相应处理
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:#对img_data_aug的边界框坐标进行相应处理
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		if config.use_rotate_angle and np.random.randint(0, 2) == 0:  # 如果使用小角度旋转增强 和从[0,1,2]中随机选取的数非0
			rotate_angle = np.random.randint(-5, 6)  #旋转角度-5~5度
			matrix0 = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), rotate_angle/10.0, 1)  # cols-1 and rows-1 are the coordinate limits.
			img = cv2.warpAffine(img, matrix0, (cols, rows))
			matrix0 = matrix0[:, :2]
			x_center =(cols - 1) / 2.0
			y_center =(rows - 1) / 2.0
			for bbox in img_data_aug['bboxes']:  # 对img_data_aug的边界框坐标进行相应处理
				x1y1 = [bbox['x1']-x_center, bbox['y1']-y_center]
				x1y1_rotate = np.dot(matrix0,x1y1)
				x1y1_rotate = [x1y1_rotate[0]+x_center, x1y1_rotate[1]+y_center]

				x1y2 = [bbox['x1']-x_center, bbox['y2']-y_center]
				x1y2_rotate = np.dot(matrix0, x1y2)
				x1y2_rotate = [x1y2_rotate[0]+x_center, x1y2_rotate[1]+y_center]

				x2y1 = [bbox['x2']-x_center, bbox['y1']-y_center]
				x2y1_rotate = np.dot(matrix0, x2y1)
				x2y1_rotate = [x2y1_rotate[0]+x_center, x2y1_rotate[1]+y_center]

				x2y2 = [bbox['x2']-x_center, bbox['y2']-y_center]
				x2y2_rotate = np.dot(matrix0, x2y2)
				x2y2_rotate = [x2y2_rotate[0]+x_center, x2y2_rotate[1]+y_center]

				bbox['x1'] = np.max([1, int(np.min([x1y1_rotate[0],x1y2_rotate[0],x2y1_rotate[0],x2y2_rotate[0]]))])
				bbox['y1'] = np.max([1, int(np.min([x1y1_rotate[1],x1y2_rotate[1],x2y1_rotate[1],x2y2_rotate[1]]))])
				bbox['x2'] = np.min([cols-1, int(np.max([x1y1_rotate[0],x1y2_rotate[0],x2y1_rotate[0],x2y2_rotate[0]]))])
				bbox['y2'] = np.min([rows-1, int(np.max([x1y1_rotate[1],x1y2_rotate[1],x2y1_rotate[1],x2y2_rotate[1]]))])

		if config.use_translation and np.random.randint(0, 2) == 0:  # 如果使用平移 和从[0,1]中随机选取的数为0
			dy_translation = 3*np.random.randint(-3, 4)
			dx_translation = 3*np.random.randint(-3, 4)
			matrix0 = np.float32([[1, 0, dx_translation], [0, 1, dy_translation]])  # 图像平移 下、上、右、左平移
			img = cv2.warpAffine(img, matrix0, (img.shape[1], img.shape[0]))
			for bbox in img_data_aug['bboxes']:#对img_data_aug的边界框坐标进行相应处理
				bbox['x1'] += dx_translation
				bbox['y1'] += dy_translation
				bbox['x2'] += dx_translation
				bbox['y2'] += dy_translation
				bbox['x1'] = np.max([1, bbox['x1']])
				bbox['y1'] = np.max([1, bbox['y1']])
				bbox['x2'] = np.min([cols-1, bbox['x2']])
				bbox['y2'] = np.min([rows-1, bbox['y2']])

		if config.rot_90:#顺时针旋转角度增强
			angle = np.random.choice([0,90,180,270],1)[0]#从列表[0,90,180,270]随机选取一个数作为增强的旋转角度
			if angle == 270:
				img = np.transpose(img, (1,0))#矩阵转置，灰度图像1024*1280
				img = cv2.flip(img, 0) #垂直翻转
			elif angle == 180:
				img = cv2.flip(img, -1) #水平垂直翻转
			elif angle == 90:
				img = np.transpose(img, (1,0))#矩阵转置，灰度图像1024*1280
				img = cv2.flip(img, 1) #水平翻转
			elif angle == 0:
				pass

			for bbox in img_data_aug['bboxes']:#对img_data_aug的边界框坐标进行相应处理
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2
				elif angle == 0:
					pass
	# for bbox in img_data_aug['bboxes']:  # 对img_data_aug的边界框坐标进行相应处理
	# 	if bbox['class'] == 'bg':
	# 		cv2.rectangle(img,(bbox['x1'],bbox['y1']),(bbox['x2'],bbox['y2']),(0, 255, 0), 2)
	# 	else:
	# 		cv2.rectangle(img,(bbox['x1'],bbox['y1']),(bbox['x2'],bbox['y2']),(0, 0, 255), 2)
	# cv2.imshow(img_data_aug['filepath'], img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	img_data_aug['width'] = img.shape[1]#增强后的图片宽度相应改变
	img_data_aug['height'] = img.shape[0]#增强后的图片高度相应改变
	return img_data_aug, img #无论怎么样，只有一种增强图片及RoI信息数据和1024*1280像素的图片数据
