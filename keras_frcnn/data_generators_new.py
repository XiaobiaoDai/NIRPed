from __future__ import absolute_import
import numpy as np
import cv2, os, pdb
import random, glob, re
import copy
from . import data_augment_new# from .import XXX 默认的就是在当前程序所在文件夹里__init__.py程序中导入data_augment，如果当前程序所在文件夹里没有__init__.py文件的话，就不能这样写，而应该写成from .A import XXX，A是指当前文件夹下你想导入的函数(或者其他的)的python程序名，如果你想导入的函数不在当前文件夹，那么就有可能用到 from .. import XXX(即上一个文件夹中的__init__.py)，或者from ..A import XXX(即上一个文件夹中的文件A)
import threading
import itertools #Python自带的用于高效循环的迭代函数集合

'''并集面积计算'''
def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union

'''交集面积计算'''
def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h

"""交并比计算"""
def iou(a: object, b: object) -> object:
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]: #剔除不正确的边界框
		return 0.0

	area_i = intersection(a, b) #交集面积计算
	area_u = union(a, b, area_i)  #并集面积计算

	return float(area_i) / float(area_u + 1e-6) #交并比计算
"""与忽略区域的交并比计算"""
def iou_ignore(a: object, b: object) -> object:
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]: #剔除不正确的边界框
		return 0.0

	area_i = intersection(a, b) #交集面积计算
	area_BBc = (b[2]-b[0])*(b[3]-b[1])  #候选框面积计算

	return float(area_i) / float(area_BBc + 1e-6) #交并比计算

'''图像resize宽和高计算。'''
def get_new_img_size(width, height, img_min_side=600):
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height #返回缩放后的图像的宽和高 像素

#用于训练样本选择的类SampleSelector(class_count),将字典classes_count传入类
class SampleSelector:
	def __init__(self, class_count): # class_count是如何传入的？？？？？？？？？？
		# ignore classes that have zero samples忽略无样本的分类
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		self.class_cycle = itertools.cycle(self.classes)#Python高效循环的迭代函数集合:创建一个self.class_cycle，从self.classes返回元素，并保存每个元素的副本。当self.classes耗尽时，从保存的副本中返回元素。无限重复。如：cycle('ABCD') --> A B C D A B C D ...
		self.curr_class = next(self.class_cycle)#返回迭代器的下一个项目

	def skip_sample_for_balanced_class(self, img_data):#为平衡类而跳过样本，img_data在调用程序的后续行传入
#字典img_data格式如：{ 'filepath': 'C:/WPy-3661/notebooks/keras_frcnn-master/images/000001.png', 'width': 1242, 'height': 375, 'bboxes': [{'class': 'Truck', 'x1': 599, 'x2': 629, 'y1': 156, 'y2': 189}, {'class': 'Car', 'x1': 387, 'x2': 423, 'y1': 181, 'y2': 203}, {'class': 'Cyclist', 'x1': 676, 'x2': 688, 'y1': 163, 'y2': 193}, {'class': 'DontCare', 'x1': 503, 'x2': 590, 'y1': 169, 'y2': 190}, {'class': 'DontCare', 'x1': 511, 'x2': 527, 'y1': 174, 'y2': 187}, {'class': 'DontCare', 'x1': 532, 'x2': 542, 'y1': 176, 'y2': 185}, {'class': 'DontCare', 'x1': 559, 'x2': 575, 'y1': 175, 'y2': 183}], 'imageset': 'trainval'} }
		class_in_img = False #类是否在图像中变量class_in_img默认为假

		for bbox in img_data['bboxes']:#用边界框变量bbox遍历列表img_data['bboxes']中的每个RoI框坐标数据

			cls_name = bbox['class']#将字典变量bbox的键'class'的值赋给变量cls_name

			if cls_name == self.curr_class:#如果cls_name==self.curr_class
				class_in_img = True#类是否在图像中变量class_in_img置为真
				self.curr_class = next(self.class_cycle)#返回迭代器的下一个类别
				break
#在一张图片所有目标数据中只要有一个目标的分类在字典classes_count，就将类是否在图像中变量class_in_img置为真
		if class_in_img:#在一张图片所有目标数据中只要有一个目标的分类在字典classes_count，返回假
			return False
		else:#在一张图片所有目标数据中目标的分类都不在字典classes_count，返回真
			return True


'''计算用于训练网络的RoIs，数目均衡在256个'''
def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
	downscale = float(C.rpn_stride) #在特征图上每移动1个像素点步长，相当于在resize的图上移动16个像素取一个box
	anchor_sizes = C.anchor_box_scales #在原始图上的锚短边的尺寸[80,120,160]  [12, 20, 35, 60, 100]
	anchor_ratios = C.anchor_box_ratios #在原始图上的锚边框短边和长边/C.anchor_box_scales（短边）的比例[[1, 1.9512195121951221]]
	num_anchors = len(anchor_sizes) * len(anchor_ratios)	#特征图上每个特征点锚的个数为5*1=9个

	# calculate the output map size based on the network architecture根据基础网络的网络架构计算输出特征映射图大小
	(output_width, output_height) = img_length_calc_function(resized_width, resized_height)
	n_anchsizes = len(anchor_sizes)#锚的短边尺寸个数n_anchsizes=3
	n_anchratios = len(anchor_ratios)#锚的比例个数n_anchratios=3

	# initialise empty output objectives初始化空输出目标
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))#TODO：初始化rpn的锚框与真实标记框box的交并比值y_rpn_overlap.shape=zeros(32,80,5)，包括内存在目标则为1，不存在为0。
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors))#TODO：是否为有效标记（也就是说是否与真实标记框box或背景重叠）y_is_box_valid=zeros(32,80,5)，有效的锚框为1，无效的锚框为0。
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))#TODO：初始化rpn的锚框到真实标记框box的回归参数y_rpn_regr=zeros(32,40,5*4=20) （tx,ty,tw,th）
	IoU_anchors = np.zeros((output_height, output_width, num_anchors))  # TODO：是否为有效标记（也就是说是否与真实标记框box或背景重叠）y_is_box_valid=zeros(32,80,5)，有效的锚框为1，无效的锚框为0。

	num_bboxes = len(img_data['bboxes'])#取出给定的一张图像中真实标记框box数量num_bboxes

	# get the GT box coordinates, and resize to account for image resizing
	#获取原始图像上的真实标记框boxes坐标映射到resized图像后的映射真实标记框boxes坐标，设置其初始值为：gta_real=zeros(num_bboxes,5)，并调整大小以适应图像大小调整
	gta_real = np.zeros((num_bboxes, 4)) #记录真实标记框boxe在sresized图像上的映射坐标，加入距离信息。gta_real= real Ground Truth Anchor (x1,x2,y1,y2，Dis)
	idx_real = []
	idx_Ign = []
	for bbox_num, bbox in enumerate(img_data['bboxes']):
		#字典img_data['bboxes']=[{'class': 'Ped', 'x1': 636, 'x2': 727, 'y1': 132, 'y2': 292, 'Dis': 20.0, 'Occ_Coe': 0, 'Dif': False, 'area': 14560, 'Age': 'Adult'}]
		# get the GT box coordinates, and resize to account for image resizing
		# x1*640/1280 对原始图上的真实标记坐标缩放到resized图上真实锚框坐标。注意：resized图上的真实坐标会出现小数
		gta_real[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
		gta_real[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
		gta_real[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))  # y1*512/1024
		gta_real[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
		distance = float(bbox['Dis'])
		if distance <= 0:
			distance = C.Dis_threshold
			bbox['Dis'] = distance
		if bbox['class'] in ['Pedestrian', 'pedestrian', 'Ped', 'ped']:
			if bbox['Dif'] or float(bbox['Occ_Coe']) > C.Occ_threshold or distance > C.Dis_threshold:
				idx_Ign.append(bbox_num)
			else:
				idx_real.append(bbox_num)
		else:  #['bicycledriver', 'motorbikedriver', 'ignore', 'Sed']
			idx_Ign.append(bbox_num)
		# TODO:将标记分成两类：用于训练和被忽视区域的索引，被忽视的标记包括骑车的人、被遮挡的、难以识别或距离大于C.Dis_threshold的。
	gta_Ign = gta_real[idx_Ign, ...]  #gta_Ign = array([], shape=(0, 4), dtype=float64)
	gta_real = gta_real[idx_real, ...] #gta_real=array([[318. , 363.5,  66. , 146. ]])
	num_bboxes = gta_real.shape[0]
	num_bboxes_Ign = gta_Ign.shape[0]

	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
	# 记录与num_bboxes个标记框IoU>C.rpn_max_overlap=0.7的锚的个数，初始值为0，num_anchors_for_bbox=zeros(num_bboxes，1)
	best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
	# 记录与num_bboxes个标记框最好重合的锚框坐标初始值best_anchor_for_bbox=-ones(num_bboxes,4)
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
	# 记录与num_bboxes个标记框最好重合的锚框的交并比初始值为0，best_iou_for_bbox=zeros(num_bboxes，1)
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
	# 记录与num_bboxes个标记框，在resized的图片上重合最好的锚框的坐标初始值best_x_for_bbox=zeros(num_bboxes, 4)？？？？？
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)
	# best_dx_for_bbox5 = np.zeros((num_bboxes, num5_regr)).astype(np.float32)
	# 记录与num_b	boxes个标记框，在resized的图片上重合最好的锚框的坐标修正参数初始值best_dx_for_bbox=zeros(num_bboxes, 4)
	for anchor_size_idx in range(n_anchsizes): #锚尺寸索引anchor_size_idx=0,1,2 分别对应[80,120,160]
		for anchor_ratio_idx in range(n_anchratios): #锚长宽比例索引anchor_ratio_idx=0,1,2 分别对应[[1, 1.2], [1,2.0], [1,2.8]]
			w_anchor = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]#计算5*1=9个锚框中第anchor_size_idx行第anchor_ratio_idx列的锚框的水平宽anchor_x
			h_anchor = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]#计算5*1=9个锚框中第anchor_size_idx行第anchor_ratio_idx列的锚框的垂直高anchor_y
			for ix in range(output_width):		#水平遍历特征图上的像素点ix=0，1，...，38，39
				# x-coordinates of the current anchor box  实际上是从resized图上的第16*0.5=8个像素点开始遍历的
				ix_resized = downscale * (ix + 0.5)  #ix=0 ix_resized=4
				x1_anchor = ix_resized - w_anchor / 2  #锚点对应的锚框，映射到resized的图片的左x坐标x1_anc=16*(ix + 0.5) - w_anchor / 2=-2
				x2_anchor = ix_resized + w_anchor / 2  #锚点对应的锚框，映射到resized的图片的右x坐标x1_anc=16*(ix + 0.5) + w_anchor / 2=10

				# ignore boxes that go across image boundaries忽略水平跨越出图像左、右边界的锚框
				if x1_anchor < 0 or x2_anchor > resized_width: #TODO: 跨界的锚框剔除掉
					continue

				for jy in range(output_height):	#垂直遍历特征图上的像素点iy=0，1，...，30，31
					jy_resized = downscale * (jy + 0.5)
					# y-coordinates of the current anchor box  实际上是从resized图上的第16*0.5=8个像素点开始遍历的
					y1_anchor = jy_resized - h_anchor / 2  # 锚点对应的锚框，映射到resized的图片的上坐标：y1_anchor=16*(jy + 0.5) - h_anchor / 2
					y2_anchor = jy_resized + h_anchor / 2  # 锚点对应的锚框，映射到resized的图片的下坐标：y1_anchor=16*(iy + 0.5) + w_anchor / 2
					# ignore boxes that go across image boundaries忽略垂直跨越出图像上、下边界的锚框
					if y1_anchor < 0 or y2_anchor > resized_height: #TODO: 跨界的锚框剔除掉
						continue

					#上面这一段计算了anchor的长宽，然后比较重要的就是把特征图的每一个点作为一个锚点，通过乘以downscale，映射到resized的图片上的实际映射尺寸，
					#再结合anchor的尺寸，忽略掉超出图片范围的。一个个大小、比例不一的矩形选框就跃然纸上了。对这些选框进行遍历，对每个选框进行下面的计算：
					bbox_type = 'neg' #bbox_type默认设置为负样本，如果是正样本后续进行更正
					# this is the best IOU for the (x,y) coord and the current anchor这是（x，y）坐标和当前锚点的最佳IOU
					# note that this is different from the best IOU for a GT bbox请注意，这与GT bbox的最佳IOU不同
					best_iou_for_loc = 0.0 #设置特征图上当前特征点（ix，jy）位置的局部最佳交并比初始值=0

					#接下来是根据特征图上当前特征点（ix，jy）对应的num_anchors=(5*1)个anchor中当前anchor的表现对其进行标注。
					for bbox_num in range(num_bboxes): #遍历当前图像中的num_bboxes个标记框，即：bbox_num=0，1，...，num_bboxes
						# get IOU of the current GT box and the current anchor box 获取当前真实标记框和当前锚框的IOU
						curr_iou = iou([gta_real[bbox_num, 0], gta_real[bbox_num, 2], gta_real[bbox_num, 1], gta_real[bbox_num, 3]], [x1_anchor, y1_anchor, x2_anchor, y2_anchor])

						# calculate the regression targets if they will be needed 如果需要，则计算回归目标
						if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap: #初始best_iou_for_bbox[bbox_num]=0  C.rpn_max_overlap=0.7
							cx_gt = 0.5*(gta_real[bbox_num, 0] + gta_real[bbox_num, 1]) #计算第bbox_num个真实标记框在resized的图片上对应的真实锚框的中心坐标
							cy_gt = 0.5*(gta_real[bbox_num, 2] + gta_real[bbox_num, 3])
							w_gt = gta_real[bbox_num, 1] - gta_real[bbox_num, 0]
							h_gt = gta_real[bbox_num, 3] - gta_real[bbox_num, 2]

							tx = (cx_gt - ix_resized) / w_anchor #真实标记框相对于锚框的中心水平坐标回归tx
							#当前锚框在resized的图片上的中心与第bbox_num个标记框在resized的图片上的中心的水平偏差/当前锚框在resized的图片上的宽度
							ty = (cy_gt - jy_resized) / h_anchor #真实标记框相对于锚框的中心垂直坐标回归
							# 当前锚框在resized的图片上的中心与第bbox_num个标记框在resized的图片上的中心的垂直偏差/当前锚框在resized的图片上的高度
							tw = np.log(w_gt / w_anchor) #log(标记框在resized的图片上的水平宽度/锚框在resized的图片上的宽度)
							th = np.log(h_gt / h_anchor) #log(标记框在resized的图片上的垂直高度/锚框在resized的图片上的高度)

							#  all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best所有GT框都应该映射到一个锚箱，所以我们跟踪哪个锚箱是最好的
						if curr_iou > best_iou_for_bbox[bbox_num]:
							#如果当前锚框（resized的图片上的）与第bbox_num个标记框（resized的图片上的）的交并比>best_iou_for_bbox[bbox_num]
							best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
							#记录与第bbox_num个标记框交并比最好的锚框在特征图上的位置和锚框比例和尺寸信息
							#best_anchor_for_bbox = [num_anchors=5*1=5, [rows=jy, cols=ix,, anchor_ratio_idx, anchor_size_idx] ]
							best_iou_for_bbox[bbox_num] = curr_iou
							#同时更新当前图片中第bbox_num个（resized的图片上的）标记框的最好交并比的值
							best_x_for_bbox[bbox_num, :] = [x1_anchor, x2_anchor, y1_anchor, y2_anchor]
							#记录与第bbox_num个（resized的图片上的）标记框交并比最好的锚框,在resized的图片上的坐标信息
							best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]
							#best_dx_for_bbox5[bbox_num,:] = [tx, ty, tw, th, td]
							#记录与第bbox_num个（resized的图片上的）标记框交并比最好的锚框坐标的回归参数[tx, ty, tw, th, td]
							IoU_anchors[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = curr_iou  # 作为背景样本处理，设置为可用样本（负样本）

						# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)如果IOU大于0.7，我们将锚设置为正（如果有另一个更好的方框，则无关紧要，它会被覆盖）
						if curr_iou > C.rpn_max_overlap: #C.rpn_max_overlap=0.7
							bbox_type = 'pos' #将当前锚框的类型标记为正样本，初始值默认为'neg'.
							num_anchors_for_bbox[bbox_num] += 1 #记录与第bbox_num个（resized的图片上的）标记框IoU>C.rpn_max_overlap=0.7的锚框的数量加1
							# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
							#在当前特征点（ix，jy），如果当前锚框与第bbox_num个（resized的图片上的）标记框的IoU>0.7>best_iou_for_loc，则更新best_iou_for_loc
							if curr_iou > best_iou_for_loc: #特征图上当前特征点（ix，jy）位置的局部最佳交并比初始值设置best_iou_for_loc = 0.0
								best_iou_for_loc = curr_iou #在当前特征点（ix，jy），更新与第bbox_num个（resized的图片上的）标记框的局部最佳交并比值best_iou_for_loc=curr_iou
								best_regr = (tx, ty, tw, th)#用元组记录当前特征的局部最佳交并比IoU对应的最佳坐标回归参数best_regr=(tx, ty, tw, th, td)
								#best_regr5 = (tx, ty, tw, th, td)#用元组记录当前特征的局部最佳交并比IoU对应的最佳坐标回归参数best_regr=(tx, ty, tw, th, td)

						# if the IOU is 0.3<IOU<0.7, it is ambiguous and no included in the objective
						#如果0.3<IOU<0.7，则它不明确是否与目标重叠的灰色区域
						if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
							# gray zone between neg and pos
							if bbox_type != 'pos': #如果尚无好锚就中立，已有好锚就丢弃
								bbox_type = 'neutral'
					#TODO:对与忽略重叠面积占比大于0.5的候选框，如果已被标记为背景，那么将其改成中性样本。
					best_iou_for_ignore_box = 0.0
					for bbox_num in range(num_bboxes_Ign): #遍历当前图像中的num_bboxes个标记框，即：bbox_num=0，1，...，num_bboxes
						# get IOU of the current GT box and the current anchor box 获取当前真实标记框和当前锚框的IOU
						curr_iou_ignore = iou_ignore([gta_Ign[bbox_num, 0], gta_Ign[bbox_num, 2], gta_Ign[bbox_num, 1], gta_Ign[bbox_num, 3]], [x1_anchor, y1_anchor, x2_anchor, y2_anchor])
						if curr_iou_ignore > best_iou_for_ignore_box:
							best_iou_for_ignore_box = curr_iou_ignore
					if bbox_type == 'neg' and best_iou_for_ignore_box >= 0.5:
						bbox_type = 'neutral'
					#TODO:对与忽略重叠面积占比大于0.5的候选框，如果已被标记为背景，那么将其改成中性样本。

					# turn on or off outputs depending on IOUs 根据IOU打开或关闭输出
					if bbox_type == 'neg':#如果在当前特征点（ix，jy），当前锚框的类型仍标记为负样本，则可作为有效的背景框记录下来。
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'neutral':#如果在当前特征点（ix，jy），当前锚框的类型标记为'neutral'，则可作为无效的背景框记录下来。
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'pos':#如果在当前特征点（ix，jy），当前锚框的类型标记为'pos'，则可作为有效的正样本框记录下来。
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx) #特征图上每个像素点，都有5*1=9个锚框，每个锚框都有5个坐标回归数据
						y_rpn_regr[jy, ix, start:start+4] = best_regr #记最佳锚框在特征图上的当前位置点以及回归参数和距离

	for idx in range(num_anchors_for_bbox.shape[0]): #用idx遍历与num_bboxes个标记框(记录了IoU>C.rpn_max_overlap=0.7的锚的个数)
		if num_anchors_for_bbox[idx] == 0: #如果第idx(即：idx/bbox_num)个标记框没有匹配到IoU>C.rpn_max_overlap=0.7好锚框
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1: #如果第idx个标记框也没有匹配到0<IoU<C.rpn_max_overlap=0.7次好锚框(完全不相交)，跳过后续程序重新循环
				continue
			# 如果有0<IoU<C.rpn_max_overlap=0.7次好锚就将次好锚是否可用标记改成可用，重叠标记也改成重叠，重新记录y_rpn_regr
			y_is_box_valid[best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *best_anchor_for_bbox[idx,3]] = 1 #重新将次好锚框标记为有效可用的
			y_rpn_overlap[best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *best_anchor_for_bbox[idx,3]] = 1  #重新将次好锚框标记成正锚框
			start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
			y_rpn_regr[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start + 4] = best_dx_for_bbox[idx, :] #负样本(背景)是没有回归参数的
			# best_anchor_for_bbox = [bbox_num=5*1=5, [rows=jy, cols=ix,, anchor_ratio_idx, anchor_size_idx] ]


	#原始：y_rpn_overlap=[rows=jy, cols=ix, num_anchors=5*1=5]
	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))   #变换后：y_rpn_overlap=[num_anchors=5*1=5,rows=jy, cols=ix]
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)  #变换后：y_rpn_overlap=[1,num_anchors=5*1=5,rows=jy, cols=ix]
	#原始：y_is_box_valid=[rows=jy, cols=ix,num_anchors=5*1=5]
	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))  #变换后：y_is_box_valid=[num_anchors=5*1=5, rows=jy, cols=ix]
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0) #变换后：y_is_box_valid=[1,num_anchors=5*1=5, rows=jy, cols=ix]
	IoU_anchors = np.transpose(IoU_anchors, (2, 0, 1))  #变换后：y_is_box_valid=[num_anchors=5*1=5, rows=jy, cols=ix]

	#原始：y_rpn_regr=[rows=jy, cols=ix,, (5*1)*(4+1)=45]
	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1)) #变换后：  y_rpn_regr=[(5*1)*4=5*(tx, ty, tw, th)=20,rows=jy, cols=ix]
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)  #变换后：y_rpn_regr=[1,(5*1)*4=5*(tx, ty, tw, th)=20,rows=jy, cols=ix]


	# TODO：取出采样前的样本的IoU
	sample_locs = np.where(y_is_box_valid[0, :, :, :] == 1)  # 找出所有负锚框(锚框里面是负样本)在特征图上的位置及锚框索引

	IoUs_RPN_original = []
	for index in range(len(sample_locs[0])):
		IoUs_RPN_original.append(IoU_anchors[sample_locs[0][index], sample_locs[1][index], sample_locs[2][index]])
	# TODO：取出采样前的样本的IoU

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))#找出所有正锚框(锚框里面是正样本)在特征图上的位置及锚框索引
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))#找出所有负锚框(锚框里面是负样本)在特征图上的位置及锚框索引

	num_pos = len(pos_locs[0])#计算正锚框(锚框里面是正样本)个数num_pos=8
	num_neg = len(neg_locs[0])#计算正锚框(锚框里面是正样本)个数num_neg=7535

	#print('num_pos={},num_neg={}'.format(num_pos, num_neg)) #查看可用于训练的正负样本数量
	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	#问题是RPN有更多的负锚框(锚框里面是负样本)而不是正锚框(锚框里面是正样本)，所以我们关闭了一些负锚框(锚框里面是负样本)
	# regions. We also limit it to 256 regions.我们还将正负总锚框数限制在256个。
	#接下来通过numpy大法进行了一系列操作，对pos和neg的anchor进行了定位。
	num_regions = C.batch_size_rpn #总RoI数 ##########################################################Pause20181108

	# TODO：按样本交并比大小采样***按样本交并比大小采样***按样本交并比大小采样***按样本交并比大小采样***
	IoUs_positive = []
	for index in range(len(pos_locs[0])):  #TODO：计算正样本质量Fqs_positive。
		IoU_anchor = IoU_anchors[pos_locs[0][index], pos_locs[1][index], pos_locs[2][index]]
		IoUs_positive.append(IoU_anchor)
	index_IoUs_pos_sorted = np.argsort(np.array(IoUs_positive)) #TODO：按找交并比大小排序正样本索引。

	IoUs_negative = []
	for index in range(len(neg_locs[0])): #TODO：计算所有负样本质量Fqs_negative。
		IoU_anchor = IoU_anchors[neg_locs[0][index], neg_locs[1][index], neg_locs[2][index]]
		IoUs_negative.append(IoU_anchor)

	index_IoUs_bg = np.where(np.array(IoUs_negative) == 0)[0]
	index_IoUs_neg_sorted = np.argsort(np.array(IoUs_negative))
	# TODO：按样本交并比大小采样***按样本交并比大小采样***按样本交并比大小采样***按样本交并比大小采样***

	if num_pos > num_regions:
		# val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2) #随机取出超128个的正锚框(锚框里面是正样本)的序号。
		val_locs = index_IoUs_pos_sorted[:num_pos-num_regions]
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0 #y_is_box_valid=[1,num_anchors=5*1=5, rows=jy, cols=ix]
		# 将原本正锚框(锚框里面是正样本)重新标记成无效的，可能某个正样本标记框的正锚框(锚框里面是正样本)都干掉了？？
		num_pos = num_regions #重新计算正锚框(锚框里面是正样本)数量
	# 如果有大于128个正锚框(锚框里面是正样本)，处理了过多的正锚框(锚框里面是正样本)，接下来就着手处理负锚框(锚框里面是负样本)
	if num_pos == num_regions:# 如果正负锚框总数等于num_regions=256，所有的负锚框都处理掉。
		y_is_box_valid[0, neg_locs[0][:], neg_locs[1][:], neg_locs[2][:]] = 0  # y_is_box_valid=[1,num_anchors=5*1=5, rows=jy, cols=ix]
	else:  # 如果正负锚框总数还大于num_regions=256，接下来将多余的负锚框处理掉。
		if num_neg -len(index_IoUs_bg) + num_pos > num_regions:
			val_locs = index_IoUs_neg_sorted[:num_pos-num_regions]
			# 从负锚框(锚框里面是负样本)中，随机取出超正锚框(锚框里面是正样本)数的负锚框序号，将其剔除出去，使得正负锚框数相等，最大数量为128个。
			y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0  # y_is_box_valid=[1,num_anchors=5*1=5, rows=jy, cols=ix]
		else:
			val_locs = random.sample(index_IoUs_bg.tolist(), num_neg + num_pos - num_regions)
			y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0  # y_is_box_valid=[1,num_anchors=5*1=5, rows=jy, cols=ix]
	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)

	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

	#TODO：取出被采样的256个训练样本的IoU
	sample_locs = np.where(y_is_box_valid[0, :, :, :] == 1)  # 找出所有负锚框(锚框里面是负样本)在特征图上的位置及锚框索引
	IoUs_RPN = []
	for index in range(len(sample_locs[0])):
		IoUs_RPN.append(IoU_anchors[sample_locs[0][index], sample_locs[1][index], sample_locs[2][index]])
	#TODO：取出被采样的256个训练样本的IoU
	return np.copy(y_rpn_cls), np.copy(y_rpn_regr), IoUs_RPN, IoUs_RPN_original

class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by serializing call to the `next` method of given iterator/generator.
	通过序列化对给定迭代器/生成器的`next`方法的调用，获取一个迭代器/生成器并使其成为线程安全的。
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)		

def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g

'''产生用于训练的真实锚点数据的函数：产生用于训练的真实锚框数据的generator（生成器）函数：'''
def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):
	sample_selector = SampleSelector(class_count) #调用 用于训练样本选择的类SampleSelector(class_count),将字典classes_count传入类SampleSelector，
	while True:
		if mode == 'train':
			random.shuffle(all_img_data)#对表all_img_data中的字典元素重新随机排序

		for img_data in all_img_data: #遍历表all_img_data的每个字典元素(即记录每张图片目标信息的字典)，每次取出一张图片img_data
			try:   #如果 类别数量平衡 C.balanced_classes and 在一张图片所有目标数据中目标的分类都不在字典classes_count(返回真)
				if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):#C.balanced_classes = False #类别数量是否平衡：默认否；sample_selector.skip_sample_for_balanced_class
					continue #告诉Python跳过当前循环的剩余语句，然后继续进行下一轮循环。# read in image, and optionally add augmentation读入图像，并可选择是否添加扩充
				if mode == 'train':#如果给定的是训练模式，对训练样本进行扩充。返回增强图片及RoI信息数据img_data_aug和图片数据x_img0
					img_data_aug, x_img = data_augment_new.augment(img_data, C, augment=True) #x_img.shape=(1024,1280)
				else: #如果给定的不是训练模式，对训练样本不进行扩充
					img_data_aug, x_img = data_augment_new.augment(img_data, C, augment=False)

				(width, height) = (img_data_aug['width'], img_data_aug['height']) #(width, height)=(1280, 512)
				(rows, cols) = x_img.shape[:2] #原始程序：(rows, cols, _) = x_img.shape，RGB图变成灰度图只剩下2维 (rows, cols)=(512, 1280)
				assert cols == width  #进一步确认cols == width=1280
				assert rows == height #进一步确认rows == height= 512
				(resized_width, resized_height) = (C.im_cols, C.im_rows) #(resized_width, resized_height)=(C.im_cols=640, C.im_rows=256)
				x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)#对图像矩阵操作 #所用的插值方法interpolation=cv2.INTER_CUBIC(4x4像素邻域的双三次插值)resize后的图像矩阵x_img.shape=(512,640)

				x_img0 = np.copy(x_img)
				if len(C.img_channel_mean) == 3:  # 如果是RGB图像
					x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB图像色彩通道BGR改成RGB
					x_img = x_img.astype(np.float32)  # 将numpy图像512*640矩阵变成np.float32型矩阵，为后续计算做准备
					x_img[:, :, 0] -= C.img_channel_mean[0]
					x_img[:, :, 1] -= C.img_channel_mean[1]
					x_img[:, :, 2] -= C.img_channel_mean[2]
					x_img = np.transpose(x_img, (2, 0, 1))
				elif len(C.img_channel_mean) == 1:  # 如果是Gray图像
					x_img = x_img.astype(np.float32)  # 将numpy图像512*640矩阵变成np.float32型矩阵，为后续计算做准备
					x_img -= C.img_channel_mean[0]
					x_img /= C.img_scaling_factor
					x_img = np.expand_dims(x_img, axis=0)  # 更改后的程序其实expand_dims(a, axis)就是在axis的那一个轴上把数据维加上去，这个数据在axis这个轴的0位置。
				x_img = np.expand_dims(x_img, axis=0)  # 将图片的维度变成了x_img.shape=(Samples=1,Channels=1,512,640)
				if backend == 'channels_last':
					x_img = np.transpose(x_img, (0, 2, 3, 1))  # 变换后：x_img.shape=(Samples=1,512,640,Channels=1)

				try: #尝试获取region proposal network 的分类和坐标回归
					y_rpn_cls, y_rpn_regr, IoUs_RPN, IoUs_RPN_original = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)

				except Exception as e: #Exception可以将所有的异常包括在内；将异常赋予变量e
					print('Exception: {}'.format(e)) #打印出异常变量值
					continue # Zero-center by mean pixel, and preprocess image零中心均值像素和预处理图像

				y_rpn_regr[:, y_rpn_regr.shape[1]//2:y_rpn_regr.shape[1]-1, :, :] *= C.std_scaling  #改后可能存在问题，要注意了！！！！！！！！！！！！！！！！
				if backend == 'channels_last':
					y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1)) #变换后：y_rpn_cls.shape = (samples=1,rows=jy=32,cols=ix=80,(5*1)+(5*1)=10)
					y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1)) #变换后：y_rpn_regr.shape = [samples=1,rows=jy=32, cols=ix=80,5*4+5*4=90]

				yield np.copy(x_img), np.copy(y_rpn_cls), np.copy(y_rpn_regr), img_data_aug, x_img0, IoUs_RPN, IoUs_RPN_original

			except Exception as e:
				print(e)
				continue