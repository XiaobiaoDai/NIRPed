import os, json, glob, shutil, argparse, pdb, sys,  cv2, datetime, pickle,  operator,math, copy
# from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
#np.set_printoptions(precision=6, threshold=np.inf, edgeitems=10, linewidth=260, suppress=True)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from PIL import Image, ImageDraw, ImageFont
from keras_frcnn.coco import COCO
from keras_frcnn import config
np.set_printoptions(precision=6, threshold=np.inf, edgeitems=10, linewidth=260, suppress=True)
cfg = config.Config()
time_now_min = datetime.datetime.now().strftime('%Y%m%d%H%M')
time_now_day = datetime.datetime.now().strftime('%Y%m%d')
# MINOVERLAP = 0.40  # TODO:default value (defined in the PASCAL VOC2012 challenge)最小交并比MINOVERLAP = 0.5
MINOVERLAP = 0.50  # TODO:default value (defined in the PASCAL VOC2012 challenge)最小交并比MINOVERLAP = 0.5
Hf = 3985  #TODO:4125

cfg.evaluate_subset = 'Reasonable'
# cfg.evaluate_subset = 'All'

# evalute_new = 0 #根据得分阈值score_threshold，再次计算MR-FPPI曲线,评估测距结果
evalute_new = 1  #显示MR-2时的检测结果

idx_Dis_max = 8

idx_step = 100
max_boxes = 300

IoU_threshold_rpn = 0.70
score_threshold_rpn = 0.50
IoU_threshold_cls = 0.50
score_threshold_cls = 0.001
imageset = 'val'
# imageset = 'test'

Detection_results_dir = "./results/dt_results_%s_B%d_%s" % (imageset, max_boxes, str(score_threshold_cls)[2:])

class NpEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(NpEncoder, self).default(obj)
def error(msg):  # throw error and exit  抛出错误并退出
	print(msg)
	sys.exit(0)  # 干净利落地退出系统

""" ground-truth 真实标记框    Load each of the ground-truth files into a temporary '.json' file. 将每个真实标记框文件加载到一个临时的“.json”文件中。
	 Create a list of all the class names present in the ground-truth (gt_classes).创建一个包含真实标记框（gt_类）中所有类名的列表。"""
def Temp_gt_Generation(TEMP_FILES_Path):
	cocoGt = COCO(cfg.annos_file)
	imgIds = sorted(cocoGt.getImgIds())  # imgIds=[100013, 100024, 100063, 100065, 100074, 100084, 100143, 100154, 100159, 100164,...]
	gt_counter_per_class = {}
	ele = [0]*12
	for class_name in ['Ped', 'Peo', 'Bic', 'Mot','Ign','bg']:
		gt_counter_per_class[class_name] = {'All': copy.deepcopy(ele), 'short': copy.deepcopy(ele), 'normal_height': copy.deepcopy(ele), 'tall': copy.deepcopy(ele),
											'Sun': copy.deepcopy(ele), 'Rain': copy.deepcopy(ele), 'Spate': copy.deepcopy(ele)}

	num_test_imgs = {'All': 0, 'Sun': 0, 'Rain': 0, 'Spate': 0}

	for i in range(len(imgIds)):
		anno_ids = cocoGt.getAnnIds(imgIds=imgIds[i])  # anno_id=1037542#anno_ids= [7000000, 7000001]
		annos = cocoGt.loadAnns(ids=anno_ids)
		# annos=[{'occluded': False, 'Dif': False, 'bbox': [349, 227, 20, 41], 'id': 7000000, 'category_id': 1, 'image_id': 7000000, 'pose_id': 1, 'tracking_id': 7000000, 'ignore': 0, 'area': 820, 'truncated': False},
		# {'occluded': False, 'Dif': False, 'bbox': [645, 239, 21, 40], 'id': 7000001, 'category_id': 1, 'image_id': 7000000, 'pose_id': 1, 'tracking_id': 7000001, 'ignore': 0, 'area': 840, 'truncated': False}]
		# Show the annotation in its image
		image = cocoGt.loadImgs(ids=imgIds[i])[0]
		# image={'height': 640, 'width': 1024, 'daytime': 'night', 'file_name': '58c58285bc26013700140940.png', 'id': 1096678,'recordings_id': 15.0, 'timestamp': 1598649939}
		img_name = image['file_name']
		img_id = img_name.split('.', 1)[0]

		if img_id[:12] in ['Data20181219', 'Data20181220', 'Data20190113']:
			image['weathers_id'] = 0

		num_test_imgs['All'] += 1

		if image['weathers_id'] == 0:
			Wea = 'Sun'
		elif image['weathers_id'] == 2:
			Wea = 'Spate'
		else:
			Wea = 'Rain'
		num_test_imgs[Wea] += 1

		# img_path = os.path.join('E:\\Datasets\\NIRPed2021\\NIRPed\\images\\{}\\{}'.format(imageset, img_name))
		img_path = os.path.join('.\\data\\miniNIRPed\\images\\{}\\{}'.format(imageset, img_name))

		if img_path == None:
			print('Notion:{} is not exist.'.format(img_path))
			continue

		groundTruth_boxes = []
		try:
			for anno in annos:
				cat = cocoGt.loadCats(ids=anno['category_id'])[0]  # cat={'name': 'pedestrian', 'id': 1}
				class_name = cat['name']
				if class_name in ['Pedestrian', 'pedestrian', 'Ped', 'ped']:
					class_name = 'Ped'
				bbox = anno['bbox']
				height_gt = bbox[3]
				distance = float(anno['Dis'])
				if distance <= 0 or distance == 100:
					distance = round(Hf /height_gt, 1)  # TODO: 将像素高度直接估算为距离。
				if distance > 110:
					distance = 110
					# distance = cfg.Dis_threshold
				box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
				# if flag_mr_height == 1:
				dis_gt = distance
				dis_gt_index = int(dis_gt//10)
				dis_from_height_gt = Hf / height_gt
				error_rate_dis_from_height_gt = 100 * (dis_from_height_gt - dis_gt) / dis_gt
				if error_rate_dis_from_height_gt >= 10:
					height = 'tall'
				elif error_rate_dis_from_height_gt < -20:
					height = 'short'
				else:
					height = 'normal_height'

				if class_name in ['Ped', 'Peo', 'Bic', 'Mot']:
					vis_box = anno['vis_box']
					aera_box = float(bbox[2]) * float(bbox[3])
					if 100 < aera_box < 1280 * 720 * 0.9:
						vis_aera = float(vis_box[2]) * float(vis_box[3])
						Occlusion_coefficient = 1 - vis_aera / aera_box
						if 0 <= Occlusion_coefficient < 1:
							anno['Occ_Coe'] = round(Occlusion_coefficient, 2)
						else:
							# print('Wrong visable box yeild to Occlusion_coefficient = {}, and change it to Occlusion_coefficient=0'.format(Occlusion_coefficient))
							anno['Occ_Coe'] = 0
						if cfg.evaluate_subset == 'Reasonable':
							if anno['Occ_Coe'] < cfg.Occ_threshold and not anno['Dif']:
								gt_counter_per_class[class_name]['All'][dis_gt_index] += 1
								gt_counter_per_class[class_name][height][dis_gt_index] += 1
								gt_counter_per_class[class_name][Wea][dis_gt_index] += 1

						elif cfg.evaluate_subset == 'All':
							gt_counter_per_class[class_name]['All'][dis_gt_index] += 1
							gt_counter_per_class[class_name][height][dis_gt_index] += 1
							gt_counter_per_class[class_name][Wea][dis_gt_index] += 1

					groundTruth_boxes.append({'class_name': class_name, 'bbox': box, 'Dis': dis_gt, 'used': False,
											  'Dif': anno['Dif'], 'Occ_Coe': anno['Occ_Coe'], 'Age': anno['Age'],
											  'area': anno['area'], 'Height': height, 'Wea': Wea})

				else:
					gt_counter_per_class[class_name]['All'][dis_gt_index] += 1
					groundTruth_boxes.append({'class_name': class_name, 'bbox': box, 'Dis': dis_gt, 'used': False,
											 'Dif': anno['Dif'], 'Occ_Coe': anno['Occ_Coe'], 'Age': anno['Age'],
											 'area': anno['area'], 'Height': height, 'Wea': Wea})
					gt_counter_per_class[class_name][height][dis_gt_index] += 1
					gt_counter_per_class[class_name][Wea][dis_gt_index] += 1

			if len(groundTruth_boxes) == 0:
				groundTruth_boxes.append({'class_name': 'Bg_Img', 'Height': 'normal_height', 'Wea': Wea})

		except Exception as e:
			s = sys.exc_info()
			print("Exception: Error '%s' happened on line %d with image:'%s'" % (s[1], s[2].tb_lineno, img_id))
			pdb.set_trace()

		gt_TEMP_FILES_Path = os.path.join(TEMP_FILES_Path, img_id + '_gt.json')
		if not os.path.exists(gt_TEMP_FILES_Path):
			outfile = open(gt_TEMP_FILES_Path, 'w')
			json.dump(groundTruth_boxes, outfile)
			outfile.close()

	# get a list with the ground-truth files 获取包含真实标记框文件的列表
	gt_files_list = glob.glob(TEMP_FILES_Path + '/*.json')
	# gt_files_list = ['E:\\Daixb\\Evaluation_mAP_MissRate-FPPI\\MissRate-FPPI_results\\NIRPed_WeightsResnet50NIR1RGB64_1024_128_2o5L0.18\\.temp_files\\Data20181219200348_020000_gt.json', ……]
	gt_files_list.sort()

	num_test_imgs['short'] = num_test_imgs['All']
	num_test_imgs['normal_height'] = num_test_imgs['All']
	num_test_imgs['tall'] = num_test_imgs['All']
	for attr in num_test_imgs:
		print('NIRPed：num_test_imgs_%s=%d' % (attr, num_test_imgs[attr]))
	# for class_name in ['Ped', 'Peo', 'Bic', 'Mot', 'Ign', 'bg']:
	for class_name in ['Ped']:
		for attr in gt_counter_per_class[class_name]:
			print('class_name:%s,attribute:%s'% (class_name, attr))
			print(gt_counter_per_class[class_name][attr])

	return gt_files_list, gt_counter_per_class, num_test_imgs

# TODO: *** 有序显示标记。*** 有序显示标记。*** 有序显示标记。*** 有序显示标记。*** 有序显示标记。*** 有序显示标记。*** 有序显示标记。
colors = {'Slice': ['white', 'white'], 'GT': ['red', 'black'], 'DT': ['white', 'white'], 'TP': ['lime', 'lime'], 'FP': ['magenta', 'magenta'], 'miss_Ped': ['red', 'red'], 'AER_bad': ['red', 'red'],
		  'Peo': ['orchid', 'black'], 'Ped': ['orangered', 'black'], 'Bic': ['hotpink', 'black'], 'Mot': ['fuchsia', 'black'], 'Ign': ['limegreen', 'black'], 'bg': ['seagreen', 'black']}

# color_white = 'white'
# color_white = 'lightgrey'
color_white = 'dimgray'
color_Obj = colors['Ped']
color_Ped = colors['Ped']
color_bg = colors['bg']
# color_GT = colors['GT']
# color_TP = colors['TP']
# color_FP = colors['FP']
# color_AER_bad = colors['AER_bad']
# color_FN = colors['miss_Ped']
color_GT = colors['Slice']
color_TP = colors['Slice']
color_FP = colors['Slice']
color_AER_bad = colors['Slice']
color_FN = colors['Slice']

size_font_slice = 28
width_line_slice = 1
font_slice = ImageFont.truetype('arial.ttf', size=30)
# font_image = ImageFont.truetype('arial.ttf', size=32)
font_image = ImageFont.truetype('arial.ttf', size=50)
# width_line_image = 1
width_line_image = 2
size_font_highlight = 2
edge_kept = 2
edge_top = 0
def draw_tags_orderly(draw,label,Doted_text,color_cls,left,right,top,bottom,h_box,edge_kept=5):
	label_size = draw.textsize(label, font_image)
	# text_bbox = draw.textbbox(label, font_image)
	# text_length = draw.textbbox(label, font_image)
	if bottom < cfg.im_rows_show - 10 - edge_kept - label_size[1]:
		bottom_text_boundary = bottom + 20
	else:
		bottom_text_boundary = bottom - 20

	if h_box < cfg.im_rows_show / 4 and top - edge_kept - label_size[1] > 0:
		text_origin = np.array([int(min(max(0.5 * (left + right) - 0.5 * label_size[0], 0), cfg.im_cols_show - label_size[0])), int(edge_kept)])
	else:
		text_origin = np.array([int(min(max(0.5 * (left + right) - 0.5 * label_size[0], 0), cfg.im_cols_show - label_size[0])), int(cfg.im_rows_show - edge_kept - label_size[1])])

	x = text_origin[0]
	y = text_origin[1]
	y_modified = y
	dy = label_size[1] // 5
	y_range = int((cfg.im_rows_show - 2 * edge_kept) / dy)

	for tp_all in range(0, y_range):
		flagT = 1
		if tp_all > 0:
			if y_modified < top - label_size[1] - 10:
				y_modified = y_modified + dy
			elif y_modified >= bottom_text_boundary:
				y_modified = y_modified - dy
			if top - label_size[1] - 10 <= y_modified < bottom_text_boundary:
				if np.random.randint(0, 2) == 0:
					y_modified = edge_kept
				else:
					y_modified = cfg.im_rows_show - edge_kept - label_size[1]

		if Doted_text != []:
			for Dot_xy in Doted_text:
				dis_x = np.abs(Dot_xy[0] - x)
				dis_y = np.abs(Dot_xy[1] - y_modified)
				if x < Dot_xy[0] and y_modified < Dot_xy[1]:
					if dis_x < label_size[0] + 2 and dis_y < label_size[1] + 2:
						flagT = 0
						break
				elif x < Dot_xy[0] and y_modified >= Dot_xy[1]:
					if dis_x < label_size[0] + 2 and dis_y < Dot_xy[3] + 2:
						flagT = 0
						break
				elif x >= Dot_xy[0] and y_modified < Dot_xy[1]:
					if dis_x < Dot_xy[2] + 2 and dis_y < label_size[1] + 2:
						flagT = 0
						break
				elif x >= Dot_xy[0] and y_modified >= Dot_xy[1]:
					if dis_x < Dot_xy[2] + 2 and dis_y < Dot_xy[3] + 2:
						flagT = 0
						break
		if flagT == 1:
			try:
				text_origin = np.array([x, y_modified])
				text_origin_rectangle = np.array([x-1, y_modified])
				draw.rectangle([tuple(text_origin_rectangle), tuple(text_origin_rectangle + label_size)], outline=color_cls[0], fill=color_white, width=width_line_image) #TODO:fill=(255, 255, 255) 将改成：fill=color_white
				'''如果你讀了灰度圖像並嘗試繪製顏色矩形就可以了，你會得到錯誤信息：TypeError: function takes exactly 1 argument (3 given)
				你需要給一個顏色像outline=(255)，沒有RGB色彩如outline=(255, 0, 0)。否則，你會得到錯誤，因爲你給了3個顏色參數，而不是一個。
				如果你想在灰度圖像上繪製顏色，你可以先將圖像轉換爲RGB：img = img.convert('RGB')'''
				draw.text(text_origin, label, fill=color_cls[1], font=font_image)
				Doted_text.append([text_origin[0], text_origin[1], label_size[0], label_size[1]])
			except:
				print('error in draw.rectangle()')
				pdb.set_trace()
			break

	if flagT == 0:
		text_origin = np.array([x, bottom_text_boundary])
		text_origin_rectangle = np.array([x-1, bottom_text_boundary])
		draw.rectangle([tuple(text_origin_rectangle), tuple(text_origin_rectangle + label_size)], outline=color_cls[0], fill=(255, 255, 255), width=width_line_image)
		draw.text(text_origin, label, fill=color_cls[1], font=font_image)
		Doted_text.append([text_origin[0], text_origin[1], label_size[0], label_size[1]])

	if top > text_origin[1] + label_size[1]:
		draw.line((int(0.5 * (left + right)), top, int(0.5 * (left + right)), text_origin[1] + label_size[1]), fill=color_cls[0], width=width_line_image)
	if bottom < text_origin[1]:
		draw.line((int(0.5 * (left + right)), bottom, int(0.5 * (left + right)), text_origin[1]), fill=color_cls[0], width=width_line_image)
	# pdb.set_trace()
	# draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 255, 255, 20))
	# draw.text(text_origin, label, fill=color_Ped[1], font=font_image)
	# draw.rectangle([left, top, right, bottom], outline=color_cls[0], width=width_line_image)  # 淡红色单框：缩放到resized图上标记框
	return draw, Doted_text
# TODO: *** 有序显示标记。*** 有序显示标记。*** 有序显示标记。*** 有序显示标记。*** 有序显示标记。*** 有序显示标记。*** 有序显示标记。

#TODO:*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.
#TODO:*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.
def MR_FPPI_Plot(MR_FPPI_dis_dic, detectResults_Path, max_or_seg):# TODO: ***图像显示MissRate_FPPI曲线。
	font_size = 24
	font = {'family': 'Times New Roman', 'weight': 'normal', 'size': font_size}
	colors = ['green', 'lime', 'blue', 'purple', 'hotpink', 'orangered', 'fuchsia', 'red', 'black']
	FPPI_0_2 = np.logspace(-2, 0, 9)
	FPPI_0_4 = np.logspace(-4, 0, 9)
	for attr in MR_FPPI_dis_dic:
		fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # fig = plt.figure(facecolor='w')
		for dis_index in range(0, idx_Dis_max):
			dis_idx = idx_Dis_max-1 - dis_index
			MR_FPPI_array = np.array(MR_FPPI_dis_dic[attr][dis_idx])

			MissRate_list = []
			for i, x in enumerate(FPPI_0_4):
				idx = np.argmin(np.abs(MR_FPPI_array[:, 0]-x))
				MissRate_list.append(MR_FPPI_array[idx, 1])
			# print('MissRate_list_4 of MR_FPPI[%s][%s]='% (attr, dis_index))
			# print(MissRate_list)
			LAMR_4 = pow(np.prod(MissRate_list), 1 / 9)

			MissRate_list = []
			for i, x in enumerate(FPPI_0_2):
				idx = np.argmin(np.abs(MR_FPPI_array[:, 0]-x))
				MissRate_list.append(MR_FPPI_array[idx, 1])

			# print('MissRate_list_2 of MR_FPPI[%s][%s]=' % (attr, dis_index))
			# print(MissRate_list)
			LAMR_2 = pow(np.prod(MissRate_list), 1 / 9)

			if max_or_seg == 'max':
				dis_txt = 'd<%dm' % (dis_idx * 10 + 10)
			elif max_or_seg == 'seg':
				dis_txt = '%d≤d<%dm' % (dis_idx * 10, dis_idx * 10 + 10)
			# pdb.set_trace()
			if dis_idx * 10 + 10 == cfg.Dis_threshold:
				if max_or_seg == 'max' and cfg.evaluate_subset=='Reasonable':
					plt.loglog(MR_FPPI_array[:, 0], MR_FPPI_array[:, 1], label='%.2f%%(%.2f%%) %s\n[reasonable]' % (LAMR_2*100, LAMR_4*100, dis_txt), color=colors[dis_idx], linestyle='-', linewidth=2)
				else:
					plt.loglog(MR_FPPI_array[:, 0], MR_FPPI_array[:, 1], label='%.2f%%(%.2f%%) %s' % (LAMR_2 * 100, LAMR_4 * 100, dis_txt), color=colors[dis_idx], linestyle='-', linewidth=2)
			else:
				plt.loglog(MR_FPPI_array[:, 0], MR_FPPI_array[:, 1], label='%.2f%%(%.2f%%) %s' % (LAMR_2*100, LAMR_4*100, dis_txt), color=colors[dis_idx], linestyle='-', linewidth=1)

		# pdb.set_trace()
		plt.grid(color='b', linestyle='--', linewidth=0.5, alpha=0.3)
		plt.xlabel('false positives per image', font)
		plt.ylabel('miss rate ({})'.format(attr), font)
		# plt.xlim(0.00004, 0.2)
		plt.xlim(0.00003, 10)  # T35
		plt.ylim(0.001, 1)
		plt.yticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.14, 0.2, 0.3, 0.4, 0.5, 0.64, 0.8, 1],
				   ['.001', '.002', '.005', '.01', '.02', '.05', '.10', '.14', '.20', '.30', '.40', '.50', '.64', '.80', '1'])


		plt.tick_params(labelsize=font_size-10)  # 刻度字体大小13
		ax.spines['top'].set_visible(False)  # 顶边界不可见
		ax.spines['right'].set_visible(False)  # 右边界不可见
		legend = ax.legend(loc="lower left", fontsize=font_size-9)

		if max_or_seg == 'seg':
			legend = ax.legend(loc="upper right", fontsize=font_size-9)
		frame = legend.get_frame()
		frame.set_alpha(1)
		frame.set_edgecolor('none')  # 设置图例legend背景透明
		frame.set_facecolor('none')  # 设置图例legend背景透明
		plt.grid(color='b', linestyle='--', linewidth=0.5, alpha=0.3)
		plt.grid(True)

		fig.subplots_adjust(left=0.12, right=0.97, top=0.98, bottom=0.12)
		# plt.show()
		fig.savefig(os.path.join(os.path.dirname(detectResults_Path), 'Miss_RateVsFPPI_loglog_NIRPed_dis_{}_{}_{}.png'.format(max_or_seg, cfg.evaluate_subset, attr)), bbox_inches='tight')
	return None
#TODO:*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.

#TODO:*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.
def PR_Plot(PR_dis_dic, detectResults_Path, max_or_seg):# TODO: ***图像显示PR曲线。
	font_size = 24
	font = {'family': 'Times New Roman', 'weight': 'normal', 'size': font_size}
	colors = ['green', 'lime', 'blue', 'purple', 'hotpink', 'orangered', 'fuchsia', 'red', 'black']
	r_list = np.linspace(0, 1, 11)
	# pdb.set_trace()
	for attr in PR_dis_dic:
		fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # fig = plt.figure(facecolor='w')
		for dis_index in range(0, idx_Dis_max):
			dis_idx = idx_Dis_max-1 - dis_index
			PR_array = np.array(PR_dis_dic[attr][dis_idx])
			if PR_array.size == 0:
				if max_or_seg == 'seg':
					print('Did not detect any pedestrian in distance segment [%d,%d)]' % (dis_idx*10,(dis_idx+1)*10))
				elif max_or_seg == 'max':
					print('Did not detect any pedestrian in distance range [3,%d)]' % ((dis_idx + 1) * 10))
				continue

			pr_list = []
			for ri in r_list:
				index_ri = np.argmin(np.abs(PR_array[:, 1] - ri))
				pr_list.append(np.max(PR_array[index_ri:, 0]))

			AP = np.mean(pr_list)

			if max_or_seg == 'max':
				dis_txt = 'd<%dm' % (dis_idx * 10 + 10)
			elif max_or_seg == 'seg':
				dis_txt = '%d≤d<%dm' % (dis_idx * 10, dis_idx * 10 + 10)

			if dis_idx * 10 + 10 == cfg.Dis_threshold:
				if max_or_seg == 'max' and cfg.evaluate_subset=='Reasonable':
					plt.plot(PR_array[:, 0], PR_array[:, 1], label='AP=%.2f%% %s[reasonable]' % (100*AP, dis_txt), color=colors[dis_idx], linestyle='-', linewidth=2)
				else:
					plt.plot(PR_array[:, 0], PR_array[:, 1], label='AP=%.2f%% %s' % (100*AP, dis_txt), color=colors[dis_idx], linestyle='-', linewidth=2)
			else:
				plt.plot(PR_array[:, 0], PR_array[:, 1], label='AP=%.2f%% %s' % (100*AP, dis_txt), color=colors[dis_idx], linestyle='-', linewidth=1)

		# pdb.set_trace()
		plt.grid(color='b', linestyle='--', linewidth=0.5, alpha=0.3)
		plt.xlabel('recall', font)
		plt.ylabel('precision (%s)' % (attr), font)

		plt.xlim(0.4, 1)  # T35
		plt.ylim(0, 1)

		plt.tick_params(labelsize=font_size-10)  # 刻度字体大小13
		ax.spines['top'].set_visible(False)  # 顶边界不可见
		ax.spines['right'].set_visible(False)  # 右边界不可见
		legend = ax.legend(loc="lower left", fontsize=font_size-9)

		if max_or_seg == 'seg':
			legend = ax.legend(loc="upper right", fontsize=font_size-9)
		frame = legend.get_frame()
		frame.set_alpha(1)
		frame.set_edgecolor('none')  # 设置图例legend背景透明
		frame.set_facecolor('none')  # 设置图例legend背景透明
		plt.grid(color='b', linestyle='--', linewidth=0.5, alpha=0.3)
		plt.grid(True)

		# fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.09)
		fig.subplots_adjust(left=0.12, right=0.97, top=0.98, bottom=0.12)
		plt.show()
		# pdb.set_trace()
		time_now_min = datetime.datetime.now().strftime('%Y%m%d%H%M')
		fig.savefig(os.path.join(os.path.dirname(detectResults_Path), 'PR_NIRPed_dis_{}_{}_{}.png'.format(max_or_seg, cfg.evaluate_subset, attr, time_now_min)), bbox_inches='tight')
	return None

#TODO:*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.*** 图像显示MissRate_FPPI.
def post_processing(DE_conf, abs_flag = False):
	DE_max_dic = copy.deepcopy(DE_conf)
	DE_seg_dic = copy.deepcopy(DE_conf)
	for attr in DE_conf:
		Results_all_dis = []
		DE_results = {}
		for key_dis, v_dis in DE_conf[attr].items():
			Results_all_dis += v_dis
			num_dis = len(v_dis)
			if abs_flag:
				v_dis = np.abs(v_dis)
				# pdb.set_trace()
			if num_dis > 0:
				mean = round(np.mean(v_dis), 2)
				sigma = round(np.std(v_dis), 2)
			else:
				mean = 0
				sigma = 0
			DE_seg_dic[attr][key_dis] = [mean, sigma, num_dis]

			num_dis_all = len(Results_all_dis)
			if abs_flag:
				Results_all_dis = np.abs(Results_all_dis)
			if num_dis_all > 0:
				mean_all = round(np.mean(Results_all_dis), 2)
				sigma_all = round(np.std(Results_all_dis), 2)
			else:
				mean_all = 0
				sigma_all = 0
			DE_max_dic[attr][key_dis] = [mean_all, sigma_all, num_dis_all]

	return DE_seg_dic, DE_max_dic

#TODO: ***计算MissRate_FPPI曲线的函数。 ***计算MissRate_FPPI曲线的函数。 ***计算MissRate_FPPI曲线的函数。 ***计算MissRate_FPPI曲线的函数。
#TODO: ***计算MissRate_FPPI曲线的函数。 ***计算MissRate_FPPI曲线的函数。 ***计算MissRate_FPPI曲线的函数。 ***计算MissRate_FPPI曲线的函数。
def MR_FPPI_PR_Calculate(detectResults_Path):
	""" Create a '.temp_files/' and 'results/' directory 创建“.temp_files/”和“results/”目录"""
	TEMP_FILES_DIR = os.path.dirname(detectResults_Path)

	TEMP_FILES_Path = os.path.join(TEMP_FILES_DIR, '.temp_files')
	if not os.path.exists(TEMP_FILES_Path): # if it doesn't exist already如果'.temp_files'文件夹不存在与当前目录
		os.makedirs(TEMP_FILES_Path) #在程序当前目录下创建一个新的文件夹TEMP_FILES_Path = '.temp_files'

	gt_files_list, gt_counter_per_class, num_test_imgs = Temp_gt_Generation(TEMP_FILES_Path)
	gt_classes = list(gt_counter_per_class.keys())  # 获取标记框分类计数的键值——类别，并转化成list=['Ped', 'Peo', 'Bic', 'Mot', 'Ign', 'bg']

	for class_index, class_name in enumerate(gt_classes): #gt_classes = ['pedestrian', 'motorbikedriver', 'ignore', 'bicycledriver']  ['bg', 'Peo', 'Ped', 'Mot', 'Ign', 'Bic']
		"""Load detection-result of that class 加载该类的检测结果"""
		if class_name not in ['Ped', 'ped', 'Pedestrian', 'pedestrian']:
			continue
		dt_data = json.load(open(detectResults_Path)) #加载json文件中的数据
		num_dt_boxes = len(dt_data) #  网络检测到的当前类的总数量（包括正正例tp和假正例FP）：num_dt_boxes = 1639
		ele = np.zeros((num_dt_boxes, 12))
		tp_dic = {'All':copy.deepcopy(ele), 'Sun':copy.deepcopy(ele), 'Rain':copy.deepcopy(ele), 'Spate':copy.deepcopy(ele),
				'short':copy.deepcopy(ele), 'normal_height':copy.deepcopy(ele), 'tall':copy.deepcopy(ele)}
		fp_dic = copy.deepcopy(tp_dic)

		MR_FPPI_dis_max_dic = {'All':{}, 'Sun':{}, 'Rain':{}, 'Spate':{}, 'short':{}, 'normal_height':{}, 'tall':{}}
		MR_FPPI_dis_seg_dic = copy.deepcopy(MR_FPPI_dis_max_dic)
		PR_dis_max_dic = copy.deepcopy(MR_FPPI_dis_max_dic)
		PR_dis_seg_dic = copy.deepcopy(MR_FPPI_dis_max_dic)

		AE_dtp_max_dic = copy.deepcopy(MR_FPPI_dis_max_dic)
		AE_dtp_seg_dic = copy.deepcopy(MR_FPPI_dis_max_dic)
		AE_DE_max_dic = copy.deepcopy(MR_FPPI_dis_max_dic)
		AE_DE_seg_dic = copy.deepcopy(MR_FPPI_dis_max_dic)

		AER_dtp_max_dic = copy.deepcopy(MR_FPPI_dis_max_dic)
		AER_dtp_seg_dic = copy.deepcopy(MR_FPPI_dis_max_dic)
		AER_DE_max_dic = copy.deepcopy(MR_FPPI_dis_max_dic)
		AER_DE_seg_dic = copy.deepcopy(MR_FPPI_dis_max_dic)

		annos_AER_bad = {}
		annos_FN = {}
		annos_FP = {}

		for attr in MR_FPPI_dis_max_dic:
			for dis_index in range(0, idx_Dis_max):
				MR_FPPI_dis_max_dic[attr][dis_index] = []
				MR_FPPI_dis_seg_dic[attr][dis_index] = []
				PR_dis_max_dic[attr][dis_index] = []
				PR_dis_seg_dic[attr][dis_index] = []
				AE_dtp_max_dic[attr][dis_index] = []
				AE_dtp_seg_dic[attr][dis_index] = []
				AE_DE_max_dic[attr][dis_index] = []
				AE_DE_seg_dic[attr][dis_index] = []
				AER_dtp_max_dic[attr][dis_index] = []
				AER_dtp_seg_dic[attr][dis_index] = []
				AER_DE_max_dic[attr][dis_index] = []
				AER_DE_seg_dic[attr][dis_index] = []

		"""#*************************************************************************对当前类所有检测框与真实标记框进行匹配，并做好TP/FP记录。"""
		min_overlap = MINOVERLAP  # 最小交并比设置为：MINOVERLAP = 0.5
		gt_file_used = []
		ignored_dt_count = [0]*12 #TODO: 记录被忽略检测框的数量。
		# dis_ignored_dt_count = 0 #TODO: 记超出距离范围而被忽略检测框的数量。
		# TODO: 按检测框得分高低开始评估MR-FPPI和AP。
		# TODO: 按检测框得分高低开始评估MR-FPPI和AP。
		# idx = 0, box_detection = {'class_name': 'Ped', 'confidence': '1.0', 'file_id': 'Data20200624201506_971695N850F12', 'bbox': [1040.0, 202.0, 1168.0, 517.0], 'Dis': 14.78}
		for idx, box_detection in enumerate(dt_data):
			img_id = box_detection['file_id']
			if img_id[:10] in ['Data202007']:  #TODO：更改2020年7月份拍摄的图像的年份到2018年。
				img_id = 'Data201807' + img_id[10:]
			# assign box_detection-result to ground truth object if any
			# open ground-truth with that file_id
			gt_file = os.path.join(TEMP_FILES_Path, img_id + '_gt.json')  # gt_file = '.temp_files/Tongxy13m161R850_gt.json'
			# gt_file_temp = os.path.join(TEMP_FILES_Path_temp, img_id + '_gt_temp.json')  # gt_file = '.temp_files/Tongxy13m161R850_gt.json'

			# if img_id not in gt_file_used:
			try:
				gt_data_temp = json.load(open(gt_file))
			except:
				print("\033[5;31;47m\tImage:{}, is not in ground-truth. Please check that ground-truth and detect results are matched.\033[0m".format(img_id))
				continue
			# else:
			# 	gt_data_temp = json.load(open(gt_file_temp))
			# gt_data = [{'class_name': 'Ped', 'bbox': '101 312 217 580 16.3', 'used': False}, {'class_name': 'Ped', 'bbox': '384 281 460 500 20.9', 'used': False}]
			ovmax = -1  # 最大交并比初值设为-1
			a_percentage_max = -1  # 最大交并比初值设为-1
			gt_matched = -1  # 当前第idx预测框没有匹配上GT
			# load detected object bounding-box
			bbdtF = [float(x) for x in box_detection['bbox']]  # 检测框信息：bbdtF = [1040.0, 202.0, 1168.0, 517.0]
			cx_dt = 0.5*(bbdtF[0]+bbdtF[2])
			cy_dt = 0.5*(bbdtF[1]+bbdtF[3])
			width_dt = bbdtF[2] - bbdtF[0]
			height_dt = bbdtF[3] - bbdtF[1]

			dis_dt = min(110, box_detection['Dis'])
			dis_dt_index = int(dis_dt//10)
			in_boxes = False
			""" Assign detection-result to ground-truth objects 将检测结果与标记对象进行匹配。"""
			for obj in gt_data_temp:
				'''gt_data_temp=[{"class_name": "Ped", "bbox": [39, 191, 153, 525], "Dis": 13.4, "used": false, "Dif": false, "Occ_Coe": 0.0, "Age": "Adult", "area": 38076}, 
				{"class_name": "Ped", "bbox": [434, 208, 569, 511], "Dis": 13.0, "used": false, "Dif": false, "Occ_Coe": 0.0, "Age": "Adult", "area": 40905}, 
				{"class_name": "Ped", "bbox": [196, 217, 346, 517], "Dis": 13.2, "used": false, "Dif": false, "Occ_Coe": 0.0, "Age": "Adult", "area": 45000}]'''
				# look for a class_name match obj = {'class_name': 'Ped', 'bbox': '466 313 518 433 38', 'used': False, 'Dif': True}
				if obj['class_name'] != 'Bg_Img':
					bbgtF = [float(x) for x in obj['bbox']]  # 真实标记框信息：bbgtF = [466.0, 313.0, 518.0, 433.0, 38.0]
					if (bbgtF[0]-2 <= cx_dt <= bbgtF[2]+2) and (bbgtF[1]-2 <= cy_dt <= bbgtF[3]+2):
						in_boxes = True
						bi = [max(bbdtF[0], bbgtF[0]), max(bbdtF[1], bbgtF[1]), min(bbdtF[2], bbgtF[2]), min(bbdtF[3], bbgtF[3])]  # 用于计算检测框和真实标记框的交集面积
						iw = bi[2] - bi[0] + 1  # 计算交集宽
						ih = bi[3] - bi[1] + 1  # 计算交集高
						if iw > 0 and ih > 0:  # 如果相交，才开始计算交并比 compute overlap (IoU) = area of intersection / area of union
							if obj['class_name'] in ['Ped', 'ped', 'Pedestrian', 'pedestrian']:
								ua = (width_dt + 1) * (height_dt + 1) + (bbgtF[2] - bbgtF[0] + 1) * (bbgtF[3] - bbgtF[1] + 1) - iw * ih  # 计算并集面积
								ov = iw * ih / ua  # 计算交并比
								if ov > ovmax:
									ovmax = ov
									gt_matched = obj  # 用gt_match记录与当前检测框bb匹配的真实标记框gt_match = {'class_name': 'Bic', 'bbox': [1035, 229, 1156, 526],
														# 'Dis': 15.4, 'used': False, 'Dif': False, 'Occ_Coe': 0.0, 'Age': 'Adult', 'area': 35937}
							else:
								a_percentage = iw * ih / ((width_dt + 1) * (height_dt + 1))
								if a_percentage > a_percentage_max:
									a_percentage_max = a_percentage
									gt_matched = obj

			try:  # gt_match={'class_name': 'Ped', 'bbox': '98 309 212 584 16.3', 'used': False, 'scale_dis': 'near'}gt_match=-1将出错不能进行。
				#if gt_matched['class_name'] in ['Peo', 'Ign', 'ignore', 'bicycledriver', 'Bic', 'Cyc', 'motorbikedriver', 'Mot']:
				if not in_boxes:   #TODO:{'class_name': 'Ped', 'confidence': '1.0', 'file_id': 'Data20190113195620_050000', 'bbox': [1040.0, 0.0, 1312.0, 652.0], 'Dis': 20520.62}
					gt_matched = gt_data_temp[0]
					fp_dic['All'][idx, dis_dt_index] = 1
					fp_dic[gt_matched['Wea']][idx, dis_dt_index] = 1
					fp_dic[gt_matched['Height']][idx, dis_dt_index] = 1
				else: #TODO:对匹配上扎推的人群、骑单车、摩托车的人、背景画、或者人类都难以识别的行人进行忽略，即：既不判定为TP又不判定为FP。
					if gt_matched['class_name'] in ['Ped', 'ped', 'Pedestrian', 'pedestrian']:
						if ovmax >= min_overlap:  # 最小交并比设置为：min_overlap = MINOVERLAP = 0.5
							dis_gt = gt_matched['Dis']
							dis_gt_index = int(dis_gt // 10)
							bbgtF = [float(x) for x in gt_matched['bbox']]
							if cfg.evaluate_subset == 'Reasonable':
								if not (gt_matched['Dif'] or float(gt_matched['Occ_Coe']) > cfg.Occ_threshold or dis_gt >= cfg.Dis_threshold):
									if not bool(gt_matched['used']):  # 换成布尔值
										tp_dic['All'][idx, dis_gt_index] = 1
										tp_dic[gt_matched['Wea']][idx, dis_gt_index] = 1
										tp_dic[gt_matched['Height']][idx, dis_gt_index] = 1

										# TODO：正确检测目标的距离误差求解
										dis_dtp = round(Hf / height_dt, 2)  # TODO: 将预测框像素高度直接估算为距离。
										AE_dtp_instance = abs(dis_dtp - dis_gt)  # TODO:AE=absolute_error
										AE_dtp_seg_dic['All'][dis_gt_index].append(AE_dtp_instance)
										AE_dtp_seg_dic[gt_matched['Wea']][dis_gt_index].append(AE_dtp_instance)
										AE_dtp_seg_dic[gt_matched['Height']][dis_gt_index].append(AE_dtp_instance)

										AE_DE_instance = abs(dis_dt - dis_gt)  # TODO:AE=absolute_error
										AE_DE_seg_dic['All'][dis_gt_index].append(AE_DE_instance)
										AE_DE_seg_dic[gt_matched['Wea']][dis_gt_index].append(AE_DE_instance)
										AE_DE_seg_dic[gt_matched['Height']][dis_gt_index].append(AE_DE_instance)

										AER_dtp_instance = round(100 * AE_dtp_instance / dis_gt, 3)  # TODO:AE=error
										AER_dtp_seg_dic['All'][dis_gt_index].append(AER_dtp_instance)
										AER_dtp_seg_dic[gt_matched['Wea']][dis_gt_index].append(AER_dtp_instance)
										AER_dtp_seg_dic[gt_matched['Height']][dis_gt_index].append(AER_dtp_instance)

										AER_DE_instance = round(100 * AE_DE_instance / dis_gt, 3)  # TODO:AER = error_rate
										AER_DE_seg_dic['All'][dis_gt_index].append(AER_DE_instance)
										AER_DE_seg_dic[gt_matched['Wea']][dis_gt_index].append(AER_DE_instance)
										AER_DE_seg_dic[gt_matched['Height']][dis_gt_index].append(AER_DE_instance)

										if (abs(AER_DE_instance) > 15):
											if img_id not in annos_AER_bad:
												annos_AER_bad[img_id] = []
											annos_AER_bad[img_id].append({'class_name': gt_matched['class_name'],
																		  'BB_gt': gt_matched['bbox'], 'Dis_gt': dis_gt,
																		  'BB_dt': bbdtF,
																		  'Dis_dt': dis_dt, 'Wea': gt_matched['Wea'],
																		  'Height': gt_matched['Height']})

										# TODO：正确检测目标的距离误差求解
										gt_matched['used'] = True  # 真实标记框已被检测到，标记为已使用，以后的预测框不再考虑此真实标记框。
										with open(gt_file, 'w') as f:  # gt_file = '.temp_files/Tongxy13m161R850_gt.json'
											f.write(json.dumps(gt_data_temp))  # 更新gt_data
										f.close()
										if img_id not in gt_file_used:
											gt_file_used.append(img_id)
									else:# false positive (multiple box_detection)
										fp_dic['All'][idx, dis_gt_index] = 1
										fp_dic[gt_matched['Wea']][idx, dis_gt_index] = 1
										fp_dic[gt_matched['Height']][idx, dis_gt_index] = 1
								else: #TODO: 忽略此检测框，不计入误报。
									ignored_dt_count[dis_gt_index] += 1

							elif cfg.evaluate_subset == 'All':
								if dis_dt < cfg.Dis_threshold:
									if not bool(gt_matched['used']):  # 换成布尔值
										tp_dic['All'][idx, dis_gt_index] = 1
										tp_dic[gt_matched['Wea']][idx, dis_gt_index] = 1
										tp_dic[gt_matched['Height']][idx, dis_gt_index] = 1

										# TODO：正确检测目标的距离误差求解
										dis_dtp = round(Hf / height_dt, 2)  # TODO: 将预测框像素高度直接估算为距离。
										AE_dtp_instance = abs(dis_dtp - dis_gt)  # TODO:AE=absolute_error
										# AE_dtp_instance = dis_dtp - dis_gt   #TODO:AE=absolute_error
										AE_dtp_seg_dic['All'][dis_gt_index].append(AE_dtp_instance)
										AE_dtp_seg_dic[gt_matched['Wea']][dis_gt_index].append(AE_dtp_instance)
										AE_dtp_seg_dic[gt_matched['Height']][dis_gt_index].append(AE_dtp_instance)

										AE_DE_instance = abs(dis_dt - dis_gt)  # TODO:AE=absolute_error
										AE_DE_seg_dic['All'][dis_gt_index].append(AE_DE_instance)
										AE_DE_seg_dic[gt_matched['Wea']][dis_gt_index].append(AE_DE_instance)
										AE_DE_seg_dic[gt_matched['Height']][dis_gt_index].append(AE_DE_instance)

										AER_dtp_instance = round(100 * AE_dtp_instance / dis_gt, 3)  # TODO:AE=error
										AER_dtp_seg_dic['All'][dis_gt_index].append(AER_dtp_instance)
										AER_dtp_seg_dic[gt_matched['Wea']][dis_gt_index].append(AER_dtp_instance)
										AER_dtp_seg_dic[gt_matched['Height']][dis_gt_index].append(AER_dtp_instance)

										AER_DE_instance = round(100 * AE_DE_instance / dis_gt, 3)  # TODO:AER = error_rate
										AER_DE_seg_dic['All'][dis_gt_index].append(AER_DE_instance)
										AER_DE_seg_dic[gt_matched['Wea']][dis_gt_index].append(AER_DE_instance)
										AER_DE_seg_dic[gt_matched['Height']][dis_gt_index].append(AER_DE_instance)

										if (abs(AER_DE_instance) > 15):
											if img_id not in annos_AER_bad:
												annos_AER_bad[img_id] = []
											annos_AER_bad[img_id].append({'class_name': gt_matched['class_name'],
																	  'BB_gt': gt_matched['bbox'], 'Dis_gt': dis_gt,
																	  'BB_dt': bbdtF,
																	  'Dis_dt': dis_dt, 'Wea': gt_matched['Wea'],
																	  'Height': gt_matched['Height']})

										gt_matched['used'] = True  # 真实标记框已被检测到，标记为已使用，以后的预测框不再考虑此真实标记框。
										with open(gt_file, 'w') as f:  # gt_file = '.temp_files/Tongxy13m161R850_gt.json'
											f.write(json.dumps(gt_data_temp))  # 更新gt_data
										f.close()
										if img_id not in gt_file_used:
											gt_file_used.append(img_id)
									else:# TODO:第idx个检测框匹配上当前类的标记框，且交并比不小于min_overlap=0.5，但此标记框在之前已被检测框匹配到了，判定为真正例FP
										fp_dic['All'][idx, dis_gt_index] = 1
										fp_dic[gt_matched['Wea']][idx, dis_gt_index] = 1
										fp_dic[gt_matched['Height']][idx, dis_gt_index] = 1
								else: #TODO: 忽略此检测框，不计入误报。
									ignored_dt_count[dis_gt_index] += 1
						elif ovmax > 0:  # 如果ovmax < min_overlap: # false positive
							if cfg.evaluate_subset == 'Reasonable':
								if not (gt_matched['Dif'] or float(gt_matched['Occ_Coe']) > cfg.Occ_threshold or dis_dt >= cfg.Dis_threshold):
									fp_dic['All'][idx, dis_dt_index] = 1
									fp_dic[gt_matched['Wea']][idx, dis_dt_index] = 1
									fp_dic[gt_matched['Height']][idx, dis_dt_index] = 1
								else: #TODO: 忽略此检测框，不计入误报。
									ignored_dt_count[dis_dt_index] += 1
							elif cfg.evaluate_subset == 'All':
								if dis_dt < cfg.Dis_threshold:
									fp_dic['All'][idx, dis_dt_index] = 1
									fp_dic[gt_matched['Wea']][idx, dis_dt_index] = 1
									fp_dic[gt_matched['Height']][idx, dis_dt_index] = 1
								else: #TODO: 忽略此检测框，不计入误报。
									ignored_dt_count[dis_dt_index] += 1

					else:  # elif gt_matched['class_name'] in ['Peo', 'Bic', 'Mot', 'Ign', 'bg', 'Bg', 'BG']:
						if a_percentage_max >= min_overlap:  #TODO:匹配上任何忽略区域，将被忽略。
							ignored_dt_count[dis_dt_index] += 1
							#print('Exception: {} in img_id:{}'.format(e, img_id))
						else:  #TODO:没有匹配上目标，也没有匹配上任何忽略区域，检测框判定为假正例fp。
							fp_dic['All'][idx, dis_dt_index] = 1
							fp_dic[gt_matched['Wea']][idx, dis_dt_index] = 1
							fp_dic[gt_matched['Height']][idx, dis_dt_index] = 1

			except Exception as e:  #TODO:对没有匹配上任何目标的检测框判定为假正例fp。
				# errormessage = '{}'.format(e)
				s = sys.exc_info()
				print("Exception: Error '%s' happened on line %d with image:'%s'" % (s[1], s[2].tb_lineno, img_id))
				pdb.set_trace()
				#fp_all[idx, dis_dt_index] = 1


			if idx % idx_step != 0:  #每个100个检测结果，计算一次MR_FPPI和PR
				continue
			flag_break = 0

			for attr in MR_FPPI_dis_max_dic:
				for dis_index in range(0, idx_Dis_max):
					num_tp = tp_dic[attr][:idx+1, dis_index:dis_index+1].sum()  # 找出不同距离段的TP的位置序列，#正确检测到的行人统计。tp_all = np.zeros((num_dt_boxes, 1))
					num_fp = fp_dic[attr][:idx+1, dis_index:dis_index+1].sum()  # 找出不同距离段的TP的位置序列，#错误检测到的行人统计。fp_all = np.zeros((num_dt_boxes, 1))
					num_gt_boxes = sum(gt_counter_per_class[class_name][attr][dis_index:dis_index+1])
					FPPI_seg = round(num_fp/num_test_imgs[attr], 6)
					if num_gt_boxes == 0:
						MissRate_seg=0
					else:
						MissRate_seg = round((num_gt_boxes - num_tp) / num_gt_boxes, 6)

					if num_tp+num_fp != 0: #防止num_tp=0.0; num_fp=0.0
						re_seg = 1 - MissRate_seg
						pr_seg = round(num_tp/(num_tp+num_fp), 6)
						PR_dis_seg_dic[attr][dis_index].append([pr_seg, re_seg])

					MR_FPPI_dis_seg_dic[attr][dis_index].append([FPPI_seg, MissRate_seg])

					num_tp = tp_dic[attr][:idx+1, :dis_index+1].sum()  # 找出不同距离段的TP的位置序列，#正确检测到的行人统计。tp_all = np.zeros((num_dt_boxes, 1))
					num_fp = fp_dic[attr][:idx+1, :dis_index+1].sum()  # 找出不同距离段的TP的位置序列，#错误检测到的行人统计。fp_all = np.zeros((num_dt_boxes, 1))
					num_gt_boxes = sum(gt_counter_per_class[class_name][attr][:dis_index+1])
					FPPI_max = round(num_fp / num_test_imgs[attr], 6)
					if num_gt_boxes == 0:
						MissRate_max = 0
					else:
						MissRate_max = round((num_gt_boxes - num_tp) / num_gt_boxes, 6)

					MR_FPPI_dis_max_dic[attr][dis_index].append([FPPI_max, MissRate_max])

					if (num_tp + num_fp) != 0:  #防止num_tp=0.0; num_fp=0.0
						re_max = 1-MissRate_max
						pr_max = round(num_tp / (num_tp+num_fp), 6)
						PR_dis_max_dic[attr][dis_index].append([pr_max, re_max])

					if (attr == 'All') and (dis_index*10+10 == cfg.Dis_threshold):
						score_threshold = float(box_detection['confidence'])
						if idx % 200 == 0:
							print('idx=%d/%d;d<=%d;score_threshold=\033[30;41m %.4f\033[0m:[FPPI, MissRate]=[\033[30;42m %.6f\033[0m, \033[30;41m %.4f\033[0m]; ignored_dt_count=%d' % (idx, num_dt_boxes, dis_index * 10 + 10, score_threshold, FPPI_max, MissRate_max, sum(ignored_dt_count)))
							# print(ignored_dt_count)
						if FPPI_max > 10:
							flag_break = 1 #TODO: 提前终止计算标识。

			if flag_break == 1:
				break

			# TODO:***将判定为假正例fp的检测框保存下来，为后续可视化做准备。***将判定为假正例fp的检测框保存下来，为后续可视化做准备。***将判定为假正例fp的检测框保存下来，为后续可视化做准备。
			if max(fp_dic['All'][idx, :int(cfg.Dis_threshold // 10)]) == 1:
				if img_id not in annos_FP:
					annos_FP[img_id] = []
				annos_FP[img_id].append({'class_name': box_detection['class_name'], 'confidence': box_detection['confidence'],
									'BB_dt': box_detection['bbox'], 'Dis_dt': box_detection['Dis'], 'Wea':gt_matched['Wea'],
									'Height':gt_matched['Height']})

		for gt_file in gt_files_list:
			img_id = os.path.basename(gt_file)
			img_id = img_id.split('_gt', 1)[0]
			try:
				gt_data_temp = json.load(open(gt_file))
			except:
				print("\033[5;31;47m\tImage:{}, is not in ground-truth. Please check that ground-truth and detect results are matched.\033[0m".format(img_id))
				continue
			for anno in gt_data_temp:
				if anno['class_name'] not in ['Ped', 'ped', 'Pedestrian', 'pedestrian']:
					continue
				if anno['used']:
					continue

				if cfg.evaluate_subset == 'Reasonable':
					if anno['Dif'] or (anno['Occ_Coe'] > cfg.Occ_threshold) or (anno['Dis'] > cfg.Dis_threshold):  #TODO：检查此步是否执行到位？？？？
						continue
				elif cfg.evaluate_subset == 'All':
					if anno['Dis'] > cfg.Dis_threshold:  #TODO：检查此步是否执行到位？？？？
						continue
				if img_id not in annos_FN:
					annos_FN[img_id] = []
				annos_FN[img_id].append(anno)

		AE_dtp_seg_dic, AE_dtp_max_dic = post_processing(AE_dtp_seg_dic)
		AE_DE_seg_dic, AE_DE_max_dic = post_processing(AE_DE_seg_dic)
		AER_dtp_seg_dic, AER_dtp_max_dic = post_processing(AER_dtp_seg_dic)
		AER_DE_seg_dic, AER_DE_max_dic = post_processing(AER_DE_seg_dic)

		# TODO:***将判定为假正例fp的检测框保存下来，为后续可视化做准备。***将判定为假正例fp的检测框保存下来，为后续可视化做准备。***将判定为假正例fp的检测框保存下来，为后续可视化做准备。
		# pdb.set_trace()
		print('\n\nMR_FPPI_dis_max_dic:')
		MR_FPPI_Plot(MR_FPPI_dis_max_dic, detectResults_Path, 'max') #TODO:max_or_seg
		print('\n\nPR_dis_max_dic:')
		PR_Plot(PR_dis_max_dic, detectResults_Path, 'max') #TODO:max_or_seg
		print('\n\nMR_FPPI_dis_seg_dic:')
		MR_FPPI_Plot(MR_FPPI_dis_seg_dic, detectResults_Path, 'seg') #TODO:max_or_seg
		print('\n\nPR_dis_seg_dic:')
		PR_Plot(PR_dis_seg_dic, detectResults_Path, 'seg')  # TODO:max_or_seg

	shutil.rmtree(TEMP_FILES_Path)  # 删除临时文件夹及其下所有文件。
	# shutil.rmtree(TEMP_FILES_Path_temp)  # 删除临时文件夹及其下所有文件。
	return MR_FPPI_dis_max_dic, MR_FPPI_dis_seg_dic,PR_dis_max_dic,PR_dis_seg_dic, AE_dtp_seg_dic, AE_dtp_max_dic,\
		   AE_DE_seg_dic, AE_DE_max_dic,AER_dtp_seg_dic, AER_dtp_max_dic,AER_DE_seg_dic, AER_DE_max_dic,\
		   annos_AER_bad, annos_FN, annos_FP
#TODO: ***计算MissRate_FPPI曲线的函数。 ***计算MissRate_FPPI曲线的函数。 ***计算MissRate_FPPI曲线的函数。 ***计算MissRate_FPPI曲线的函数。TEMP_FILES_Path_temp

#TODO:*** 图像显示检测结果.***图像显示检测结果.***图像显示检测结果.***图像显示检测结果.***图像显示检测结果.***图像显示检测结果.***图像显示检测结果.
#TODO:*** 图像显示检测结果.***图像显示检测结果.***图像显示检测结果.***图像显示检测结果.***图像显示检测结果.***图像显示检测结果.***图像显示检测结果.
def visulize_FP_FN_bad_ER(imgs_data, dt_data_plot, imgs_dt_show_dir, annos_FP_Dis, annos_AER_bad_Dis, annos_miss_Ped_Dis, score_threshold):
	# TODO: ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。
	# TODO: ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。
	txt_output_path = os.path.join(imgs_dt_show_dir, 'annos_GT_txt_FP_AER_bad_miss_Ped.txt')
	annos_GT_txt = open(txt_output_path, 'w')
	Dis_threshold = 50
	annos_GT_txt.write('path, x1, y1, x2, y2, Dis, Age, Dif, Sce, Wea, Occlusion, Ignore, pose, truncation, ID, cls\n')

	# TODO: ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。

	num_Bg_Img = 700000
	cocoGt = COCO(cfg.annos_file)
	images = imgs_data['images']

	scenes = imgs_data['scenes']
	weathers = imgs_data['weathers']

	path_img_dt_FP_region_dir = os.path.join(imgs_dt_show_dir, 'visualize_FP')
	if not os.path.exists(path_img_dt_FP_region_dir):
		os.makedirs(path_img_dt_FP_region_dir)
	path_img_dt_FP_region_dir_Bg = os.path.join(path_img_dt_FP_region_dir, 'visualize_FP_Bg')
	if not os.path.exists(path_img_dt_FP_region_dir_Bg):
		os.makedirs(path_img_dt_FP_region_dir_Bg)
	path_img_dt_FP_region_dir_FP = os.path.join(path_img_dt_FP_region_dir, 'visualize_FP_FP')
	if not os.path.exists(path_img_dt_FP_region_dir_FP):
		os.makedirs(path_img_dt_FP_region_dir_FP)
	path_img_dt_FP_region_dir_TP = os.path.join(path_img_dt_FP_region_dir, 'visualize_FP_TP')
	if not os.path.exists(path_img_dt_FP_region_dir_TP):
		os.makedirs(path_img_dt_FP_region_dir_TP)

	path_img_dt_AER_bad_region_dir = os.path.join(imgs_dt_show_dir, 'visualize_AER_bad')
	if not os.path.exists(path_img_dt_AER_bad_region_dir):
		os.makedirs(path_img_dt_AER_bad_region_dir)
	path_img_dt_AER_bad_region_dir_plus = os.path.join(path_img_dt_AER_bad_region_dir, 'visualize_AER_bad_Plus')
	if not os.path.exists(path_img_dt_AER_bad_region_dir_plus):
		os.makedirs(path_img_dt_AER_bad_region_dir_plus)
	path_img_dt_AER_bad_region_dir_minus = os.path.join(path_img_dt_AER_bad_region_dir, 'visualize_AER_bad_minus')
	if not os.path.exists(path_img_dt_AER_bad_region_dir_minus):
		os.makedirs(path_img_dt_AER_bad_region_dir_minus)

	path_img_dt_FN_region_dir = os.path.join(imgs_dt_show_dir, 'visualize_FN')
	if not os.path.exists(path_img_dt_FN_region_dir):
		os.makedirs(path_img_dt_FN_region_dir)
	path_img_dt_FN_region_dir_Bg = os.path.join(path_img_dt_FN_region_dir, 'visualize_FN_Bg')
	if not os.path.exists(path_img_dt_FN_region_dir_Bg):
		os.makedirs(path_img_dt_FN_region_dir_Bg)
	path_img_dt_FN_region_dir_FP = os.path.join(path_img_dt_FN_region_dir, 'visualize_FN_FP')
	if not os.path.exists(path_img_dt_FN_region_dir_FP):
		os.makedirs(path_img_dt_FN_region_dir_FP)
	num_test_imgs = len(images)
	for img_idx, img_data in enumerate(images):  # img_data={'id': 100000, 'file_name': 'Data20181219200348_010000.png', 'height': 720, 'width': 1280, 'daytime': 'night', 'scenes_id': 2, 'weathers_id': 1, 'seasons_id': 0, 'recordings_id': 0, 'imageset': 'train'}
		img_name = img_data['file_name']
		image_id = img_name.split('.', 1)[0]

		print('\033[1;30;43m Process:%d / %d \033[0m:%s' % (img_idx, num_test_imgs, image_id))

		# img_path = os.path.join('E:\\Datasets\\NIRPed2021\\NIRPed\\images\\{}\\{}'.format(imageset, img_name))
		try:
			img_path = '.\\data\\miniNIRPed\\images\\{}\\{}'.format(imageset, img_name)
		except:
			continue


		anno_ids = cocoGt.getAnnIds(imgIds=img_data['id'])  # anno_id=1037542
		annos_GT = cocoGt.loadAnns(ids=anno_ids)
		'''annos_GT =[{'occluded': None, 'difficult': None, 'bbox': [453, 207, 30, 54], 'id': 1000007, 'category_id': 4, 'image_id': 1000043, 'pose_id': 5, 'tracking_id': 1000000, 'ignore': 1, 'area': 1620, 'truncated': False}, 
			{'occluded': None, 'difficult': None, 'bbox': [514, 233, 95, 40], 'id': 1000010, 'category_id': 4, 'image_id': 1000043, 'pose_id': 5, 'tracking_id': 1000001, 'ignore': 1, 'area': 3800, 'truncated': False}]
		 '''
		Wea = weathers[img_data['weathers_id']]['name']
		Sce = scenes[img_data['scenes_id']]['name']

		img_dt_results_path = os.path.join(imgs_dt_show_dir, image_id + '.png')
		if os.path.exists(img_dt_results_path):
			print('Notion:Detect results have shown for image {} .'.format(img_dt_results_path))
			continue
		dt_boxes = [box for box in dt_data_plot if box['file_id'] == image_id]
		# dt_boxes = [{'class_name': 'Ped', 'confidence': '0.999998', 'file_id': 'Data20181219200348_020000', 'bbox': [448.0, 202.0, 576.0, 517.0], 'Dis': 13.91},
		# {'class_name': 'Ped', 'confidence': '0.999998', 'file_id': 'Data20181219200348_020000', 'bbox': [448.0, 202.0, 576.0, 517.0], 'Dis': 13.91},
		# {'class_name': 'Ped', 'confidence': '0.999997', 'file_id': 'Data20181219200348_020000', 'bbox': [32.0, 179.0, 160.0, 539.0], 'Dis': 12.74},
		# {'class_name': 'Ped', 'confidence': '0.999997', 'file_id': 'Data20181219200348_020000', 'bbox': [32.0, 179.0, 160.0, 539.0], 'Dis': 12.74},
		# {'class_name': 'Ped', 'confidence': '0.999964', 'file_id': 'Data20181219200348_020000', 'bbox': [208.0, 202.0, 336.0, 494.0], 'Dis': 15.45},
		# {'class_name': 'Ped', 'confidence': '0.999964', 'file_id': 'Data20181219200348_020000', 'bbox': [208.0, 202.0, 336.0, 494.0], 'Dis': 15.45}]
		if dt_boxes == []:
			print('Notion:image {} detect no object. Continue!'.format(img_path))
			#continue
		try:
			img = cv2.imread(img_path)
			# (rows, cols) = img.shape[:2]
		except:
			print('Notion:do not find image: {}.'.format(img_path))
			#pdb.set_trace()
		# TODO: ***将背景图片写入记事本。  ***将背景图片写入记事本。  ***将背景图片写入记事本。  ***将背景图片写入记事本。  ***将背景图片写入记事本。  ***将背景图片写入记事本。
		# TODO: ***将背景图片写入记事本。  ***将背景图片写入记事本。  ***将背景图片写入记事本。  ***将背景图片写入记事本。  ***将背景图片写入记事本。  ***将背景图片写入记事本。
		if annos_GT == []:
			#num_Bg_Img = num_Bg_Img + 1
			annos_GT_txt.write(img_dt_results_path + ',' + '0,0,0,0' + ',' + '100' + ',' + '0,0,0,0' + ',' + 'Neu' + ',' + 'Esay' + ',' + Sce + ',' + Wea + ',' + str(num_Bg_Img) + ',' + 'Bg_Img' + '\n')

		# TODO: ***将背景图片写入记事本。  ***将背景图片写入记事本。  ***将背景图片写入记事本。  ***将背景图片写入记事本。  ***将背景图片写入记事本。  ***将背景图片写入记事本。

		image_with_boxes =Image.fromarray(np.uint8(img)) # image_with_boxes = image_with_boxes.resize((width, height), Image.ANTIALIAS)/= image_with_boxes.resize((cfg.im_cols_show, cfg.im_rows_show), Image.BICUBIC)
		draw = ImageDraw.Draw(image_with_boxes)
		'''gt_data_temp=[{"class_name": "Ped", "bbox": [39, 191, 153, 525], "Dis": 13.4, "used": false, "Dif": false, "Occ_Coe": 0.0, "Age": "Adult", "area": 38076}, 
						{"class_name": "Ped", "bbox": [434, 208, 569, 511], "Dis": 13.0, "used": false, "Dif": false, "Occ_Coe": 0.0, "Age": "Adult", "area": 40905}, 
						{"class_name": "Ped", "bbox": [196, 217, 346, 517], "Dis": 13.2, "used": false, "Dif": false, "Occ_Coe": 0.0, "Age": "Adult", "area": 45000}]'''

		# TODO: ***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。
		# TODO: ***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。
		if image_id in annos_FP_Dis:
			annos_FP = annos_FP_Dis[image_id]
			# 	pdb.set_trace()
			for index, anno in enumerate(annos_FP): #anno={'class_name': 'Ped', 'confidence': '0.999982', 'BB_dt': [240.0, 224.0, 288.0, 404.0], 'Dis_dt': 25.35}
				b = anno['BB_dt']
				left, top, right, bottom = int(b[0]), int(b[1]), int(b[2]), int(b[3])
				#draw.rectangle([left-2, top-2, right+2, bottom+2], outline=color_FP[0], width=size_font_highlight)  # 淡红色单框：缩放到resized图上标记框
				# TODO:***切片显示FP。***切片显示FP。***切片显示FP。***切片显示FP。***切片显示FP。***切片显示FP。
				ovmax = 0
				'''annos_GT=[{'id': 10036185, 'category_id': 1, 'image_id': 107911, 'pose_id': 0, 'tracking_id': 100000, 'bbox': [294, 200, 68, 225], 'Dis': 19.2, 'vis_box': [294, 200, 68, 225],
				'Occ_Coe': 0.0, 'Dif': False, 'Ign': 0, 'area': 15300, 'Tru': 0, 'Age': 'Adult'}, {'id': 10036186, 'category_id': 1, 'image_id': 107911, 'pose_id': 0, 'tracking_id': 100000,
				'bbox': [343, 214, 61, 196], 'Dis': 21.1, 'vis_box': [343, 214, 61, 196], 'Occ_Coe': 0.0, 'Dif': False, 'Ign': 0, 'area': 11956, 'Tru': 0, 'Age': 'Adult'},
				{'id': 10036187, 'category_id': 1, 'image_id': 107911, 'pose_id': 0, 'tracking_id': 100000, 'bbox': [255, 219, 49, 178], 'Dis': 23.3, 'vis_box': [255, 219, 49, 178],
				'Occ_Coe': 0.0, 'Dif': False, 'Ign': 0, 'area': 8722, 'Tru': 0, 'Age': 'Adult'}, {'id': 10036188, 'category_id': 1, 'image_id': 107911, 'pose_id': 0, 'tracking_id': 100000,
				'bbox': [491, 218, 59, 173], 'Dis': 25.0, 'vis_box': [491, 218, 59, 173], 'Occ_Coe': 0.0, 'Dif': False, 'Ign': 0, 'area': 10207, 'Tru': 0, 'Age': 'Adult'},
				{'id': 10036189, 'category_id': 1, 'image_id': 107911, 'pose_id': 0, 'tracking_id': 100000, 'bbox': [619, 220, 78, 163], 'Dis': 27.0, 'vis_box': [619, 220, 78, 163],
				'Occ_Coe': 0.0, 'Dif': False, 'Ign': 0, 'area': 12714, 'Tru': 0, 'Age': 'Adult'}]'''
				for gt_box in annos_GT:  # TODO: ***以标记框高度排序。
					cat = cocoGt.loadCats(ids=gt_box['category_id'])[0]  # cat={'name': 'pedestrian', 'id': 1}
					class_name = cat['name']
					if class_name not in ['Ped', 'ped', 'Pedestrian', 'pedestrian']:
						continue
					b_gt = gt_box['bbox']
					x1, y1, x2, y2 = b_gt[0], b_gt[1], b_gt[0] + b_gt[2], b_gt[1] + b_gt[3]
					bi = [max(left, x1), max(top, y1), min(right, x2), min(bottom, y2)]  # 用于计算检测框和真实标记框的交集面积
					iw = bi[2] - bi[0] + 1  # 计算交集宽
					ih = bi[3] - bi[1] + 1  # 计算交集高
					if iw > 0 and ih > 0:  # 如果相交，才开始计算交并比 compute overlap (IoU) = area of intersection / area of union
						ua = (right - left + 1) * (bottom - top + 1) + (x2 - x1 + 1) * (y2 - y1 + 1) - iw * ih  # 计算并集面积
						ov = iw * ih / ua  # 计算交并比
						if 0 < ov <= 1:
							if ov > ovmax:
								ovmax = ov
								gt_match_box = gt_box  # 用gt_match记录与当前检测框bb匹配的真实标记框gt_match = {'class_name': 'Bic', 'bbox': [1035, 229, 1156, 526],
						else:
							print('error overlap = %.1f%%' % ov * 100)
				if ovmax >= 0.5:
					if (gt_match_box['Dif'] or float(gt_match_box['Occ_Coe']) > cfg.Occ_threshold or float(gt_match_box['Dis']) >= cfg.Dis_threshold):
						continue
				if ovmax > 0.1:
					dis_gt = gt_match_box['Dis']
					b_gt = gt_match_box['bbox']
					w = b_gt[2]
					h = b_gt[3]
					x1, y1, x2, y2 = b_gt[0], b_gt[1], b_gt[0] + w, b_gt[1] + h
					w_edge = int(0.2 * w)
					h_edge = int(0.12 * h)
					'''annos_GT =[{'occluded': None, 'difficult': None, 'bbox': [453, 207, 30, 54], 'id': 1000007, 'category_id': 4, 'image_id': 1000043, 'pose_id': 5, 'tracking_id': 1000000, 'ignore': 1, 'area': 1620, 'truncated': False},
						{'occluded': None, 'difficult': None, 'bbox': [514, 233, 95, 40], 'id': 1000010, 'category_id': 4, 'image_id': 1000043, 'pose_id': 5, 'tracking_id': 1000001, 'ignore': 1, 'area': 3800, 'truncated': False}]
					 '''
					dt_match_box = -1
					if ovmax >= 0.5:
						ovmax2 = MINOVERLAP
						# pdb.set_trace()
						for dt_box in dt_boxes:
							class_name_dt = dt_box['class_name']
							if class_name_dt not in ['Ped', 'ped', 'Pedestrian', 'pedestrian']:
								continue
							b2 = dt_box['bbox']
							if abs(b2[0]-left+b2[1]-top+b2[2]-right+b2[3]-bottom) < 10:
								continue
							left2, top2, right2, bottom2 = b2[0], b2[1], b2[2], b2[3]
							bi = [max(left2, x1), max(top2, y1), min(right2, x2), min(bottom2, y2)]  # 用于计算检测框和真实标记框的交集面积
							iw = bi[2] - bi[0] + 1  # 计算交集宽
							ih = bi[3] - bi[1] + 1  # 计算交集高
							if iw > 0 and ih > 0:  # 如果相交，才开始计算交并比 compute overlap (IoU) = area of intersection / area of union
								ua = (right2 - left2 + 1) * (bottom2 - top2 + 1) + (x2 - x1 + 1) * (y2 - y1 + 1) - iw * ih  # 计算并集面积
								ov = iw * ih / ua  # 计算交并比
								if 0 < ov <= 1:
									if ov >= ovmax2:
										ovmax2 = ov
										dt_match_box = dt_box  # 用gt_match记录与当前检测框bb匹配的真实标记框gt_match = {'class_name': 'Bic', 'bbox': [1035, 229, 1156, 526],
								else:
									print('error overlap = %.1f%%' % ov * 100)

						if dt_match_box != -1:
							b2 = dt_match_box['bbox']
							left2, top2, right2, bottom2 = b2[0], b2[1], b2[2], b2[3]
							x1_slice, y1_slice, x2_slice, y2_slice = np.min([left, left2, x1]) - w_edge, np.min([top, top2, y1]) - h_edge, np.max([right, right2, x2]) + w_edge, np.max([bottom, bottom2,y2]) + h_edge
						else:
							x1_slice, y1_slice, x2_slice, y2_slice = min(left, x1) - w_edge, min(top, y1) - h_edge, max(right, x2) + w_edge, max(bottom, y2) + h_edge

						path_img_dt_FP_region_path = os.path.join(path_img_dt_FP_region_dir_TP, '%s_FP_TP%d_%d.png' % (image_id, index, dis_gt))
					else:
						x1_slice, y1_slice, x2_slice, y2_slice = min(left, x1) - w_edge, min(top, y1) - h_edge, max(right, x2) + w_edge, max(bottom, y2) + h_edge
						path_img_dt_FP_region_path = os.path.join(path_img_dt_FP_region_dir_FP, '%s_FP_FP%d_%d.png' % (image_id, index, dis_gt))

					region_FP = image_with_boxes.crop((x1_slice, y1_slice, x2_slice, y2_slice))  # 截取图片
					h_resize = 300
					scale_resize = h_resize / (y2_slice - y1_slice)
					w_resize = int((x2_slice - x1_slice) * scale_resize)
					region_FP = region_FP.resize((w_resize, h_resize))

					draw_FP = ImageDraw.Draw(region_FP)
					draw_FP.rectangle([int(scale_resize * (left - x1_slice)), int(scale_resize * (top - y1_slice)), int(scale_resize * (right - x1_slice)), int(scale_resize * (bottom - y1_slice))], outline=color_FP[0], width=width_line_slice)
					draw_FP.rectangle([int(scale_resize * (x1 - x1_slice)), int(scale_resize * (y1 - y1_slice)), int(scale_resize * (x2 - x1_slice)), int(scale_resize * (y2 - y1_slice))], outline=color_GT[0], width=width_line_slice+1)


					label = '%.1f%%' % (100 * ovmax)
					font_slice = ImageFont.truetype('arial.ttf', size=size_font_slice)
					label_size = draw_FP.textsize(label, font_slice)
					text_origin = np.array([int(0.5 * w_resize - 0.5 * label_size[0]), int(0.5 * h_resize - 2.5 * label_size[1])])
					draw_FP.rectangle([int(0.5 * w_resize - 0.5 * label_size[0]), int(0.5 * h_resize - 2.5 * label_size[1]+2), int(0.5 * w_resize + 0.5 * label_size[0]), int(0.5 * h_resize - 1.5 * label_size[1])], fill=(128, 128, 128))
					draw_FP.text(text_origin, label, fill=color_FP[1], font=font_slice)

					label = '%.1fm' % dis_gt
					label_size = draw_FP.textsize(label, font_slice)
					text_origin = np.array([int(0.5 * w_resize - 0.5 * label_size[0]), h_resize - label_size[1] - 2])
					draw_FP.rectangle([int(0.5 * w_resize - 0.5 * label_size[0]), h_resize - label_size[1], int(0.5 * w_resize + 0.5 * label_size[0]), h_resize - 2], fill=(128, 128, 128))
					draw_FP.text(text_origin, label, fill=color_GT[1], font=font_slice)

					if ovmax >= 0.5 and dt_match_box != -1: #TODO： 框出交并比更高的检测框TP
						draw_FP.rectangle([int(scale_resize * (left2 - x1_slice)), int(scale_resize * (top2 - y1_slice)), int(scale_resize * (right2 - x1_slice)), int(scale_resize * (bottom2 - y1_slice))], outline=color_TP[0], width=width_line_slice)
						label = '%.1f%%' % (100 * ovmax2)
						font_slice = ImageFont.truetype('arial.ttf', size=size_font_slice)
						label_size = draw_FP.textsize(label, font_slice)
						text_origin = np.array([int(0.5 * w_resize - 0.5 * label_size[0]), int(0.5 * h_resize - 0.5 * label_size[1])])
						draw_FP.rectangle([int(0.5 * w_resize - 0.5 * label_size[0]), int(0.5 * h_resize - 0.5 * label_size[1]+2), int(0.5 * w_resize + 0.5 * label_size[0]), int(0.5 * h_resize + 0.5 * label_size[1])], fill=(128, 128, 128))
						draw_FP.text(text_origin, label, fill=color_TP[1], font=font_slice)
				else:
					w = right-left
					h = bottom-top
					w_edge = int(0.2*w)
					h_edge = int(0.12*h)
					x1_slice, y1_slice, x2_slice, y2_slice = left - w_edge, top - h_edge, right + w_edge, bottom + h_edge
					region_FP = image_with_boxes.crop((x1_slice, y1_slice, x2_slice, y2_slice))  # 截取图片
					h_resize = 300
					scale_resize = h_resize / (y2_slice - y1_slice)
					w_resize = int((x2_slice - x1_slice) * scale_resize)
					region_FP = region_FP.resize((w_resize, h_resize))
					draw_FP = ImageDraw.Draw(region_FP)
					draw_FP.rectangle([int(scale_resize*w_edge), int(scale_resize*h_edge), int(scale_resize*(w_edge+w)), int(scale_resize*(h_edge+h))], outline=color_FP[0], width=width_line_slice)  # 淡红色单框：缩放到resized图上标记框

					Dis_dt = anno['Dis_dt']
					label = '%.1fm' % Dis_dt
					font_slice = ImageFont.truetype('arial.ttf', size=size_font_slice)
					label_size = draw_FP.textsize(label, font_slice)
					text_origin = np.array([int(0.5 * w_resize - 0.5 * label_size[0]), edge_top])
					draw_FP.rectangle([int(0.5 * w_resize - 0.5 * label_size[0]), edge_top+2, int(0.5 * w_resize + 0.5 * label_size[0]), int(edge_top + label_size[1])], fill=(128, 128, 128))
					draw_FP.text(text_origin, label, fill=color_FP[1], font=font_slice)

					path_img_dt_FP_region_path = os.path.join(path_img_dt_FP_region_dir_Bg, '%s_FP_Bg%d_%d.png' % (image_id, index, Dis_dt))

				region_FP.save(path_img_dt_FP_region_path)  # 保存图片Data20181220192501_020000_FP0_51
		# 		TODO:***切片显示FP。***切片显示FP。***切片显示FP。***切片显示FP。***切片显示FP。***切片显示FP。
		# TODO: ***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。***切片FP 检测结果。
        #
		# TODO: ***切片AER_bad 检测结果。***切片AER_bad 检测结果。***切片AER_bad 检测结果。***切片AER_bad 检测结果。
		# TODO: ***切片AER_bad 检测结果。***切片AER_bad 检测结果。***切片AER_bad 检测结果。***切片AER_bad 检测结果。
		if image_id in annos_AER_bad_Dis:
			annos_AER_bad = annos_AER_bad_Dis[image_id]
			for index, anno in enumerate(annos_AER_bad):
				b = anno['BB_dt']
				left, top, right, bottom = int(b[0]), int(b[1]), int(b[2]), int(b[3])
				# draw.rectangle([left-2, top-2, right+2, bottom+2], outline=color_AER_bad[0], width=size_font_highlight)  # 淡红色单框：缩放到resized图上标记框

				# TODO:***切片显示bad error rate of distance estimation。***切片显示bad error rate of distance estimation。
				# TODO:***切片显示bad error rate of distance estimation。***切片显示bad error rate of distance estimation。
				b_gt = anno['BB_gt']
				x1, y1, x2, y2 = int(b_gt[0]), int(b_gt[1]), int(b_gt[2]), int(b_gt[3])
				w = x2-x1
				h = y2-y1

				bi = [max(left, x1), max(top, y1), min(right, x2), min(bottom, y2)]  # 用于计算检测框和真实标记框的交集面积
				iw = bi[2] - bi[0] + 1  # 计算交集宽
				ih = bi[3] - bi[1] + 1  # 计算交集高
				ua = (right - left + 1) * (bottom - top + 1) + (x2 - x1 + 1) * (y2 - y1 + 1) - iw * ih  # 计算并集面积
				IoU = iw * ih / ua  # 计算交并比

				dis_gt = anno['Dis_gt']
				dis_dt = anno['Dis_dt']
				'''annos_GT =[{'occluded': None, 'difficult': None, 'bbox': [453, 207, 30, 54], 'id': 1000007, 'category_id': 4, 'image_id': 1000043, 'pose_id': 5, 'tracking_id': 1000000, 'ignore': 1, 'area': 1620, 'truncated': False},
					{'occluded': None, 'difficult': None, 'bbox': [514, 233, 95, 40], 'id': 1000010, 'category_id': 4, 'image_id': 1000043, 'pose_id': 5, 'tracking_id': 1000001, 'ignore': 1, 'area': 3800, 'truncated': False}]
				 '''
				w_edge = int(0.2 * w)
				h_edge = int(0.12 * h)
				x1_slice, y1_slice, x2_slice, y2_slice = min(left, x1) - w_edge, min(top, y1) - h_edge, max(right, x2) + w_edge, max(bottom, y2) + h_edge
				region_AER_bad = image_with_boxes.crop((x1_slice, y1_slice, x2_slice, y2_slice))  # 截取图片
				h_resize = 300
				scale_resize = h_resize / (y2_slice - y1_slice)
				w_resize = int((x2_slice-x1_slice)*scale_resize)
				region_AER_bad = region_AER_bad.resize((w_resize, h_resize))

				draw_AER_bad = ImageDraw.Draw(region_AER_bad)
				draw_AER_bad.rectangle([int(scale_resize*(left-x1_slice)), int(scale_resize*(top-y1_slice)), int(scale_resize*(right-x1_slice)), int(scale_resize*(bottom-y1_slice))], outline=color_TP[0], width=width_line_slice)  # 淡红色单框：缩放到resized图上标记框
				draw_AER_bad.rectangle([int(scale_resize*(x1-x1_slice)), int(scale_resize*(y1-y1_slice)), int(scale_resize*(x2-x1_slice)), int(scale_resize*(y2-y1_slice))], outline=color_GT[0], width=width_line_slice+1)  # 淡红色单框：缩放到resized图上标记框

				# pdb.set_trace()
				ER_bad = 100*(dis_dt-dis_gt)/dis_gt
				label = '%.1f%%' % ER_bad
				# label = '%.1fm' % dis_dt
				font_slice = ImageFont.truetype('arial.ttf', size=size_font_slice)
				label_size = draw_AER_bad.textsize(label, font_slice)
				text_origin = np.array([int(0.5*w_resize - 0.5*label_size[0]), edge_top])
				draw_AER_bad.rectangle([int(0.5*w_resize - 0.5*label_size[0]), edge_top+2, int(0.5*w_resize + 0.5*label_size[0]), int(edge_top+label_size[1])], fill=(100,100,100))
				draw_AER_bad.text(text_origin, label, fill=color_TP[1], font=font_slice)

				label = '%d%%' % (100*IoU)
				font_slice = ImageFont.truetype('arial.ttf', size=size_font_slice)
				label_size = draw_AER_bad.textsize(label, font_slice)
				text_origin = np.array([int(0.5*w_resize - 0.5*label_size[0]), int(0.5*h_resize-0.5*label_size[1])])
				draw_AER_bad.rectangle([int(0.5 * w_resize - 0.5 * label_size[0]), int(0.5*h_resize-0.5*label_size[1]+2), int(0.5 * w_resize + 0.5 * label_size[0]), int(0.5*h_resize+0.5*label_size[1])],fill=(128, 128, 128))
				draw_AER_bad.text(text_origin, label, fill=color_TP[1], font=font_slice)

				label = '%.1fm' % dis_gt
				label_size = draw_AER_bad.textsize(label, font_slice)
				text_origin = np.array([int(0.5*w_resize - 0.5*label_size[0]), h_resize-label_size[1]-2])
				draw_AER_bad.rectangle([int(0.5 * w_resize - 0.5 * label_size[0]), h_resize-label_size[1], int(0.5 * w_resize + 0.5 * label_size[0]), h_resize-2],fill=(128, 128, 128))
				draw_AER_bad.text(text_origin, label, fill=color_GT[1], font=font_slice)

				if dis_dt < dis_gt:
					path_img_dt_AER_bad_region = os.path.join(path_img_dt_AER_bad_region_dir_minus, '%s_AER_bad_minus%d_%d.png' % (image_id, index, dis_gt))
				else:
					path_img_dt_AER_bad_region = os.path.join(path_img_dt_AER_bad_region_dir_plus, '%s_AER_bad_plus%d_%d.png' % (image_id, index, dis_gt))

				region_AER_bad.save(path_img_dt_AER_bad_region)  # 保存图片


				# TODO:***切片显示bad error rate of distance estimation。***切片显示bad error rate of distance estimation。

		# TODO: ***切片AER_bad 检测结果。***切片AER_bad 检测结果。***切片AER_bad 检测结果。***切片AER_bad 检测结果。
        #
		# TODO: ***切片miss_Ped(FN) 检测结果。 ***切片miss_Ped(FN) 检测结果。 ***切片miss_Ped(FN) 检测结果。 ***切片miss_Ped(FN) 检测结果。
		# TODO: ***切片miss_Ped(FN) 检测结果。 ***切片miss_Ped(FN) 检测结果。 ***切片miss_Ped(FN) 检测结果。 ***切片miss_Ped(FN) 检测结果。
		if image_id in annos_miss_Ped_Dis:
			annos_FN = annos_miss_Ped_Dis[image_id]
			for anno in annos_FN:
				b_gt = anno['bbox']
				dis_gt = anno['Dis']
				x1, y1, x2, y2 = int(b_gt[0]), int(b_gt[1]), int(b_gt[2]), int(b_gt[3])
				w = x2 - x1
				h = y2 - y1
				# draw.rectangle([x1-2, y1-2, x2+2, y2+2], outline=color_FN[0], width=size_font_highlight)  # 淡红色单框：缩放到resized图上标记框
				# dt_boxes = [{'class_name': 'Ped', 'confidence': '0.999998', 'file_id': 'Data20181219200348_020000', 'bbox': [448.0, 202.0, 576.0, 517.0], 'Dis': 13.91},
				# {'class_name': 'Ped', 'confidence': '0.999998', 'file_id': 'Data20181219200348_020000', 'bbox': [448.0, 202.0, 576.0, 517.0], 'Dis': 13.91},
				# {'class_name': 'Ped', 'confidence': '0.999997', 'file_id': 'Data20181219200348_020000', 'bbox': [32.0, 179.0, 160.0, 539.0], 'Dis': 12.74},
				# {'class_name': 'Ped', 'confidence': '0.999997', 'file_id': 'Data20181219200348_020000', 'bbox': [32.0, 179.0, 160.0, 539.0], 'Dis': 12.74},
				# {'class_name': 'Ped', 'confidence': '0.999964', 'file_id': 'Data20181219200348_020000', 'bbox': [208.0, 202.0, 336.0, 494.0], 'Dis': 15.45},
				# {'class_name': 'Ped', 'confidence': '0.999964', 'file_id': 'Data20181219200348_020000', 'bbox': [208.0, 202.0, 336.0, 494.0], 'Dis': 15.45}]
				ovmax = 0
				for index, dt_box in enumerate(dt_boxes):
					class_name = dt_box['class_name']
					if class_name not in ['Ped', 'ped', 'Pedestrian', 'pedestrian']:
						continue
					if float(dt_box['confidence']) < score_threshold:
						continue
					b = dt_box['bbox']
					left, top, right, bottom = b[0], b[1], b[2], b[3]
					bi = [max(left, x1), max(top, y1), min(right, x2), min(bottom, y2)]  # 用于计算检测框和真实标记框的交集面积
					iw = bi[2] - bi[0] + 1  # 计算交集宽
					ih = bi[3] - bi[1] + 1  # 计算交集高
					if iw > 0 and ih > 0:  # 如果相交，才开始计算交并比 compute overlap (IoU) = area of intersection / area of union
						ua = (right - left + 1) * (bottom - top + 1) + (x2 - x1 + 1) * (y2 - y1 + 1) - iw * ih  # 计算并集面积
						ov = iw * ih / ua  # 计算交并比
						if 0 < ov <= 1:
							if ov > ovmax:
								ovmax = ov
								dt_match_box = dt_box  # 用gt_match记录与当前检测框bb匹配的真实标记框gt_match = {'class_name': 'Bic', 'bbox': [1035, 229, 1156, 526],
						else:
							print('error overlap = %.1f%%' % ov * 100)

				# TODO:***切片显示FN。***切片显示FN。***切片显示FN。***切片显示FN。***切片显示FN。***切片显示FN。
				# TODO:***切片显示FN。***切片显示FN。***切片显示FN。***切片显示FN。***切片显示FN。***切片显示FN。
				if ovmax > 0.1:
					dis_dt = dt_match_box['Dis']
					b = dt_match_box['bbox']
					left, top, right, bottom = b[0], b[1], b[2], b[3]
					'''annos_GT =[{'occluded': None, 'difficult': None, 'bbox': [453, 207, 30, 54], 'id': 1000007, 'category_id': 4, 'image_id': 1000043, 'pose_id': 5, 'tracking_id': 1000000, 'ignore': 1, 'area': 1620, 'truncated': False},
						{'occluded': None, 'difficult': None, 'bbox': [514, 233, 95, 40], 'id': 1000010, 'category_id': 4, 'image_id': 1000043, 'pose_id': 5, 'tracking_id': 1000001, 'ignore': 1, 'area': 3800, 'truncated': False}]
					 '''
					w_edge = int(0.2 * w)
					h_edge = int(0.12 * h)
					x1_slice, y1_slice, x2_slice, y2_slice = min(left, x1) - w_edge, min(top, y1) - h_edge, max(right, x2) + w_edge, max(bottom, y2) + h_edge
					region_FN = image_with_boxes.crop((x1_slice, y1_slice, x2_slice, y2_slice))  # 截取图片
					h_resize = 300
					scale_resize = h_resize / (y2_slice - y1_slice)
					w_resize = int((x2_slice - x1_slice) * scale_resize)

					region_FN = region_FN.resize((w_resize, h_resize))

					draw_FN = ImageDraw.Draw(region_FN)
					draw_FN.rectangle([int(scale_resize * (left - x1_slice)), int(scale_resize * (top - y1_slice)), int(scale_resize * (right - x1_slice)), int(scale_resize * (bottom - y1_slice))],
									  outline=color_FP[0], width=width_line_slice)
					draw_FN.rectangle([int(scale_resize * (x1 - x1_slice)), int(scale_resize * (y1 - y1_slice)), int(scale_resize * (x2 - x1_slice)), int(scale_resize * (y2 - y1_slice))],
									  outline=color_GT[0], width=width_line_slice+1)

					# label = '%.1fm' % dis_dt
					# font_slice = ImageFont.truetype('arial.ttf', size=size_font_slice)
					# label_size = draw_FN.textsize(label, font_slice)
					# text_origin = np.array([int(0.5 * w_resize - 0.5 * label_size[0]), 2])
					# draw_FN.text(text_origin, label, fill=color_FP[1], font=font_slice)

					label = '%.1f%%' % (100.0 * ovmax)
					font_slice = ImageFont.truetype('arial.ttf', size=size_font_slice)
					label_size = draw_FN.textsize(label, font_slice)
					text_origin = np.array([int(0.5 * w_resize - 0.5 * label_size[0]), int(0.5 * h_resize - 0.5 * label_size[1])])
					draw_FN.rectangle([int(0.5 * w_resize - 0.5 * label_size[0]), int(0.5 * h_resize - 0.5 * label_size[1]+2), int(0.5 * w_resize + 0.5 * label_size[0]), int(0.5 * h_resize + 0.5 * label_size[1])], fill=(128, 128, 128))
					draw_FN.text(text_origin, label, fill=color_FP[1], font=font_slice)

					label = '%.1fm' % dis_gt
					label_size = draw_FN.textsize(label, font_slice)
					text_origin = np.array([int(0.5 * w_resize - 0.5 * label_size[0]), h_resize - label_size[1] - 2])
					draw_FN.rectangle([int(0.5 * w_resize - 0.5 * label_size[0]), h_resize - label_size[1], int(0.5 * w_resize + 0.5 * label_size[0]), h_resize - 2], fill=(128, 128, 128))
					draw_FN.text(text_origin, label, fill=color_GT[1], font=font_slice)

					path_img_dt_FN_region_path = os.path.join(path_img_dt_FN_region_dir_FP, '%s_FN_FP%d_%d.png' % (image_id, index, dis_gt))

				else:
					w_edge = int(0.2 * w)
					h_edge = int(0.12 * h)
					x1_slice, y1_slice, x2_slice, y2_slice = x1 - w_edge, y1 - h_edge, x2 + w_edge, y2 + h_edge
					region_FN = image_with_boxes.crop((x1_slice, y1_slice, x2_slice, y2_slice))  # 截取图片
					h_resize = 300
					scale_resize = h_resize / (y2_slice - y1_slice)
					w_resize = int((x2_slice - x1_slice) * scale_resize)
					region_FN = region_FN.resize((w_resize, h_resize))
					draw_FN = ImageDraw.Draw(region_FN)
					draw_FN.rectangle([int(scale_resize * w_edge), int(scale_resize * h_edge), int(scale_resize * (w_edge + w)), int(scale_resize * (h_edge + h))], outline=color_GT[0], width=width_line_slice+1)

					label = '%.1fm' % dis_gt
					font_slice = ImageFont.truetype('arial.ttf', size=size_font_slice)
					label_size = draw_FN.textsize(label, font_slice)
					text_origin = np.array([int(0.5 * w_resize - 0.5 * label_size[0]), int(h_resize-label_size[1]-2)])
					draw_FN.rectangle([int(0.5 * w_resize - 0.5 * label_size[0]), h_resize - label_size[1], int(0.5 * w_resize + 0.5 * label_size[0]), h_resize - 2], fill=(128, 128, 128))
					draw_FN.text(text_origin, label, fill=color_GT[1], font=font_slice)
					path_img_dt_FN_region_path = os.path.join(path_img_dt_FN_region_dir_Bg, '%s_FN_Bg%d_%d.png' % (image_id, index, dis_gt))

				region_FN.save(path_img_dt_FN_region_path)  # 保存图片
				# TODO:***切片显示FN。***切片显示FN。***切片显示FN。***切片显示FN。***切片显示FN。***切片显示FN。
		# TODO: ***切片miss_Ped(FN) 检测结果。 ***切片miss_Ped(FN) 检测结果。 ***切片miss_Ped(FN) 检测结果。 ***切片miss_Ped(FN) 检测结果。

		if image_id in annos_FP_Dis:
		# TODO: ***突出FP 检测结果。 ***突出FP 检测结果。 ***突出FP 检测结果。 ***突出FP 检测结果。 ***突出FP 检测结果。 ***突出FP 检测结果。 ***突出FP 检测结果。
		# TODO: ***突出FP 检测结果。 ***突出FP 检测结果。 ***突出FP 检测结果。 ***突出FP 检测结果。 ***突出FP 检测结果。 ***突出FP 检测结果。 ***突出FP 检测结果。
			annos_FP = annos_FP_Dis[image_id]
			for index, anno in enumerate(annos_FP):
				b = anno['BB_dt']
				left, top, right, bottom = int(b[0]), int(b[1]), int(b[2]), int(b[3])
				draw.rectangle([left-2, top-2, right+2, bottom+2], outline=color_FP[0], width=size_font_highlight)  # 淡红色单框：缩放到resized图上标记框

		if image_id in annos_AER_bad_Dis:
		# TODO: ***突出AER_bad 检测结果。 ***突出AER_bad 检测结果。 ***突出AER_bad 检测结果。 ***突出AER_bad 检测结果。 ***突出AER_bad 检测结果。 ***突出AER_bad 检测结果。 ***突出AER_bad 检测结果。
		# TODO: ***突出AER_bad 检测结果。 ***突出AER_bad 检测结果。 ***突出AER_bad 检测结果。 ***突出AER_bad 检测结果。 ***突出AER_bad 检测结果。 ***突出AER_bad 检测结果。 ***突出AER_bad 检测结果。
			annos_AER_bad = annos_AER_bad_Dis[image_id]
			for index, anno in enumerate(annos_AER_bad):
				b = anno['BB_dt']
				left, top, right, bottom = int(b[0]), int(b[1]), int(b[2]), int(b[3])
				draw.rectangle([left-2, top-2, right+2, bottom+2], outline=color_Ped[0], width=size_font_highlight)  # 淡红色单框：缩放到resized图上标记框

		if image_id in annos_miss_Ped_Dis:
		# TODO: ***突出miss_Ped 检测结果。 ***突出miss_Ped 检测结果。 ***突出miss_Ped 检测结果。 ***突出miss_Ped 检测结果。 ***突出miss_Ped 检测结果。
		# TODO: ***突出miss_Ped 检测结果。 ***突出miss_Ped 检测结果。 ***突出miss_Ped 检测结果。 ***突出miss_Ped 检测结果。 ***突出miss_Ped 检测结果。

			annos_FN = annos_miss_Ped_Dis[image_id]
			for anno in annos_FN:
				b = anno['bbox']
				left, top, right, bottom = int(b[0]), int(b[1]), int(b[2]), int(b[3])
				draw.rectangle([left-2, top-2, right+2, bottom+2], outline=color_Ped[0], width=size_font_highlight)  # 淡红色单框：缩放到resized图上标记框

		Doted_text = []
		# TODO:***显示检测结果。***显示检测结果。***显示检测结果。***显示检测结果。***显示检测结果。***显示检测结果。***显示检测结果。***显示检测结果。
		h_boxes = np.array([float(b['bbox'][3] - b['bbox'][1]) for b in dt_boxes if float(b['confidence']) >= score_threshold])  # 真实标记框信息：bbgtF = [466.0, 313.0, 518.0, 433.0, 38.0]
		h_box_index = np.argsort(-h_boxes, axis=0)
		h_box_index = h_box_index.tolist()
		# dt_boxes = [{'class_name': 'Ped', 'confidence': '0.999998', 'file_id': 'Data20181219200348_020000', 'bbox': [448.0, 202.0, 576.0, 517.0], 'Dis': 13.91},...]
		for index in h_box_index:
			DT = dt_boxes[index]
			b = DT['bbox']
			left, top, right, bottom = int(b[0]), int(b[1]), int(b[2]), int(b[3])
			h_box = bottom - top
			class_name = DT['class_name']
			conf = 100 * float(DT['confidence'])
			# label = '%s%d%%' % (class_name, conf)
			label = 'p%d%%' % (conf)

			distance = round(float(DT['Dis']), 1)
			label = label+'d{}'.format(distance)

			color_cls = colors['DT']
			draw.rectangle([left, top, right, bottom], outline=color_cls[0], width=width_line_image)  # 淡红色单框：缩放到resized图上标记框
			# TODO: ***  只显示预测框及预测概率。***  只显示预测框及预测概率。***  只显示预测框及预测概率。
			# TODO: ***  只显示预测框及预测概率。***  只显示预测框及预测概率。***  只显示预测框及预测概率。
			# font_predict = ImageFont.truetype('arial.ttf', size=15)
			# label = 'P%.0f%%' % (conf)
			# label_size = draw.textsize(label, font_predict)
			# text_origin = np.array([left+2, bottom - label_size[1]-2])
			# # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 255, 255), width=width_line_image)
			# draw.text(text_origin, label, fill=color_cls[1], font=font_predict)
			# TODO: ***  只显示预测框及预测概率。***  只显示预测框及预测概率。***  只显示预测框及预测概率。

			# TODO: *** 有序显示标记。*** 有序显示标记。*** 有序显示标记。*** 有序显示标记。*** 有序显示标记。*** 有序显示标记。*** 有序显示标记。
			draw, Doted_text = draw_tags_orderly(draw, label, Doted_text, color_cls, left, right, top, bottom, h_box, edge_kept=edge_kept)
		# TODO:***检测结果。***检测结果。***检测结果。***检测结果。***检测结果。***检测结果。***检测结果。***检测结果。***检测结果。***检测结果。

		# TODO:***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。
		# TODO:***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。
		h_box = np.array([anno['bbox'][3] for anno in annos_GT])
		h_box_index = np.argsort(-h_box, axis=0)# TODO: ***对标记索引按标记框高度进行排序。***对标记索引按标记框高度进行排序。***对标记索引按标记框高度进行排序。
		h_box_index = h_box_index.tolist()
		for anno_index in h_box_index:  # TODO: ***以标记框高度排序。
			anno = annos_GT[anno_index]
			cat = cocoGt.loadCats(ids=anno['category_id'])[0]  # cat={'name': 'pedestrian', 'id': 1}
			class_name = cat['name']
			b = anno['bbox']
			# TODO: ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。
			# TODO: ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。
			x1, y1, x2, y2 = b[0] + 1, b[1] + 1, b[0] + b[2] + 1, b[1] + b[3] + 1
			if anno['Dif']:
				Dif = 'Diff'
			else:
				Dif = 'Easy'
			Age = anno['Age']
			ID = str(anno['tracking_id'])
			'''anno={'id': 10000004, 'category_id': 5, 'image_id': 100000, 'pose_id': 0, 'tracking_id': 0, 'bbox': [100, 100, 1160, 600], 'Dis': 100.0, 'vis_box': [100, 100, 1160, 600],
			 'Occ_Coe': 0.0, 'Dif': False, 'Ign': 1, 'area': 696000, 'Tru': 0, 'Age': 'Adult'}'''

			vis_box = anno['vis_box']
			Vx1, Vy1, Vx2, Vy2 = vis_box[0] + 1, vis_box[1] + 1, vis_box[0] + vis_box[2] + 1, vis_box[1] + vis_box[3] + 1
			Dis = anno['Dis'] # path,x1,y1,x2,y2,Dis,Vx1,Vy1,Vx2,Vy2,Age,Dif,Sen,Wea,ID,cls
			annos_GT_txt.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(img_dt_results_path, x1, y1, x2, y2, Dis, Vx1, Vy1, Vx2, Vy2, Age, Dif, Sce, Wea, ID, class_name))
			# TODO: ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。 ***将标记写入记事本。

			# TODO: ***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。
			left, top, right, bottom = int(b[0]), int(b[1]), int(b[0]+b[2]), int(b[1]+b[3])
			area_bbox = b[2] * b[3]
			h_box = int(b[3])
			label = class_name
			# label = label
			vis_box_exist = False
			# if class_name in ['Ped', 'ped', 'Pedestrian', 'pedestrian', 'Bic', 'Mot', 'Peo']:
			if class_name in ['Ped', 'ped', 'Pedestrian', 'pedestrian', 'Bic', 'Mot']:
				if anno['Age'] == 'Child':
					label = label + ':Child'

				distance = float(anno['Dis'])
				if class_name in ['Ped', 'ped', 'Pedestrian', 'pedestrian']:
					# height_from_distance = round(Hf / distance, 1)
					# height_from_distance_abs_error = abs(height_from_distance - h_box)
					# if height_from_distance_abs_error > 40 + 1000 / distance:
					# 	distance_from_height = int(Hf / h_box)  # TODO: 将像素高度直接估算为距离。
					# 	label = label + '\nd*%d' % distance_from_height
					if distance <= 0 or distance == 100:
						distance_from_height = round(Hf / h_box, 1)
						# label = label + ':dh*%.1f' % distance_from_height
						label = label + ':d*%.1f' % distance_from_height
					else:
						label = label + ':d*%.1f' % distance
				else:
					label = label + ':d*%.1f' % distance

				vis_box = anno['vis_box']
				area_vis_box = vis_box[2] * vis_box[3]
				if area_vis_box != area_bbox:
					vis_left, vis_top, vis_right, vis_bottom = int(vis_box[0]), int(vis_box[1]), int((vis_box[0] + vis_box[2])), int((vis_box[1] + vis_box[3]))
					occ_coef = int(100 * (1 - area_vis_box / area_bbox))
					label = label + '\nOcc:{}%'.format(occ_coef)
					vis_box_exist = True

				if anno['Dif']:
					label = label + '\nDifficult'
			color_cls = colors[class_name]
			draw.rectangle([left, top, right, bottom], outline=color_cls[0], width=width_line_image)  # 淡红色单框：缩放到resized图上标记框

			draw, Doted_text = draw_tags_orderly(draw, label, Doted_text, color_cls, left, right, top, bottom, h_box, edge_kept=edge_kept)
			# if vis_box_exist:
			# 	draw.rectangle([vis_left + 2, vis_top + 2, vis_right - 2, vis_bottom - 2], width=width_line_image, outline='#ffffff')  # 淡红色单框：缩放到resized图上标记框'''
		# TODO:***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。***显示GT标记。
		image_with_boxes.save(img_dt_results_path)
	annos_GT_txt.close()
#TODO:*** 图像显示检测结果.***图像显示检测结果.***图像显示检测结果.***图像显示检测结果.***图像显示检测结果.***图像显示检测结果.***图像显示检测结果.

if __name__ == '__main__':  # 在此__name__如何得来？？？
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = ' '  # 指定无GPU参与运算。
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--ignore', nargs='+', type=str, help='ignore a list of classes.')
	args = parser.parse_args()
	'''
		0,0 ------> x (width)
		 |
		 |  (Left,Top)
		 |      *_________
		 |      |         |
				|         |
		 y      |_________|
	  (height)            *
					(Right,Bottom)
	'''

	# if there are no classes to ignore then replace None by empty list 如果没有要忽略的类，那么用空列表替换None
	if args.ignore is None:
		args.ignore = ['Ign', 'ignore', 'bicycledriver', 'Bic', 'Cyc', 'motorbikedriver', 'Mot', 'Sed']
	'''一般来说，分数阈值越高，FPPI 会越低，而Miss rate会越高。那么当评估一个识别方法时，通过设置不同的分数阈值，可以得到一组（Miss Rate， FPPI）值，从而可以画出MR-FPPI曲线。如上图，可以按log-log scale 画。 '''

	if evalute_new == 0:
		# TODO: ***计算MissRate_FPPI曲线。 ***计算MissRate_FPPI曲线。 ***计算MissRate_FPPI曲线。 ***计算MissRate_FPPI曲线。 ***计算MissRate_FPPI曲线。
		# TODO: ***计算MissRate_FPPI曲线。 ***计算MissRate_FPPI曲线。 ***计算MissRate_FPPI曲线。 ***计算MissRate_FPPI曲线。 ***计算MissRate_FPPI曲线。
		cfg.Occ_threshold = 0.35
		cfg.Dis_threshold = 80
		if imageset == 'val':
			cfg.annos_file = cfg.val_file
			detectResults_Path = os.path.join(Detection_results_dir, 'DtResults_NIRPed_%s.json' % (imageset))
		elif imageset == 'test':
			detectResults_Path = os.path.join(Detection_results_dir, 'DtResults_NIRPed_%s.json' % (imageset))
			cfg.annos_file = cfg.test_file

		if not os.path.exists(detectResults_Path): # print('The detectResults_Path is not exist:{}'.format(detectResults_Path))
			messages = 'The detectResults_Path is not exist:{}'.format(detectResults_Path)
			error(messages)
		print('Start to evaluate NIRPed_{}...............................................'.format(imageset))
		MR_FPPI_dis_max_dic, MR_FPPI_dis_seg_dic, PR_dis_max_dic, PR_dis_seg_dic, AE_dtp_seg_dic, AE_dtp_max_dic, \
		AE_DE_seg_dic, AE_DE_max_dic, AER_dtp_seg_dic, AER_dtp_max_dic, AER_DE_seg_dic, AER_DE_max_dic, \
		annos_AER_bad, annos_FN, annos_FP = MR_FPPI_PR_Calculate(detectResults_Path)

		#TODO: ***计算MissRate_FPPI曲线。 ***计算MissRate_FPPI曲线。 ***计算MissRate_FPPI曲线。 ***计算MissRate_FPPI曲线。 ***计算MissRate_FPPI曲线。
		# TODO: ***MissRate_FPPI_dis_max_dic、score_threshold_dis_max_dic。
		results_Path = os.path.join(Detection_results_dir, 'MR_FPPI_dis_max_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(MR_FPPI_dis_max_dic, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'MR_FPPI_dis_seg_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(MR_FPPI_dis_seg_dic, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'PR_dis_max_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(PR_dis_max_dic, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'PR_dis_seg_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(PR_dis_seg_dic, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'AE_dtp_seg_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(AE_dtp_seg_dic, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'AE_dtp_max_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(AE_dtp_max_dic, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'AE_DE_seg_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(AE_DE_seg_dic, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'AE_DE_max_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(AE_DE_max_dic, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'AER_dtp_seg_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(AER_dtp_seg_dic, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'AER_dtp_max_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(AER_dtp_max_dic, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'AER_DE_seg_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(AER_DE_seg_dic, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'AER_DE_max_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(AER_DE_max_dic, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'annos_AER_bad_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(annos_AER_bad, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'annos_FN_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(annos_FN, file_obj)
		file_obj.close()

		results_Path = os.path.join(Detection_results_dir, 'annos_FP_results_%s.json' % (imageset))
		file_obj = open(results_Path, 'w')
		json.dump(annos_FP, file_obj)
		file_obj.close()

	elif evalute_new == 1:
		# TODO: ***图像显示不好的检测结果（丢失、误报和大距离误差）。 ***图像显示不好的检测结果（丢失、误报和大距离误差）。 ***图像显示不好的检测结果（丢失、误报和大距离误差）。
		print('Start to show detection results of NIRPed using images.....................................')
		cfg.im_rows_show = 720  # 近红外图像的短边360->352->320->256
		cfg.im_cols_show = 1280  # 近红外图像的短边1280->640
		class_name = 'Ped'
		cfg.Occ_threshold = 0.35
		cfg.Dis_threshold = 80
		if imageset == 'val':
			cfg.annos_file = cfg.val_file
			detectResults_Path = os.path.join(Detection_results_dir, 'DtResults_NIRPed_%s.json' % (imageset))
		elif imageset == 'test':
			cfg.annos_file = cfg.test_file
			detectResults_Path = os.path.join(Detection_results_dir, 'DtResults_NIRPed_%s.json' % (imageset))

		if not os.path.exists(detectResults_Path): # print('The detectResults_Path is not exist:{}'.format(detectResults_Path))
			messages = 'The detectResults_Path is not exist:{}'.format(detectResults_Path)
			print(messages)
			# error(messages)

		dt_data_plot = json.load(open(detectResults_Path))  # 加载json文件中的数据

		if os.path.exists(cfg.annos_file):
			file_obj = open(cfg.annos_file, 'r')
			imgs_data = json.load(file_obj)
			file_obj.close()
		else:
			print('Notion: can not find training dataset!')
			sys.exit(0)  # 干净利落地退出系统

		# pdb.set_trace()
		annos_FP_results_Path = os.path.join(Detection_results_dir, 'annos_FP_results_%s.json' % (imageset))
		annos_FP_results = json.load(open(annos_FP_results_Path))
		imgs_dt_show_dir = os.path.join(cfg.model_dir, "dt_results_%s_Visualization" % (imageset))
		if not os.path.exists(imgs_dt_show_dir):
			os.makedirs(imgs_dt_show_dir)

		annos_AER_bad_results_Path = os.path.join(Detection_results_dir, 'annos_AER_bad_results_%s.json' % (imageset))
		annos_AER_bad_results = json.load(open(annos_AER_bad_results_Path))
		annos_miss_Ped_results_Path = os.path.join(Detection_results_dir, 'annos_FN_results_%s.json' % (imageset))
		annos_miss_Ped_results = json.load(open(annos_miss_Ped_results_Path))
		score_threshold = 0.5

		visulize_FP_FN_bad_ER(imgs_data, dt_data_plot, imgs_dt_show_dir, annos_FP_results, annos_AER_bad_results, annos_miss_Ped_results, score_threshold)
		# TODO: ***图像显示不好的检测结果（丢失、误报和大距离误差）。 ***图像显示不好的检测结果（丢失、误报和大距离误差）。 ***图像显示不好的检测结果（丢失、误报和大距离误差）。