import pdb
import os, sys
import numpy as np
from keras_frcnn.coco import COCO
import json
np.set_printoptions(precision=6, threshold=np.inf, edgeitems=10, linewidth=260, suppress=True)
from keras_frcnn import config
cfg = config.Config()  # 实例化config.py文件中的类Config，存储到变量cfg中

def get_data(cfg):
	if os.path.exists(cfg.train_anno):
		file_obj = open(cfg.train_anno, 'r')
		imgs_data = json.load(file_obj)
		file_obj.close()
	else:
		print('Notion: can not find training dataset!')
		sys.exit(0)  # 干净利落地退出系统

	if os.path.exists(cfg.training_loss_file):
		file_obj = open(cfg.training_loss_file, 'r')
		train_loss = json.load(file_obj)
		file_obj.close()
	else:
		train_loss = []

	cocoGt = COCO(cfg.train_anno)
	images = imgs_data['images']
	imageset = 'train'

	all_class_mapping = {cls['name']: cls['id'] - 1 for cls in imgs_data['categories']}
	all_class_mapping_reverse = {v: k for k, v in all_class_mapping.items()}
	classes_count = {cls['name']: 0 for cls in imgs_data['categories']}
	found_bg = False
	imgs_dataset = []

	for img in images:
		anno_ids = cocoGt.getAnnIds(imgIds=img['id'])  # anno_id=1037542
		annos_img = cocoGt.loadAnns(ids=anno_ids)
		bboxes = []
		for anno in annos_img:
			box = anno['bbox']
			category_id = anno['category_id']
			class_name = all_class_mapping_reverse[category_id - 1]
			classes_count[class_name] += 1
			distance = float(anno['Dis'])
			if distance <= 0 or distance == 100:
				distance = cfg.Dis_max
			vis_box = anno['vis_box']
			aera_box = float(box[2]) * float(box[3])
			vis_aera = float(vis_box[2]) * float(vis_box[3])
			Occlusion_coefficient =1 - vis_aera / aera_box
			if 0 < Occlusion_coefficient < 1:
				Occlusion_coefficient = round(Occlusion_coefficient, 2)
			else:
				#print('Wrong visable box yeild to Occlusion_coefficient = {}, and change it to Occlusion_coefficient=0'.format(Occlusion_coefficient))
				Occlusion_coefficient = 0

			bboxes.append({'class': class_name,  'x1': box[0], 'x2': box[0]+box[2], 'y1': box[1], 'y2': box[1]+box[3], 'Dis': distance,
					   'Occ_Coe': Occlusion_coefficient, 'Dif': anno['Dif'], "area": anno['area'], 'Age': 'Adult'})


			if class_name in ['bg', 'Bg', 'background', 'Background'] and not found_bg:  # 如果class_name恒等于'bg' ,且没有寻找到背景类
				print('Found class name with special name bg. Will be treated as a'  # 找到具有特殊名称bg的类名。
					  ' background region (this is usually for hard negative mining).')  # 将被视为背景区域（这通常用于硬负例挖据）
				found_bg = True  # 将寻找到背景的逻辑关键字置True
				all_class_mapping[class_name] = len(all_class_mapping)  # 将网络输出类节点映射的字典class_mapping增加名为class_name的键，且键值取为字典的长度

		img_name = os.path.basename(img['file_name']) #Data20200710195000_082906N850F12

		# img_path = os.path.join('E:\\Datasets\\NIRPed2021\\NIRPed\\images\\{}\\{}'.format(imageset, img_name))
		img_path = os.path.join(cfg.train_img_dir, img_name)

		img_data = {'filepath': img_path, 'height': img['height'], 'width': img['width'], 'daytime': img['daytime'], 'imageset': imageset}
		img_data['bboxes'] = bboxes
		imgs_dataset.append(img_data)

	if 'bg' not in classes_count:  # 如果键名'bg'不在字典classes_count中
		classes_count['bg'] = 0  # 在字典classes_count中增加一名为'bg'的键，且将键值置为0
		all_class_mapping['bg'] = len(all_class_mapping)  # 在字典class_mapping中增加一名为'bg'的键，且将键值置为原字典长度
		all_class_mapping_reverse[len(all_class_mapping) - 1] = 'bg'  # 在字典class_mapping中增加一名为'bg'的键，且将键值置为原字典长度

	class_mapping = {all_class_mapping_reverse[0]: 0, all_class_mapping_reverse[len(all_class_mapping_reverse) - 1]: 1}  # 将字典class_mapping赋给类cfg的变量class_mapping

	return imgs_dataset, train_loss, classes_count, class_mapping