import pickle
import numpy as np
import re, os, sys, glob
import struct

def error(msg):
	print(msg)
	sys.exit(0) #干净利落地退出系统
def exchange_hex(data):
	'''
	16进制高低位互换函数
	struct.pack(fmt, v1, v2, …)用于将Python的值根据格式符，转换为字符串
	参数fmt是格式字符串，H表示unsigned short类型，占用2个字节，<表示小字节序、低字节序
	v1, v2, …表示要转换的python值
	'''
	i = int(data, 16)
	return struct.pack('<H', i).hex()
np.set_printoptions(precision=3, threshold=np.inf, edgeitems=10, linewidth=260, suppress=True)
def color(value):
	digit = list(map(str, range(10))) + list("ABCDEF")
	if isinstance(value, tuple):
		string = '#'
		for i in value:
			a1 = i // 16
			a2 = i % 16
			string += digit[a1] + digit[a2]
		return string
	elif isinstance(value, str):
		a1 = digit.index(value[1]) * 16 + digit.index(value[2])
		a2 = digit.index(value[3]) * 16 + digit.index(value[4])
		a3 = digit.index(value[5]) * 16 + digit.index(value[6])
		return (a1, a2, a3)

def rgb2gray(rgb):
	return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

def CalibratData_Lidar(LidarData_path = "Data20190325205705_297411.pickle"):
	with open(LidarData_path, 'rb') as DataLidar:
		DataLidari = pickle.load(DataLidar)
	#pdb.set_trace()
	Vertical_Angle = [-15 * np.pi / 180, 1 * np.pi / 180, -13 * np.pi / 180, 3 * np.pi / 180, -11 * np.pi / 180,
					  5 * np.pi / 180, -9 * np.pi / 180, 7 * np.pi / 180, -7 * np.pi / 180, 9 * np.pi / 180,
					  -5 * np.pi / 180, 11 * np.pi / 180, -3 * np.pi / 180, 13 * np.pi / 180, -1 * np.pi / 180,
					  15 * np.pi / 180]  # Vl16的方位角定义w。
	Vertical_Correction = [11.2, -0.7, 9.7, -2.2, 8.1, -3.7, 6.6, -5.1, 5.1, -6.6, 3.7, -8.1, 2.2, -9.7, 0.7, -11.2]  #
	pointcloud = []
	#for data_str in DataLidari['data_str']:  # 12个数据块的距离信息数据转换
	for data_str in DataLidari:  # 12个数据块的距离信息数据转换
		Azimuth_Bytes = re.findall('ffee(.{4})', data_str)  # 读取12数据块的12个角度数据
		# Azimuth_Bytes = ['6c81', '9481', 'bc81', 'e481', '0b82', '3382', '5b82', '8382', 'ab82', 'd182', 'fa82', '2183']
		Azimuth = [int(exchange_hex(i), 16) / 100 for i in Azimuth_Bytes]  # 转换12数据块的方位角度信息数据
		'''Azimuth = [331.32, 331.72, 332.12, 332.52, 332.91, 333.31, 333.71, 334.11, 334.51, 334.89, 335.3, 335.69]
           Azimuth = [55.11, 55.51, 55.91, 56.31, 56.71, 57.09, 57.51, 57.89, 58.29, 58.7, 59.09, 59.48]'''
		Distance_Bytes = re.findall('ffee.{4}(.{192})', data_str)  # 读取距离数据：192个字节
		# Distance_Bytes = ['a7040921051aa2044e280510a504212e050da10422530509ad0417de050da70410f00515bc040b000625150516170630a5040a25051a9804642d0506a9041744050ca404160000a1ab0415000067b20406f005550b05120000e31b051a0000ee', ……]
		# pdb.set_trace()
		ik = 0
		for item in Distance_Bytes:  # 12个数据块的距离信息数据转换
			# print('Azimuth[{}]={}度:'.format(ik, Azimuth[ik]))
			Distance_Division = re.findall('(.{4}).{2}', item)
			# Distance_Division = ['a704', '2105', 'a204', '2805', 'a504', '2e05', 'a104', '5305', 'ad04', 'de05', 'a704', 'f005', 'bc04', '0006', '1505', '1706', 'a504', '2505', '9804', '2d05', 'a904', '4405', 'a404', '0000', 'ab04', '0000', 'b204', 'f005', '0b05', '0000', '1b05', '0000']
			Reflectivity_Division = re.findall('.{4}(.{2})', item)  # 空4个字符取两个字符
			Distance = [int(exchange_hex(i), 16) * 2 / 1000 for i in Distance_Division]
			Reflectivity = [int(i, 16) for i in Reflectivity_Division]
			# Distance = [2.382, 2.626, 2.372, 2.64, 2.378, 2.652, 2.37, 2.726, 2.394, 3.004, 2.382, 3.04, 2.424, 3.072, 2.602, 3.118,
			#            2.378, 2.634, 2.352, 2.65, 2.386, 2.696, 2.376, 0.0,  2.39,  0.0,   2.404, 3.04, 2.582, 0.0,   2.614, 0.0]
			Distance0 = [Distance[i] for i in range(16)]
			Reflectivity0 = [Reflectivity[i] for i in range(16)]
			# Distance0 = [2.382, 2.626, 2.372, 2.64, 2.378, 2.652, 2.37, 2.726, 2.394, 3.004, 2.382, 3.04, 2.424, 3.072, 2.602, 3.118]
			Distance1 = [Distance[i + 16] for i in range(16)]
			Reflectivity1 = [Reflectivity[i + 16] for i in range(16)]
			# Distance1 = [2.378, 2.634, 2.352, 2.65, 2.386, 2.696, 2.376, 0.0,  2.39,  0.0,   2.404, 3.04, 2.582, 0.0,   2.614, 0.0]
			# xyz = convert2cart(distance, azimuth, elevation)
			if ik < 11:
				Azimuth_Bi = np.linspace(Azimuth[ik], Azimuth[ik + 1], 33)
			elif ik == 11:
				Azimuth_Bi = np.linspace(Azimuth[ik - 1], Azimuth[ik], 33) + (Azimuth[ik] - Azimuth[ik - 1])

			x0 = np.round(Distance0 * np.cos(Vertical_Angle) * np.sin(np.pi * Azimuth_Bi[:16] / 180), decimals=3)
			x1 = np.round(Distance1 * np.cos(Vertical_Angle) * np.sin(np.pi * Azimuth_Bi[16:16 + 16] / 180), decimals=3)
			y0 = np.round(Distance0 * np.cos(Vertical_Angle) * np.cos(np.pi * Azimuth_Bi[:16] / 180), decimals=3)  # 翻转坐标
			y1 = np.round(Distance1 * np.cos(Vertical_Angle) * np.cos(np.pi * Azimuth_Bi[16:16 + 16] / 180), decimals=3)  # 翻转坐标
			z0 = np.round(np.array(Distance0) * np.sin(Vertical_Angle) + np.array(Vertical_Correction) / 1000, decimals=3)
			z1 = np.round(np.array(Distance1) * np.sin(Vertical_Angle) + np.array(Vertical_Correction) / 1000, decimals=3)

			# x0 = np.round(Distance0 * np.cos(Vertical_Angle) * np.sin(np.pi * Azimuth[ik] / 180), decimals=3)
			# y0 = -np.round(Distance0 * np.cos(Vertical_Angle) * np.cos(np.pi * Azimuth[ik] / 180), decimals=3)  #翻转坐标
			# z0 = -np.round(np.array(Distance0) * np.sin(Vertical_Angle) + np.array(Vertical_Correction) / 1000, decimals=3)
			# pdb.set_trace()
			# pointcloud.append([x0*1000, y0*1000, z0*1000, Reflectivity0])
			pointcloud.append([-y0 * 1000, -z0 * 1000, x0 * 1000, Reflectivity0, Distance0])
			pointcloud.append([-y1 * 1000, -z1 * 1000, x1 * 1000, Reflectivity1, Distance1])

			ik += 1
	pointcloud = np.array(pointcloud)  # pointcloud.shape=(144, 5, 16)
	try:
		pointcloud = pointcloud.transpose((0, 2, 1))  # pointcloud.shape=(144, 16, 5)
		pointcloud = pointcloud.reshape((-1, 5))  # pointcloud.shape=(144, 16, 5)
	except:
		print('No LiDAR data for this image.')
		# pdb.set_trace()

	return pointcloud

def CalibratData_Lidar0(LidarData_path = "Data20190325205705_297411.pickle"):
	with open(LidarData_path, 'rb') as DataLidar:
		DataLidari = pickle.load(DataLidar)
	#pdb.set_trace()
	Vertical_Angle = [-15 * np.pi / 180, 1 * np.pi / 180, -13 * np.pi / 180, 3 * np.pi / 180, -11 * np.pi / 180,
					  5 * np.pi / 180, -9 * np.pi / 180, 7 * np.pi / 180, -7 * np.pi / 180, 9 * np.pi / 180,
					  -5 * np.pi / 180, 11 * np.pi / 180, -3 * np.pi / 180, 13 * np.pi / 180, -1 * np.pi / 180,
					  15 * np.pi / 180]  # Vl16的方位角定义w。
	Vertical_Correction = [11.2, -0.7, 9.7, -2.2, 8.1, -3.7, 6.6, -5.1, 5.1, -6.6, 3.7, -8.1, 2.2, -9.7, 0.7, -11.2]  #
	pointcloud = []
	for data_str in DataLidari['data_str']:  # 12个数据块的距离信息数据转换
	# for data_str in DataLidari:  # 12个数据块的距离信息数据转换
		Azimuth_Bytes = re.findall('ffee(.{4})', data_str)  # 读取12数据块的12个角度数据
		# Azimuth_Bytes = ['6c81', '9481', 'bc81', 'e481', '0b82', '3382', '5b82', '8382', 'ab82', 'd182', 'fa82', '2183']
		Azimuth = [int(exchange_hex(i), 16) / 100 for i in Azimuth_Bytes]  # 转换12数据块的方位角度信息数据
		'''Azimuth = [331.32, 331.72, 332.12, 332.52, 332.91, 333.31, 333.71, 334.11, 334.51, 334.89, 335.3, 335.69]
           Azimuth = [55.11, 55.51, 55.91, 56.31, 56.71, 57.09, 57.51, 57.89, 58.29, 58.7, 59.09, 59.48]'''
		Distance_Bytes = re.findall('ffee.{4}(.{192})', data_str)  # 读取距离数据：192个字节
		# Distance_Bytes = ['a7040921051aa2044e280510a504212e050da10422530509ad0417de050da70410f00515bc040b000625150516170630a5040a25051a9804642d0506a9041744050ca404160000a1ab0415000067b20406f005550b05120000e31b051a0000ee', ……]
		# pdb.set_trace()
		ik = 0
		for item in Distance_Bytes:  # 12个数据块的距离信息数据转换
			# print('Azimuth[{}]={}度:'.format(ik, Azimuth[ik]))
			Distance_Division = re.findall('(.{4}).{2}', item)
			# Distance_Division = ['a704', '2105', 'a204', '2805', 'a504', '2e05', 'a104', '5305', 'ad04', 'de05', 'a704', 'f005', 'bc04', '0006', '1505', '1706', 'a504', '2505', '9804', '2d05', 'a904', '4405', 'a404', '0000', 'ab04', '0000', 'b204', 'f005', '0b05', '0000', '1b05', '0000']
			Reflectivity_Division = re.findall('.{4}(.{2})', item)  # 空4个字符取两个字符
			Distance = [int(exchange_hex(i), 16) * 2 / 1000 for i in Distance_Division]
			Reflectivity = [int(i, 16) for i in Reflectivity_Division]
			# Distance = [2.382, 2.626, 2.372, 2.64, 2.378, 2.652, 2.37, 2.726, 2.394, 3.004, 2.382, 3.04, 2.424, 3.072, 2.602, 3.118,
			#            2.378, 2.634, 2.352, 2.65, 2.386, 2.696, 2.376, 0.0,  2.39,  0.0,   2.404, 3.04, 2.582, 0.0,   2.614, 0.0]
			Distance0 = [Distance[i] for i in range(16)]
			Reflectivity0 = [Reflectivity[i] for i in range(16)]
			# Distance0 = [2.382, 2.626, 2.372, 2.64, 2.378, 2.652, 2.37, 2.726, 2.394, 3.004, 2.382, 3.04, 2.424, 3.072, 2.602, 3.118]
			Distance1 = [Distance[i + 16] for i in range(16)]
			Reflectivity1 = [Reflectivity[i + 16] for i in range(16)]
			# Distance1 = [2.378, 2.634, 2.352, 2.65, 2.386, 2.696, 2.376, 0.0,  2.39,  0.0,   2.404, 3.04, 2.582, 0.0,   2.614, 0.0]
			# xyz = convert2cart(distance, azimuth, elevation)
			if ik < 11:
				Azimuth_Bi = np.linspace(Azimuth[ik], Azimuth[ik + 1], 33)
			elif ik == 11:
				Azimuth_Bi = np.linspace(Azimuth[ik - 1], Azimuth[ik], 33) + (Azimuth[ik] - Azimuth[ik - 1])

			x0 = np.round(Distance0 * np.cos(Vertical_Angle) * np.sin(np.pi * Azimuth_Bi[:16] / 180), decimals=3)
			x1 = np.round(Distance1 * np.cos(Vertical_Angle) * np.sin(np.pi * Azimuth_Bi[16:16 + 16] / 180), decimals=3)
			y0 = np.round(Distance0 * np.cos(Vertical_Angle) * np.cos(np.pi * Azimuth_Bi[:16] / 180),
						  decimals=3)  # 翻转坐标
			y1 = np.round(Distance1 * np.cos(Vertical_Angle) * np.cos(np.pi * Azimuth_Bi[16:16 + 16] / 180),
						  decimals=3)  # 翻转坐标
			z0 = np.round(np.array(Distance0) * np.sin(Vertical_Angle) + np.array(Vertical_Correction) / 1000,
						  decimals=3)
			z1 = np.round(np.array(Distance1) * np.sin(Vertical_Angle) + np.array(Vertical_Correction) / 1000,
						  decimals=3)

			# x0 = np.round(Distance0 * np.cos(Vertical_Angle) * np.sin(np.pi * Azimuth[ik] / 180), decimals=3)
			# y0 = -np.round(Distance0 * np.cos(Vertical_Angle) * np.cos(np.pi * Azimuth[ik] / 180), decimals=3)  #翻转坐标
			# z0 = -np.round(np.array(Distance0) * np.sin(Vertical_Angle) + np.array(Vertical_Correction) / 1000, decimals=3)
			# pdb.set_trace()
			# pointcloud.append([x0*1000, y0*1000, z0*1000, Reflectivity0])
			pointcloud.append([-y0 * 1000, -z0 * 1000, x0 * 1000, Reflectivity0, Distance0])
			pointcloud.append([-y1 * 1000, -z1 * 1000, x1 * 1000, Reflectivity1, Distance1])

			ik += 1
	pointcloud = np.array(pointcloud)  # pointcloud.shape=(144, 5, 16)
	try:
		pointcloud = pointcloud.transpose((0, 2, 1))  # pointcloud.shape=(144, 16, 5)
		pointcloud = pointcloud.reshape((-1, 5))  # pointcloud.shape=(144, 16, 5)
	except:
		print('No LiDAR data for this image.')
		# pdb.set_trace()

	return pointcloud

'''************************************************************************************获取雷达与图像时间匹配的数据对'''
def get_data(data_dir, Suffix_Merged='Merged', Suffix_Merge='m'):
	datapath_id = os.path.basename(os.path.normpath(data_dir))
	Divided_Symbols = datapath_id.split("_", 1)[0] #时间分割符号
	Divided_Symbols = Divided_Symbols[8:4+8] #时间分割符号
	#Divided_Symbols = '20190430' #时间分割符号
	Date_files_list = glob.glob(data_dir + '/*.*')
	Date_files_list.sort()
	iend = 1
	for i in range(len(Date_files_list)):
		#pdb.set_trace()
		Data_path = Date_files_list[i]
		Data_format = Data_path.split(".", 1)[1]
		LidarData_path = None
		ImageData_path1 = None
		if Data_format == 'pickle':
			#print('N={}/{}:  '.format(i, len(Date_files_list)))  # 打印融合图像的名称
			# Data_id = Data_path.split(".", 1)[0]
			# Data_id = os.path.basename(os.path.normpath(Data_id))
			Data_id = Data_path.split(".", 1)[0]
			Data_id = os.path.basename(os.path.normpath(Data_id))
			Data_time = Data_id.split(Divided_Symbols, 1)[1]
			line = list(Data_time)  # 将字符串转换为列表，列表的每一个元素为一个字符
			line[6] = '.'  # 修改字符串的第1个字符为.
			Data_time = ''.join(line)  # 将列表重新连接为字符串
			Data_time = np.round(float(Data_time), decimals=4)
			#pdb.set_trace()
			if i == 0:
				if Date_files_list[i+1].split(".", 1)[1] == 'png':
					Data_id1 = Date_files_list[i+1].split(".", 1)[0]
					Data_id1 = os.path.basename(os.path.normpath(Data_id1))
					Data_time1 = Data_id1.split(Divided_Symbols, 1)[1]
					line = list(Data_time1)  # 将字符串转换为列表，列表的每一个元素为一个字符
					line[6] = '.'  # 修改字符串的第1个字符为.
					Data_time1 = ''.join(line[:-1])  # 将列表重新连接为字符串
					Data_time1 = np.round(float(Data_time1), decimals=4)
					if abs(Data_time1-Data_time) < 0.1:
						LidarData_path = Data_path
						ImageData_path1 = Date_files_list[i+1]
			elif 0 < i < len(Date_files_list)-1:
				t0 = 5
				t1 = 10
				if Date_files_list[i-1].split(".", 1)[1] == 'png':
					Data_id0 = Date_files_list[i-1].split(".", 1)[0]
					Data_id0 = os.path.basename(os.path.normpath(Data_id0))
					Data_time0 = Data_id0.split(Divided_Symbols, 1)[1]
					line = list(Data_time0)  # 将字符串转换为列表，列表的每一个元素为一个字符
					line[6] = '.'  # 修改字符串的第1个字符为.
					Data_time0 = ''.join(line[:-1])  # 将列表重新连接为字符串
					Data_time0 = np.round(float(Data_time0), decimals=4)
					t0 = abs(Data_time0 - Data_time)
				if Date_files_list[i + 1].split(".", 1)[1] == 'png':
					Data_id1 = Date_files_list[i + 1].split(".", 1)[0]
					Data_id1 = os.path.basename(os.path.normpath(Data_id1))
					Data_time1 = Data_id1.split(Divided_Symbols, 1)[1]
					line = list(Data_time1)  # 将字符串转换为列表，列表的每一个元素为一个字符
					line[6] = '.'  # 修改字符串的第1个字符为.
					Data_time1 = ''.join(line[:-1])  # 将列表重新连接为字符串
					Data_time1 = np.round(float(Data_time1), decimals=4)
					t1 = abs(Data_time1 - Data_time)
				if t0 < t1 and t0 != 5 and Date_files_list[i-1].split(".", 1)[1] == 'png':
					LidarData_path = Data_path
					ImageData_path1 = Date_files_list[i - 1]
				elif t0 >= t1 and Date_files_list[i + 1].split(".", 1)[1] == 'png':
					LidarData_path = Data_path
					ImageData_path1 = Date_files_list[i + 1]
			elif i == len(Date_files_list)-1:
				if Date_files_list[i-1].split(".", 1)[1] == 'png':
					Data_id0 = Date_files_list[i - 1].split(".", 1)[0]
					Data_id0 = os.path.basename(os.path.normpath(Data_id0))
					Data_time0 = Data_id0.split(Divided_Symbols, 1)[1]
					line = list(Data_time0)  # 将字符串转换为列表，列表的每一个元素为一个字符
					line[6] = '.'  # 修改字符串的第1个字符为.
					Data_time0 = ''.join(line[:-1])  # 将列表重新连接为字符串
					Data_time0 = np.round(float(Data_time0), decimals=4)
					if abs(Data_time0-Data_time) < 0.1:
						LidarData_path = Data_path
						ImageData_path1 = Date_files_list[i - 1]
		else:
			continue

		if ImageData_path1 != None:
			img_min = os.path.basename(ImageData_path1).split('_', 1)[0][:11]
			Merge_dir = os.path.join(os.path.dirname(ImageData_path1), "data" + img_min + "Merge")
			# 20190503192834_127763.png
			if not os.path.exists(Merge_dir):  # if it exist already
				os.makedirs(Merge_dir)  # 在程序当前目录下创建一个新的文件夹名为results_files_path = "results"
			Image_name1 = os.path.basename(ImageData_path1)
			Img_id1 = Image_name1.split(".", 1)[0]
			Img_formate = Image_name1.split(".", 1)[1]
			MergeData_path1 = os.path.join(Merge_dir, Img_id1 + "{}.".format(Suffix_Merge) + Img_formate)
			if not os.path.exists(MergeData_path1):
				# cv2.imwrite(ImageDataWrite_path1, np.zeros((2, 2, 3), dtype=np.uint8))
				iend = i
				break

	if iend < len(Date_files_list):
		if LidarData_path != None:
			pointcloud = CalibratData_Lidar(LidarData_path)
			return LidarData_path, ImageData_path1, pointcloud
	else:
		return None, None, None

