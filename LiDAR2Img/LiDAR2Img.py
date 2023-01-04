import re
import time, pdb, cv2, h5py, os, glob, shutil
import numpy as np
from LiDAR2Img.get_LiDAR_Image_data import CalibratData_Lidar0, CalibratData_Lidar
from LiDAR2Img.util import project_pointcloud_on_image

np.set_printoptions(precision=6, threshold=np.inf, edgeitems=10, linewidth=260, suppress=True)
# data_dir = '../data/miniNIRPed/images&pickles/train'
# data_dir = '../data/miniNIRPed/images&pickles/val'
data_dir = '../data/miniNIRPed/images&pickles/test'
Suffix_Merge = 'm'
Delta_time = 0.5

ParamsF12 =\
    {'20180702':{"pitch": -4.3, "roll": 0.1, "translation_x": -70, "translation_y": 112, "translation_z": 4, "yaw": 10.8},
     '20180710':{"pitch": -4.0, "roll": 0.1, "translation_x": -70, "translation_y": 112, "translation_z": 4, "yaw": 10.6},
     '20190325':{"pitch": -0.1, "roll": 0.1, "translation_x": 0.2, "translation_y": 70, "translation_z": -2, "yaw": -0.3},
     '20190326':{"pitch": -0.6, "roll": 0.1, "translation_x": 0.2, "translation_y": 70, "translation_z": -2, "yaw": -0.1},
     '20190503':{"pitch": -1.3, "roll": 0.1, "translation_x": 0.2, "translation_y": 74, "translation_z": -12, "yaw": -0.2},
     '20190508':{"pitch": 3.4, "roll": 0.1, "translation_x": 0.2, "translation_y": 74, "translation_z": -12, "yaw": -3.6},
     '2019050820':{"pitch": 1.0, "roll": 0.1, "translation_x": 0.2, "translation_y": 74, "translation_z": -12, "yaw": -3.6}, #201905081830
     '20190508204':{"pitch": 3.0, "roll": 0.1, "translation_x": -1.2, "translation_y": 74, "translation_z": 4, "yaw": -1.1}, #2019050818301
     '20200401':{"pitch": 2.0, "roll": 0.1, "translation_x": -0.2, "translation_y": 72, "translation_z": 4, "yaw": 1.7},
     '20200402':{"pitch": 3.0, "roll": 0.1, "translation_x": -0.2, "translation_y": 72, "translation_z": 4, "yaw": 1.1},
     '20200406':{"pitch": -2.9, "roll": 0.1, "translation_x": -0.2, "translation_y": 72, "translation_z": 4, "yaw": 0.2},
     '20200425':{"pitch": -3.5, "roll": 0.1, "translation_x": 0.1, "translation_y": 170, "translation_z": 4, "yaw": -0.3},
     '20200521':{"pitch": -3.8, "roll": 0.1, "translation_x": -100, "translation_y": -10, "translation_z": 40, "yaw": 3.2},
     '20200522':{"pitch": -3.8, "roll": 0.1, "translation_x": -100, "translation_y": -10, "translation_z": 40, "yaw": 3.2},
     '20200524':{"pitch": -4.3, "roll": 0.1, "translation_x": -100, "translation_y": -10, "translation_z": 40, "yaw": 1.1},
     '20200525':{"pitch": -4.2, "roll": 0.1, "translation_x": -60, "translation_y": 100, "translation_z": 4, "yaw": 0.5},
     '20200526':{"pitch": -4.5, "roll": 0.1, "translation_x": -45, "translation_y": 100, "translation_z": 4, "yaw": 0.5},
     '20200528':{"pitch": -4.1, "roll": 0.1, "translation_x": -70, "translation_y": 112, "translation_z": 4, "yaw": 0.4},
     '20200529':{"pitch": -4.1, "roll": 0.1, "translation_x": -70, "translation_y": 112, "translation_z": 4, "yaw": 0.4},
     '20200531':{"pitch": -4.3, "roll": 0.1, "translation_x": -70, "translation_y": 112, "translation_z": 4, "yaw": 0.4},
     '20200607':{"pitch": -4.3, "roll": 0.1, "translation_x": -70, "translation_y": 112, "translation_z": 4, "yaw": 0.4},
     # '20200624':{"pitch": -4.3, "roll": 0.1, "translation_x": -70, "translation_y": 112, "translation_z": 4, "yaw": 0.4},
     '20200624':{"pitch": -4.3, "roll": 0.1, "translation_x": -70, "translation_y": 112, "translation_z": 4, "yaw": 10.4},
     # '20200630':{"pitch": -4.3, "roll": 0.1, "translation_x": -70, "translation_y": 112, "translation_z": 4, "yaw": 0.4},
     '20200630':{"pitch": -4.3, "roll": 0.1, "translation_x": -70, "translation_y": 112, "translation_z": 4, "yaw": 10.4},
     '20210107':{"pitch": -3.1, "roll": 0.1, "translation_x": 0.2, "translation_y": 75, "translation_z": 4, "yaw": 1.7},
     '20210120':{"pitch": -0.4, "roll": 0.1, "translation_x": 0.2, "translation_y": 75, "translation_z": 4, "yaw": -0.5},
     '20210310':{"pitch": -0.4, "roll": 0.1, "translation_x": 0.2, "translation_y": 75, "translation_z": 4, "yaw": -0.5},
     '20210315':{"pitch": -0.4, "roll": 0.1, "translation_x": 0.2, "translation_y": 75, "translation_z": 4, "yaw": -0.5}}
'''在航空中: pitch是围绕X轴(右侧方向为正)旋转，也叫做俯仰角。yaw是围绕Y轴(向下方向为正)旋转，也叫偏航角。roll是围绕Z轴(前进方向为正)旋转，也叫翻滚角。'''

#TODO：做一个与目前已有雷达和图像数据的接口*做一个与目前已有雷达和图像数据的接口*做一个与目前已有雷达和图像数据的接口
Img_files_list0 = glob.glob(data_dir + '/*.png')
Img_files_list_merged = glob.glob(data_dir + '/m.png')
p = re.compile('m.png|N850F08.png|N850F16.png')
Img_files_list = [x for x in Img_files_list0 if not p.findall(x)]
Img_files_list.sort(reverse=False)
Lidar_best_match_files = []
num_images = len(Img_files_list)
for index, Img_path in enumerate(Img_files_list):
    Image_name = os.path.basename(Img_path)
    Divided_Symbols = Image_name[4:4+8]  #时间分割符号Divided_Symbols = '20190430'; Data20180702193607_624927N850F12.png

    # list_OK = ['20180702', '20180710', '20190325', '20190326', '20190503', '20190508', '20200521', '20200525', '20200528', '20200529', '20200607', '20200624', '20200630', '20210120', '20210310', '20210315']
    # if Divided_Symbols in [ '20181219', '20181220', '20190113'] + list_OK:
    if Divided_Symbols in ['20181219', '20181220', '20190113']:
        continue

    Img_id = Image_name.split(".", 1)[0]
    Img_formate = Image_name.split(".", 1)[1]
    img_min = Img_id.split('_', 1)[0][:11+4]
    initial_paramsF12 = ParamsF12[Divided_Symbols]

    MergeData_path = os.path.join(data_dir, Img_id + "{}.".format(Suffix_Merge) + Img_formate)
    if os.path.exists(MergeData_path):
        continue

    print('Processing for %s: %d/%d:' % (Divided_Symbols, index, num_images))

    Img_time = Img_id.split(Divided_Symbols, 1)[1]
    Img_name_head = Img_id.split(Divided_Symbols, 1)[0]
    Img_time = Img_time.replace('_', '.')
    Img_time = np.round(float(Img_time[:13]), decimals=3) #Img_time=181045.969
    Img_time_sec = int(Img_time)
    Img_time_match_down = Img_time - Delta_time
    Img_time_match_down_sec = int(Img_time_match_down)
    Img_time_match_up = Img_time + Delta_time
    Img_time_match_up_sec = int(Img_time_match_up)
    if Img_time_sec < 100000:
        Lidar_match_files = glob.glob(data_dir + '/' + Img_name_head + Divided_Symbols + '0' + str(Img_time_sec) + '*.pickle')
    else:
        Lidar_match_files = glob.glob(data_dir + '/' + Img_name_head + Divided_Symbols + str(Img_time_sec) + '*.pickle')

    if Img_time_match_down_sec < Img_time_sec:
        if Img_time_match_down_sec < 100000:
            Lidar_match_files_down = glob.glob(data_dir + '/'+Img_name_head + Divided_Symbols + '0' + str(Img_time_match_down_sec) + '*.pickle')
        else:
            Lidar_match_files_down = glob.glob(data_dir + '/' + Img_name_head + Divided_Symbols + str(Img_time_match_down_sec) + '*.pickle')
        if len(Lidar_match_files_down) > 0:
            Lidar_match_files = [Lidar_match_files_down[-1]] + Lidar_match_files
    if Img_time_match_up_sec > Img_time_sec:
        if Img_time_match_up_sec < 100000:
            Lidar_match_files_up = glob.glob(data_dir + '/'+Img_name_head + Divided_Symbols + '0' + str(Img_time_match_up_sec) + '*.pickle')
        else:
            Lidar_match_files_up = glob.glob( data_dir + '/' + Img_name_head + Divided_Symbols + str(Img_time_match_up_sec) + '*.pickle')
        if len(Lidar_match_files_up) > 0:
            Lidar_match_files = Lidar_match_files + [Lidar_match_files_up[0]]
    Time_match_errors = []
    for Lidar_file in Lidar_match_files:
        Lidar_file_id = os.path.basename(Lidar_file)
        Lidar_time = Lidar_file_id.split(Divided_Symbols, 1)[1]
        Lidar_time = Lidar_time.replace('_', '.')
        Lidar_time = np.round(float(Lidar_time[:13]), decimals=3)  # Img_time=181045.969
        Time_match_error = abs(Img_time-Lidar_time)
        Time_match_errors.append(Time_match_error)

    if Time_match_errors == []:
        continue
    min_error = min(Time_match_errors)
    if min_error > Delta_time/2:
        continue

    index_min_error = Time_match_errors.index(min_error)
    Lidar_best_match_file = Lidar_match_files[index_min_error]
    if Lidar_best_match_file not in Lidar_best_match_files:
        Lidar_best_match_files.append(Lidar_best_match_file)

    # pdb.set_trace()
    if Divided_Symbols in ['20190325', '20190326', '20190503', '20190508']:
        pointcloud = CalibratData_Lidar0(Lidar_best_match_file)
    else:
        pointcloud = CalibratData_Lidar(Lidar_best_match_file)

    img_RGB = cv2.imread(Img_path)
    image_gray_undist = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)  # image_gray_undist.shape=(711, 1269)

    #print('Projecting process:{}/{}'.format(num_calibrate_imgs, num_data_pairs))
    translation_best = [initial_paramsF12['translation_x'], initial_paramsF12['translation_y'], initial_paramsF12['translation_z']]
    rotation_best = [initial_paramsF12['pitch'], initial_paramsF12['yaw'], initial_paramsF12['roll']]

    projected_point_cloud, intensities, distances = project_pointcloud_on_image(pointcloud[:, :3], pointcloud[:, 3:], translation_best, rotation_best, image_gray_undist)
    '''雷达投影到成像平面上的有反射的像素点坐标：projected_point_cloud.shape=(3, 1186),以及像素点对应的反射强度值：intensities.shape=(1186,) distances.shape=(1186,)
    distances=array([31.4 , 31.43, 31.51, 17.87, 27.21, 31.33, 52.08, 17.86, 27.35, 17.85, ..., 23.1 , 27.77, 24.52, 25.49,  8.69, 21.86,  9.88, 28.42,  8.64,9.93])'''

    # TODO：将当前雷达数据投影到矫正的远近两张图像上*将当前雷达数据投影到矫正的近距离的图像上*
    #pdb.set_trace()
    index_dis = np.argsort(distances)
    index_dis = index_dis[::-1]

    font = cv2.FONT_HERSHEY_SIMPLEX
    colorT = (255, 255, 255)
    dd = 8  # 文字放置的垂直间隔距离

    (rows, cols) = img_RGB.shape[:2]
    if len(distances) > 0: #雷达有返回数据
        dmin = np.min(distances)
        dmax = np.max(distances)
        dmax = min(dmax, 100)
    else:#雷达无返回数据
        dmin = 0
        dmax = 0
    drange = dmax - dmin

    for i in range(1, 40):
        index_max = np.where(distances > dmax-drange*(0.02*i))
        if len(index_max[0].tolist()) > 0.02*len(distances):
            dmax = dmax-drange*(0.02*i)
            break
    for i in range(1, 40):
        index_min = np.where(distances < dmin+drange*(0.02*i))
        if len(index_min[0].tolist()) > 0.02*len(distances):
            dmin = dmin+drange*(0.02*i)
            break

    drange = dmax - dmin
    drange = max(drange, 0.1) #防止雷达只有一个返回数据，那么最大距离与最小距离相等，会造成drange=0

    Doted = []
    for i in range(projected_point_cloud.shape[1]):
        index_i = index_dis[i]
        if len(Doted) > 100:
            Doted = []
        di = round(distances[index_i], 1) #用距离显示
        if di > 100:
            continue
        di_dmin = max(di - dmin, 0)
        r0 = int(0.8 * (255 - 255 * di_dmin / drange))
        g0 = int(0.2 * (255 - 255 * di_dmin / drange))
        b0 = int(0.2 * (255 - 255 * di_dmin / drange))
        colorB = (b0, g0, r0)
        x = int(projected_point_cloud[0, index_i])
        y = int(projected_point_cloud[1, index_i])
        if 5 < x < cols - 10 and 5 < y < rows - 10:
            cv2.circle(img_RGB, (x, y), 3, colorB, -1)
            for tp in range(2, 8):
                flagT = 1
                flagD = 1
                yd = (y + 4 + tp * dd)
                if Doted != []:
                    for xy in Doted:
                        if np.abs(xy[0] - (x - 5)) <= 2 and np.abs(xy[3] - y) <= 2 and np.abs(xy[2] - di) < 0.5:
                            flagD = 0
                            break
                        if np.abs(xy[0] - (x - 5)) < 35 and np.abs(xy[1] - yd) < 12:
                            flagT = 0
                            break
                if flagD == 0:
                    break
                if flagT == 1:
                    # di = np.round(Zc[i], decimals=1) #用深度显示
                    #(ret_val, base_line) = cv2.getTextSize(str(int(di)), cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                    (ret_val, base_line) = cv2.getTextSize(str(di), cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                    text_org = (x - 5, yd)
                    cv2.rectangle(img_RGB, (text_org[0], text_org[1]), (text_org[0] + ret_val[0] - 2, text_org[1] - ret_val[1]), colorT, -1)
                    #cv2.putText(img_RGB, str(int(di)), (x - 5, yd), font, 0.5, colorB, 1, cv2.LINE_AA)
                    cv2.putText(img_RGB, str(di), (x - 5, yd), font, 0.5, colorB, 1, cv2.LINE_AA)
                    Doted.append([x - 5, yd, di, y])
                    break
    #pdb.set_trace()
    #TODO:复制原图像,并保存融合图像到另一个文件夹用于制作标记*复制原图像,并保存融合图像到另一个文件夹用于制作标记*

    if not os.path.exists(MergeData_path):
        cv2.imwrite(MergeData_path, img_RGB)


