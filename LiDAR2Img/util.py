import copy
import numpy as np
from pyquaternion import Quaternion
from LiDAR2Img.transformations import quaternion_from_euler

'''平移雷达测点坐标***平移雷达测点坐标***平移雷达测点坐标***平移雷达测点坐标***平移雷达测点坐标***平移雷达测点坐标***平移雷达测点坐标'''
def translate(points, t):
    """ Applies a translation to the point cloud.:param x: <np.float: 3, 1>. Translation in x, y, z. """
    for i in range(3):
        points[i, :] = points[i, :] + t[i]
    return points
'''points.shape=[4,22432]'''

'''旋转雷达测点坐标***旋转雷达测点坐标***旋转雷达测点坐标***旋转雷达测点坐标***旋转雷达测点坐标***旋转雷达测点坐标***旋转雷达测点坐标'''
def rotate(points, rot_matrix):
    """ Applies a rotation.   :param rot_matrix: <np.float: 3, 3>. Rotation matrix. """
    points[:3, :] = np.dot(rot_matrix, points[:3, :])
    return points
'''points.shape=[3,22432]'''

'''将转换到相机坐标系上的雷达点投影到成像平面上来***将转换到相机坐标系上的雷达点投影到成像平面上来***将转换到相机坐标系上的雷达点投影到成像平面上来
points = view_points(pointcloud, camera_intrinsic, normalize=True)'''
def view_points(points, view, normalize):
    assert view.shape[0] <= 4  #相机内参view.shape=[3,3]
    assert view.shape[1] <= 4
    assert points.shape[0] == 3  #points.shape=[3,22432]
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view  #相机内参扩成齐次坐标viewpad.shape=[4,4]
    nbr_points = points.shape[1]  #取出雷达点数
    # Do operation in homogenous coordinates 在齐次坐标下操作
    points = np.concatenate((points, np.ones((1, nbr_points))))#扩成齐次坐标points.shape=[4,22432]
    points = np.dot(viewpad, points) #矩阵积计算
    points = points[:3, :] #取出三维坐标points.shape=[3,22432]
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
        '''xyz都除以z轴坐标值：fx/z，fy/z，1，变成成像平面上的齐次坐标点。'''
    return points
'''雷达投影到成像平面上的齐次点坐标（fx/z，fy/z，1）：points.shape=[3,22432]'''

def project_pointcloud_on_image(pointcloud, intensities, translation, rotation, image):
    """ :param translation: (x, y, z) in m    :param rotation: (yaw, pitch, roll) in degrees """
    #TODO：从matlab的mat文件读入相机参数：
    #camera_intrinsic = stereocameraParams['IntrinsicMatrix1']
    #camera_intrinsic = np.array([[2746, 0., 569], [0., 2744,  364], [0.,    0.,    1.]])#NIR850F12  no Skew V1 Data20200406
    #camera_intrinsic = np.array([[2742, 0., 627], [0., 2739,  356], [0.,    0.,    1.]])#NIR850F12  no Skew V2 Data20200406
    #camera_intrinsic = np.array([[2742, 11.5, 637], [0., 2739,  370], [0.,    0.,    1.]])#NIR850F12  Skew VS1 Data20200406
    #camera_intrinsic = np.array([[2742, 11.6, 651], [0., 2739,  357], [0.,    0.,    1.]])#NIR850F12  Skew VS2 Data20200406
    camera_intrinsic = np.array([[2751, 8.947, 623], [0., 2751,  322], [0.,    0.,    1.]])#NIR850F12  Skew VS2 Data20200426
    #translation = -np.array(translation)  #反向，在此不需要了。
    translation = np.array(translation)
    #TODO：欧拉角是如何转换成旋转矩阵
    #rotation = Quaternion(quaternion_from_euler(*np.radians(np.array(rotation)))).rotation_matrix.T
    rotation = Quaternion(quaternion_from_euler(*np.radians(np.array(rotation)))).rotation_matrix   #不要转置，直接将雷达坐标系下的点转成相机坐标系下的点
    '''rotation=array([[ 0.99981419, -0.00274765, -0.01907976],
                       [ 0.00174501,  0.99862626, -0.05236934],
                       [ 0.01919744,  0.05232631,  0.9984455 ]])
                  R =    0.9998   -0.0027   -0.0191
                         0.0017    0.9986   -0.0524
                         0.0192    0.0523    0.9984
                  R =    0.9998   -0.0017   -0.0192
                         0.0007    0.9986   -0.0523
                         0.0193    0.0523    0.9984
                         '''
    #print('rotation={}'.format(rotation))
    pc = copy.deepcopy(pointcloud)  #不能变换原始点云坐标，否则会影响远距离图像与此点云数据的融合。
    pc = pc.T #pointcloud.shape=(22432, 4)->(4, 22432)
    # pointcloud = translate(pointcloud, translation)  #平移雷达测点坐标
    # pointcloud = rotate(pointcloud, rotation) #将雷达点坐标旋转平移变换到到图像上
    pc = rotate(pc, rotation) #将雷达点坐标旋转平移变换到到图像上
    pc = translate(pc, translation)  #平移雷达测点坐标
    depths = pc[2, :] #取出投影到相机坐标系上的雷达点的深度信息，即z轴坐标值:depths.shape=[1,22432]
    points = view_points(pc, camera_intrinsic, normalize=True)
    '''雷达投影到像素平面上的齐次像素点坐标（fx/z，fy/z，1）：points.shape=[3,22432]'''
    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0) #掩膜掉没有返回非正直的深度信息的到图像的点
    mask = np.logical_and(mask, points[0, :] > 1) #掩膜掉超出图像左边的雷达投影到图像的点
    mask = np.logical_and(mask, points[0, :] < image.shape[1] - 1) #掩膜掉超出图像右边的雷达投影到图像的点
    mask = np.logical_and(mask, points[1, :] > 1) #掩膜掉超出图像上边的雷达投影到图像的点
    mask = np.logical_and(mask, points[1, :] < image.shape[0] - 1) #掩膜掉超出图像下边的雷达投影到图像的点
    points = points[:, mask]#取出经过以上未掩膜掉雷达投影到图像的点
    intensities = intensities[mask, :] #取出经过以上未掩膜掉雷达投影到图像的点的强度信息
    distances = intensities[:, 1] #取出经过以上未掩膜掉雷达投影到图像的点的强度信息
    intensities = intensities[:, 0] #取出经过以上未掩膜掉雷达投影到图像的点的强度信息
    return points, intensities, distances
'''雷达投影到成像平面上的有反射的像素点坐标：points.shape=[3,?<22432],以及像素点对应的反射强度值：intensities.shape=[1,?<22432]'''
