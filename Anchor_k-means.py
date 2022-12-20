# -*- coding: utf-8 -*-k-means方法求锚框尺寸。
"""Created on Wed Aug 21 11:32:40 2019 @author: FanXudong"""
from os.path import join
import argparse
import numpy as np
import sys
import os
import random, pdb
from keras_frcnn.coco import COCO

def IOU(x, centroids):
    '''
    :param x: 某一个ground truth的w,h
    :param centroids:  anchor的w,h的集合[(w,h),(),...]，共k个
    :return: 单个ground truth box与所有k个anchor box的IoU值集合
    '''
    IoUs = []
    w, h = x  # ground truth的w,h
    for centroid in centroids:
        c_w, c_h = centroid  # anchor的w,h
        if c_w >= w and c_h >= h:  # anchor包围ground truth
            iou = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:  # anchor宽矮
            iou = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:  # anchor瘦长
            iou = c_w * h / (w * h + c_w * (c_h - h))
        else:  # ground truth包围anchor     means both w,h are bigger than c_w and c_h respectively
            iou = (c_w * c_h) / (w * h)
        IoUs.append(iou)  # will become (k,) shape
    return np.array(IoUs)

def avg_IOU(X, centroids):
    '''
    :param X: ground truth的w,h的集合[(w,h),(),...]
    :param centroids: anchor的w,h的集合[(w,h),(),...]，共k个
    '''
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        sum += max(IOU(X[i], centroids))  # 返回一个ground truth与所有anchor的IoU中的最大值
    return sum / n  # 对所有ground truth求平均

def write_anchors_to_file(centroids, X, anchor_file, input_shape, yolo_version,num_Anchors=0):
    '''
    :param centroids: anchor的w,h的集合[(w,h),(),...]，共k个
    :param X: ground truth的w,h的集合[(w,h),(),...]
    :param anchor_file: anchor和平均IoU的输出路径
    '''
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(anchors.shape)

    if yolo_version == 'yolov2':
        for i in range(anchors.shape[0]):
            # yolo中对图片的缩放倍数为32倍，所以这里除以32，
            # 如果网络架构有改变，根据实际的缩放倍数来
            # 求出anchor相对于缩放32倍以后的特征图的实际大小（yolov2）
            anchors[i][0] *= input_shape / 32.
            anchors[i][1] *= input_shape / 32.
    elif yolo_version == 'yolov3':
        for i in range(anchors.shape[0]):
            # 求出yolov3相对于原图的实际大小
            anchors[i][0] *= input_shape
            anchors[i][1] *= input_shape
    else:
        print("the yolo version is not right!")
        exit(-1)

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))

    f.write('avg IoU: %f\n\n' % (avg_IOU(X, centroids)))

    f.write('Anchors_FRCNN_NIRPed =[')
    for i in sorted_indices:
        f.write('[%d,%d], ' % (anchors[i, 0]*640/256, float(anchors[i, 1])))#TODO：对Ours训练集标记统计结果
    f.write('] # num_Anchors={}'.format(num_Anchors))

    print()

def k_means(X, centroids, eps, anchor_file, input_shape, yolo_version):
    N = X.shape[0]  # ground truth的个数
    iterations = 200
    print("centroids.shape", centroids)
    k, dim = centroids.shape  # anchor的个数k以及w,h两维，dim默认等于2
    prev_assignments = np.ones(N) * (-1)  # 对每个ground truth分配初始标签
    iter = 0
    old_D = np.zeros((N, k))  # 初始化每个ground truth对每个anchor的IoU

    while iter < iterations:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)  得到每个ground truth对每个anchor的IoU

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))  # 计算每次迭代和前一次IoU的变化值

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)  # 将每个ground truth分配给距离d最小的anchor序号

        if (assignments == prev_assignments).all():  # 如果前一次分配的结果和这次的结果相同，就输出anchor以及平均IoU
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids, X, anchor_file, input_shape, yolo_version,N)
            return

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float64)  # 初始化以便对每个簇的w,h求和
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]  # 将每个簇中的ground truth的w和h分别累加
        for j in range(k):  # 对簇中的w,h求平均
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j) + 1)

        prev_assignments = assignments.copy()
        old_D = D.copy()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-annFile', default=r'.\data\train.json', help='path to coco annFile\n')  # 这个文件是由运行scripts文件夹中的
    # python voc_label.py
    # 目前yolo打标签可以使用labelimg中的yolo格式
    parser.add_argument('-output_dir', default=r'.\anchors', type=str, help='Output anchor directory\n')
    parser.add_argument('-num_clusters', default=5, type=int, help='number of clusters\n')
    # parser.add_argument('-num_clusters', default=9, type=int, help='number of clusters\n')
    '''
    需要注意的是yolov2输出的值比较小是相对特征图来说的，  yolov3输出值较大是相对原图来说的，  所以yolov2和yolov3的输出是有区别的
    '''
    parser.add_argument('-yolo_version', default='yolov3', type=str, help='yolov2 or yolov3\n')
    parser.add_argument('-yolo_input_shape', default=8*32, type=int, help='input images shape，multiples of 32. etc. 352*352\n')
    args = parser.parse_args()
    annotation_dims = []
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    cocoGt = COCO(args.annFile)
    imgIds = sorted(cocoGt.getImgIds())  #imgIds=[7000000, 7000001, 7000002, 7000003, 7000004, 7000005,...]
    for i in range(len(imgIds)):
        image = cocoGt.loadImgs(ids=imgIds[i])[0]
        cols, rows = image['width'], image['height']   # TODO：NightOwls图像尺寸:  cols, rows = 1920, 1024

        anno_ids = cocoGt.getAnnIds(imgIds=imgIds[i])  # anno_ids= [7000000, 7000001]
        annos = cocoGt.loadAnns(ids=anno_ids)
        if anno_ids == []:
            continue
        for anno in annos:
            cat = cocoGt.loadCats(ids=anno['category_id'])[0]  # cat={'name': 'pedestrian', 'id': 1}
            class_name = cat['name']
            if class_name not in ['Pedestrian', 'pedestrian', 'Ped', 'ped']:
                continue
            if anno['Dis'] > 80:
                continue
            bbox = anno['bbox']
            annotation_dims.append([float(bbox[2]/cols), float(bbox[3]/rows)])

    num_annos = len(annotation_dims)
    print('#annos=%d' % num_annos)
    annotation_dims = np.array(annotation_dims)  # 保存所有ground truth框的(w,h)
    eps = 0.005
    anchor_file = join(args.output_dir, '%s_%danchors%d.txt' % (args.yolo_version, args.num_clusters, args.yolo_input_shape))
    indices = [random.randrange(annotation_dims.shape[0]) for i in range(args.num_clusters)]
    centroids = annotation_dims[indices]
    k_means(annotation_dims, centroids, eps, anchor_file, args.yolo_input_shape, args.yolo_version)
    print('centroids.shape', centroids.shape)

if __name__ == "__main__":
    main(sys.argv)