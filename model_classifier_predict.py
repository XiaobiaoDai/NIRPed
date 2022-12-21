import numpy as np
import cv2, pdb
# Method to transform the coordinates of the bounding box to its original size 将边界框的坐标转换为其原始大小的方法
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio[1]))
    real_y1 = int(round(y1 // ratio[0]))
    real_x2 = int(round(x2 // ratio[1]))
    real_y2 = int(round(y2 // ratio[0]))

    return real_x1, real_y1, real_x2, real_y2

def model_classifier_predict(model_classifier, X_reImg, BBdt300_rpn, roi_helpers,class_mapping,img_data, cfg):
    '''img_data={'filepath': 'D:\\Datasets\\Data20190326\\Data20190326192242_618127.png', 'height': 512, 'width': 1280, 'daytime': 'night', 'imageset': 'train', 'bboxes': 
    [{'class': 'Ped', 'x1': 292, 'x2': 396, 'y1': 95, 'y2': 255, 'Dis': 20.4, 'Occ_Coe': 0, 'Dif': False, 'area': 16640, 'Age': 'Adult'}, 
    {'class': 'Bic', 'x1': 22, 'x2': 113, 'y1': 116, 'y2': 194, 'Dis': 43.9, 'Occ_Coe': 0, 'Dif': True, 'area': 7098, 'Age': 'Adult'}]}'''
    #BBdt300_rpn = roi_helpers.rpn_to_roi(Pcls_rpn, Pregr_rpn, cfg, K.image_dim_ordering(), overlap_thresh=0.5)#TODO：RPN的交并比阈值overlap_thresh=0.7
    # 特征图上通过快速非最大限制选出的300个交并比阈值overlap_thresh>0.7的大概率 预测锚框：BBdt300_rpn=(300, (4+1))=[x1, y1, x2, y2] 留下最后一列不取出
    # 特征图上通过快速非最大限制选出的300个交并比阈值overlap_thresh>0.7的大概率 预测锚框：BBdt300_rpn=(300, (4+1))=[x1, y1, x2, y2, Dis] 留下最后一列不取出
    # note: calc_iou converts from (x1, y1, x2, y2, td) to (x, y, w, h) format
    # note: calc_iou converts from (x1, y1, x2, y2, td) to (x, y, w, h, td) format
    img = cv2.imread(img_data['filepath'])
    (rows0, cols0) = img.shape[:2]
    ratio = [cfg.im_rows/rows0, cfg.im_cols/cols0]  # 返回前处理后的图像X_reImg：X_reImg=(Samples=1,channels=1,rows=512,cols=640)，以及缩放比例ratio = 0.5
    # convert from (x1,y1,x2,y2,Dis) to (x,y,w,h,Dis) 将300个RoIs对角坐标转化成左上角坐标+宽高
    BBdt300_rpn[:, 2] -= BBdt300_rpn[:, 0]
    BBdt300_rpn[:, 3] -= BBdt300_rpn[:, 1]
    boxes_dt = []
    # apply the spatial pyramid pooling to the proposed regions 将空间金字塔池应用于区域提案
    boxes = dict()  # 创建一个新的字典用于存放边界框回归坐标以及分类概率
    #pdb.set_trace()
    for jk in range(BBdt300_rpn.shape[0] // cfg.num_rois + 1):  # 循环jk=0,1,2,3,...,9   300 // 32 + 1 = 10
        rois = np.expand_dims(BBdt300_rpn[cfg.num_rois * jk:cfg.num_rois * (jk + 1), :], axis=0)  # 分10组rois(最后一组只有12个)，并扩维成rois.shape=(Samples=1，num_rois=32, 4)
        if rois.shape[1] == 0:  # 防止最后一组没有数据
            break
        if jk == BBdt300_rpn.shape[0] // cfg.num_rois:  # =300//32=9 如果是最后一组，填充第1维12个成32个
            # pad R
            curr_shape = rois.shape
            target_shape = (curr_shape[0], cfg.num_rois, curr_shape[2])
            rois_padded = np.zeros(target_shape).astype(rois.dtype)
            rois_padded[:, :curr_shape[1], :] = rois
            rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]  # 用最后一组的第0个RoI坐标填充第1维后面32-12=20个RoIs
            rois = rois_padded

        [Pcls_cls, Pregr_cls] = model_classifier.predict([X_reImg, rois])  # 分组预测，加快计算速度。注意：预测只输出分类预测和坐标回归预测两项
        #返回：最后锚框分类：Pcls_cls = out_class=(Samples=1, num_rois=32, nb_classes=2)；#假定分2类[行人得分，背景得分]
        #  最后锚框坐标回归：Pregr_cls = out_regr=(Samples=1, num_rois=32, 4 * (nb_classes - 1)=4)#假定分2类(回归坐标不包括背景bg)
        #  最后锚框坐标回归：Pregr_cls = out_regr=(Samples=1, num_rois=32, (4+1) * (nb_classes - 1)=(4+1))#假定分2类(回归坐标不包括背景bg)

        for ii in range(Pcls_cls.shape[1]):  # 遍历当前组32个RoIs对目标行人预测概率
            cls_num = np.argmax(Pcls_cls[0, ii, :])  # 取出分类预测概率最大值的索引
            if cls_num == Pcls_cls.shape[2]-1: #TODO： 取出当前组当前RoI的分类cls_num为最后一个（即背景），背景没有回归系数，那么就继续循环。
                continue
            if cls_num not in boxes.keys():  # 如果分类预测概率最大值的索引不在边界框boxes的键中。
                boxes[cls_num] = []  # 增加一个键cls_num对应的类
            #(x, y, w, h, Dis_rpn) = rois[0, ii, :]  # 取出当前组当前RoI的坐标
            (x, y, w, h) = rois[0, ii, :]  # 取出当前组当前RoI在特征图上的坐标
            try:
                (tx, ty, tw, th, td) = Pregr_cls[0, ii, (4+1) * cls_num:(4+1) * (cls_num + 1)]  #TODO： 取出当前组当前RoI的分类cls_num的坐标回归系数, 最后一个是背景，背景没有回归系数会抛出错误。
                # cfg.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]#边界框回归标准差系数
                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]
                Dis_cls = round(cfg.Dis_mean*np.exp(-td), 2)

                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                # 执行坐标回归计算：返回修正后的特征图上RoI左上角点坐标及宽度圆整值(像素点)
            except Exception as e:  # 抛出异常但不中断程序，继续执行except后续语句
                print(e) #TODO： 取出当前组当前RoI的分类cls_num的坐标回归系数, 最后一个是背景，背景没有回归系数会抛出错误。
                pass  # 空语句，是为了保持程序结构的完整性。pass不做任何事情，一般用做占位语句。
            boxes[cls_num].append([cfg.rpn_stride * x, cfg.rpn_stride * y, cfg.rpn_stride * (x + w), cfg.rpn_stride * (y + h), Dis_cls, np.max(Pcls_cls[0, ii, :])])  #将RoI坐标从特征图返回到预处理的前处理后图上cls_num:(x, y, w, h,  Dis_rpn, Dis_cls, Pcls )
            #boxes[cls_num].append([cfg.rpn_stride * x, cfg.rpn_stride * y, cfg.rpn_stride * (x + w), cfg.rpn_stride * (y + h), Dis_rpn, Dis_cls, np.max(Pcls_cls[0, ii, :])])  #将RoI坐标从特征图返回到预处理的前处理后图上cls_num:(x, y, w, h,  Dis_rpn, Dis_cls, Pcls )
    #pdb.set_trace()
    for cls_num, box in boxes.items():  #遍历边界框boxes的键(分类)：值(边界框坐标及概率)=(x, y, w, h, P)
        boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=0.5)  # 最后一组的第0个RoI坐标填充的20个RoIs会被抑制掉
        # 返回类cls_num通过快速非最大抑制过的边界框boxes=[x1, y1, x2, y2, Dis_C, prob]
        boxes[cls_num] = boxes_nms  # 更新类cls_num的边界框
        #class_name = class_mapping[cls_num]
        class_name = list(class_mapping.keys())[list(class_mapping.values()).index(cls_num)] #TODO:根据字典的value值取对应的key值
        #print(class_name + ":")
        for Bb in boxes_nms:
            Bb[0], Bb[1], Bb[2], Bb[3] = get_real_coordinates(ratio, Bb[0], Bb[1], Bb[2], Bb[3])  # 从前处理图返回原始图像，获取真实坐标值：ratio = 0.5
            #print('{} prob: {} and Dis:{}'.format(Bb[0: 4], np.round(Bb[-1], decimals=2), int(Bb[4])))  # 打印出类cls_num的边界框坐标和预测概率及距离
            boxes_dt.append({'class_name': class_name, 'confidence': str(np.round(Bb[-1], decimals=6)), 'bbox': list(Bb[0:4]), 'Dis': Bb[4]})
    #pdb.set_trace()
    return boxes_dt