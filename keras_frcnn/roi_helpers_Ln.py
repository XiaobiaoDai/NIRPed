import numpy as np
import math, pdb, sys #提供了许多对浮点数的数学运算函数。
import keras_frcnn.data_generators_new as data_generators
import copy

#通过calc_iou()找出300个优选ROI对应ground truth里重合度最高的bbox，从而获得model_classifier的和标签。
def calc_iou(RoIs, img_data, C, class_mapping): #RoIs=[300个优选ROI]
    try:
        T_bottom_negtive = 0.05
        bboxes = []
        bboxes_ignore = []
        for box in img_data['bboxes']:   #TODO:***对训练标记进一步限定。***对训练标记进一步限定。***对训练标记进一步限定。***对训练标记进一步限定。
            distance = float(box['Dis'])
            if distance <= 0:
                distance = C.Dis_max
            box['Dis'] = distance
            if box['class'] in ['Pedestrian', 'pedestrian', 'Ped', 'ped']:
                if box['Dif'] or float(box['Occ_Coe']) > C.Occ_threshold or distance >= C.Dis_threshold:
                    bboxes_ignore.append(box)
                else:
                    bboxes.append(box)
            else:
                bboxes_ignore.append(box)

        (width, height) = (img_data['width'], img_data['height']) #从图像数据字典中取出图像的宽度像素点数和高度像素点数(width, height)=(1280, 512)

        num5_regr = 5 #锚框与GT标记框的回归参数(tx, ty, tw, th, td)共5个
        gta_feature_map = np.zeros((len(bboxes), num5_regr)) #初始化特征图上的GT标记框gta为0矩阵
        ignore_feature_map = np.zeros((len(bboxes_ignore), num5_regr)) #初始化特征图上的GT标记框gta为0矩阵

        for bbox_num, bbox in enumerate(bboxes): #枚举字典bboxes中的键bbox_num和值bbox
            gta_feature_map[bbox_num, 0] = float(bbox['x1'] * (C.im_cols / float(width)) / C.rpn_stride) # C.rpn_stride = 8
            gta_feature_map[bbox_num, 1] = float(bbox['x2'] * (C.im_cols / float(width)) / C.rpn_stride)
            gta_feature_map[bbox_num, 2] = float(bbox['y1'] * (C.im_rows / float(height)) / C.rpn_stride)
            gta_feature_map[bbox_num, 3] = float(bbox['y2'] * (C.im_rows / float(height)) / C.rpn_stride)
            gta_feature_map[bbox_num, 4] = float(bbox['Dis'])  #取出GT标记框距离信息,并进基准距离35米规范化
        for bbox_ignore_num, bbox_ignore in enumerate(bboxes_ignore): #TODO:***对检测网络的训练标记进一步限定，忽略ignore标记。
            #get the GT box coordinates, and resize to account for image resizing 将原图像中的真实GT标记框映射到特征图上来，并进行4舍5入圆整后取整数部分
            ignore_feature_map[bbox_ignore_num, 0] = float(bbox_ignore['x1'] * (C.im_cols / float(width)) / C.rpn_stride) # C.rpn_stride = 8
            ignore_feature_map[bbox_ignore_num, 1] = float(bbox_ignore['x2'] * (C.im_cols / float(width)) / C.rpn_stride)
            ignore_feature_map[bbox_ignore_num, 2] = float(bbox_ignore['y1'] * (C.im_rows / float(height)) / C.rpn_stride)
            ignore_feature_map[bbox_ignore_num, 3] = float(bbox_ignore['y2'] * (C.im_rows / float(height)) / C.rpn_stride)
            ignore_feature_map[bbox_ignore_num, 4] = float(bbox_ignore['Dis'])  #取出GT标记框距离信息,并进基准距离35米规范化

        BBs_refined_rpn = [] #设置空RoIs数组[x1, y1, w, h，Dis]
        y_class_num = [] #300个优选ROI的分类标记
        y_class_regr_coords = [] #优选ROI的回归坐标：如果是正样本就为[sx * tx, sy * ty, sw * tw, sh * th, td]，如果是负样本就为[0,0,0,0,0]。
        y_class_regr_label = []  #优选ROI的正、负标签标签：如果是正样本就为[1,1,1,1,1]，如果是负样本就为[0,0,0,0,0]
        IoUs = []  # for debugging only
        Neutral_bg = []
        idx_RoIs_ignore = []
        IoU_RoIs = np.zeros((RoIs.shape[0], ))  # 正确检测到的行人统计。
        for ix in range(RoIs.shape[0]): #用ix遍历300个优选预测锚框的行
            (x1, y1, x2, y2) = RoIs[ix, :] #(x1, y1, x2, y2, Dis_rpn_pre) = RoIs[ix, :]
            #从第ix/300个大概率、大交并比,在特征图上的预测锚框取出回归坐标信息，并进行了圆整成整数
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))
            w = x2 - x1
            h = y2 - y1

            # TODO:对与忽略重叠面积占比小于0.5的候选框，且best_iou < C.classifier_min_overlap，才可以定义为负样本。
            best_iou_ignore = 0.0 #初设当前优选框与所有GT标记框最好的交并比为0
            for bbox_num in range(len(bboxes_ignore)): #用bbox_num遍历当前图像中的所有标记框，来找出与当前优选锚框交并比最好的标记框并记录下来
                curr_iou_ignore = data_generators.iou_ignore([ignore_feature_map[bbox_num, 0], ignore_feature_map[bbox_num, 2], ignore_feature_map[bbox_num, 1], ignore_feature_map[bbox_num, 3]], [x1, y1, x2, y2])
                """在特征图上，计算当前优选框与当前标记框交并比"""
                if curr_iou_ignore > best_iou_ignore: #更新最好的交并比值和对当前标记框序号记录为best_bbox
                    best_iou_ignore = curr_iou_ignore
            # TODO:对与忽略重叠面积占比小于0.5的候选框，且best_iou < C.classifier_min_overlap，才可以定义为负样本。

            best_iou = 0.0 #初设当前优选框与所有GT标记框最好的交并比为0
            best_bbox = -1 #初设最好的锚框对应第best_bbox个GT标记框
            for bbox_num in range(len(bboxes)): #用bbox_num遍历当前图像中的所有标记框，来找出与当前优选锚框交并比最好的标记框并记录下来
                curr_iou = data_generators.iou([gta_feature_map[bbox_num, 0], gta_feature_map[bbox_num, 2], gta_feature_map[bbox_num, 1], gta_feature_map[bbox_num, 3]], [x1, y1, x2, y2])
                """在特征图上，计算当前优选框与当前标记框交并比"""
                if curr_iou > best_iou: #更新最好的交并比值和对当前标记框序号记录为best_bbox
                    best_iou = curr_iou
                    best_bbox = bbox_num

            IoU_RoIs[ix] = best_iou   #TODO:交并比小于T_bottom_negtive=0.05的当前预测框索引暂存入中立背景。

            if (best_iou < C.classifier_max_overlap) and best_iou_ignore >= 0.5:
                idx_RoIs_ignore.append(ix)

            if (best_iou < C.classifier_min_overlap) and best_iou_ignore < 0.5: #TODO:C.classifier_min_overlap = 0.5，舍去交并比小于0.1的当前优选锚框，重新循环。
                if best_iou > T_bottom_negtive:
                    cls_name = 'bg'
                    td = -1  # 背景GT标记的距离回归系数td，统一设置为-1.
                else:
                    Neutral_bg.append(ix)   #TODO:交并比小于T_bottom_negtive=0.05的当前预测框索引暂存入中立背景。
                    continue

            elif best_iou >= C.classifier_max_overlap: #C.classifier_max_overlap=0.5<= best_iou,作为正样本处理
                cls_name = bboxes[best_bbox]['class'] #当前优选预测框的类别设为GT标记框的类别。
                if cls_name != 'bg':  # 如果当前类不等于背景'bg'  #if gta_feature_map[best_bbox, 4] != 0:
                    td = -np.log(gta_feature_map[best_bbox, 4]/C.Dis0) #GT标记的距离回归系数td。
                    #td = -math.log(gta_feature_map[best_bbox, 4] / C.Dis0, 2)
                    cx_gta = (gta_feature_map[best_bbox, 0] + gta_feature_map[best_bbox, 1]) / 2.0 #计算与当前优选锚 框相交最好的GT标记框的水平中心点
                    cy_gta = (gta_feature_map[best_bbox, 2] + gta_feature_map[best_bbox, 3]) / 2.0 #计算与当前优选锚框相交最好的GT标记框的垂直中心点
                    #对每个region遍历所有的bbox，找出重合度最高的。如果best_iou小于min_overlap，则作为副样本，大于max_overlap,则作为正样本。
                    #注意，这里的overlap阈值是针对classifier的，可以不同与之前的region_proposal，具体如何设置，有什么影响，可以自己思考一下。
                    cx = x1 + w / 2.0 #计算与当前优选锚框的水平中心点
                    cy = y1 + h / 2.0 #计算与当前优选锚框的垂直中心点

                    tx = (cx_gta - cx) / float(w) #回归系数tx:GT标记框中心与当前优选锚框中心水平距离与当前优选锚框宽度比值。
                    ty = (cy_gta - cy) / float(h) #回归系数ty:GT标记框中心与当前优选锚框中心垂直距离与当前优选锚框高度比值。
                    w_feature = gta_feature_map[best_bbox, 1] - gta_feature_map[best_bbox, 0]
                    h_feature = gta_feature_map[best_bbox, 3] - gta_feature_map[best_bbox, 2]
                    tw = np.log(w_feature / float(w)) #回归系数tw:GT标记框宽度与当前优选锚框宽度比值的对数。
                    th = np.log(h_feature / float(h)) #回归系数th:GT标记框高度与当前优选锚框高度比值的对数。

                else:
                    td = -1   #背景GT标记的距离回归系数td，统一设置为-1.
                #Dis_cls_label = gta_feature_map[bbox_num, 4] * (14 * 8) / (w_feature * h_feature)  # 注意：在特征图上，对真实标记转换到特征图上的距离也进行pool_size
            else: #程序运算超出我们预想的范围就报错，打印出当前优选锚框与GT标记框最好交并比值best_iou_roi = best_iou
                #print('best_iou_roi = {}'.format(best_iou))
                #raise RuntimeError
                continue
            BBs_refined_rpn.append([x1, y1, w, h])  # 增加RoIs数组[x1, y1, w, h, Dis_rpn]
            IoUs.append(best_iou)  # 增加RoIs的最好交并比[best_iou]

            class_num = class_mapping[cls_name] #取出当前优选锚框分类数字标记
            class_label = len(class_mapping) * [0]  # [0, 0, 0]如果分3类(加上背景)，预设类标签
            class_label[class_num] = 1 #[0, 1, 0] 对当前优选锚框打上分类标签
            y_class_num.append(copy.deepcopy(class_label)) #末尾追加第ix个优选锚框的分类标签
            coords = [0] * num5_regr * (len(class_mapping) - 1) #[0, 0, 0, 0, 0]
            labels = [0] * num5_regr * (len(class_mapping) - 1) #[0, 0, 0, 0, 0]
            if cls_name != 'bg': #如果当前类不是背景'bg'
                label_pos = num5_regr * class_num #标签位置起始值为(4+1) * class_num
                sx, sy, sw, sh = C.classifier_regr_std #C.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
                #coords[label_pos:(4+1) + label_pos] = [sx * tx, sy * ty, sw * tw, sh * th, Dis_cls_label]#第ix个优选ROI的修正后的中心坐标及宽高(对应与类映射位置)
                coords[label_pos:label_pos + num5_regr] = [sx * tx, sy * ty, sw * tw, sh * th, td]#第ix个优选ROI的修正后的中心坐标及宽高(对应与类映射位置)
                labels[label_pos:label_pos + num5_regr] = [1, 1, 1, 1, 1]  #标记当前预测框是有目标的
                y_class_regr_coords.append(copy.deepcopy(coords)) #末尾追加第ix个优选ROI的分类回归中心坐标
                y_class_regr_label.append(copy.deepcopy(labels))  #末尾追加第ix个优选ROI的分类回归标签

                """#C.classifier_min_overlap=0.1<= best_iou <C.classifier_max_overlap=0.5,作为硬反例hard negative example——背景 可能被随机选上送入并训练后续Classifier network"""
                """如果当前GT标记框cls_name == 'bg'则也做为硬反例hard negative example——背景 可能被随机选上送入并训练后续Classifier network"""
            else: #如果当前优选锚框分类为背景cls_name == 'bg'，则回归坐标和标签全为0，背景已被放到类映射的最后一个。
                y_class_regr_coords.append(copy.deepcopy(coords)) #标记当前预测框是无目标的[0, 0, 0, 0, 0]
                y_class_regr_label.append(copy.deepcopy(labels)) #标记当前预测框是无目标的
        # TODO: ***当分类网络样本不足时，随机选取部分中立背景补充负样本。 ***当分类网络样本不足时，随机选取部分中立背景补充负样本。 ***当分类网络样本不足时，随机选取部分中立背景补充负样本。
        count_all_samples = len(y_class_num)
        #print('count_all_samples={}'.format(count_all_samples))
        if count_all_samples > 0:
            y_class_num = np.array(y_class_num)
            count_bg_samples = np.sum(y_class_num[:, -1])
            count_pos_samples = count_all_samples - count_bg_samples
            y_class_num = y_class_num.tolist()
        else:
            count_bg_samples = 0
            count_pos_samples = 0

        count_neu_bg = len(Neutral_bg)
        sel_bg_samples = []
        if 0 < count_all_samples < C.num_rois and count_neu_bg > 0:
            Neutral_bg = np.array(Neutral_bg)
            if count_bg_samples < C.num_rois//2:
                if count_pos_samples > 0:
                    try:
                        sel_bg_samples = np.random.choice(Neutral_bg, C.num_rois//2-count_bg_samples, replace=False).tolist()
                    except:
                        sel_bg_samples = np.random.choice(Neutral_bg, C.num_rois//2-count_bg_samples, replace=True).tolist()
                else:
                    try:
                        sel_bg_samples = np.random.choice(Neutral_bg, C.num_rois-count_bg_samples, replace=False).tolist()
                    except:
                        sel_bg_samples = np.random.choice(Neutral_bg, C.num_rois-count_bg_samples, replace=True).tolist()
                #print(sel_bg_samples)
        elif count_all_samples == 0 and count_neu_bg > 0:
            #pdb.set_trace()
            Neutral_bg = np.array(Neutral_bg)
            try:
                sel_bg_samples = np.random.choice(Neutral_bg, C.num_rois - count_bg_samples, replace=False).tolist()
            except:
                sel_bg_samples = np.random.choice(Neutral_bg, C.num_rois - count_bg_samples, replace=True).tolist()
            #print(sel_bg_samples)

        if len(sel_bg_samples) > 0:
            for ix in sel_bg_samples:
                if ix not in idx_RoIs_ignore:
                    cls_name = 'bg'
                    (x1, y1, x2, y2) = RoIs[ix, :]  # (x1, y1, x2, y2, Dis_rpn_pre) = RoIs[ix, :]
                    # 从第ix/300个大概率、大交并比,在特征图上的预测锚框取出回归坐标信息，并进行了圆整成整数
                    x1 = int(round(x1))
                    y1 = int(round(y1))
                    x2 = int(round(x2))
                    y2 = int(round(y2))
                    w = x2 - x1
                    h = y2 - y1
                    BBs_refined_rpn.append([x1, y1, w, h])  # 增加RoIs数组[x1, y1, w, h, Dis_rpn]
                    IoUs.append(IoU_RoIs[ix])  # 增加RoIs的最好交并比[best_iou]
                    class_num = class_mapping[cls_name]  # 取出当前优选锚框分类数字标记
                    class_label = len(class_mapping) * [0]  # [0, 0, 0]如果分3类(加上背景)，预设类标签
                    class_label[class_num] = 1  # [0, 1, 0] 对当前优选锚框打上分类标签
                    y_class_num.append(copy.deepcopy(class_label))  # 末尾追加第ix个优选锚框的分类标签
                    coords = [0] * num5_regr * (len(class_mapping) - 1)  # [0, 0, 0, 0, 0]
                    labels = [0] * num5_regr * (len(class_mapping) - 1)  # [0, 0, 0, 0, 0]
                    y_class_regr_coords.append(copy.deepcopy(coords))  # 标记当前预测框是无目标的[0, 0, 0, 0, 0]
                    y_class_regr_label.append(copy.deepcopy(labels))  # 标记当前预测框是无目标的
        # TODO: ***当分类网络样本不足时，随机选取部分中立背景补充负样本。 ***当分类网络样本不足时，随机选取部分中立背景补充负样本。
        # pdb.set_trace()
        # TODO: ***RPN阶段定义的样本的IoU统计。
        IoU_RoIs_original = []
        for ix in range(RoIs.shape[0]):
            if ix not in idx_RoIs_ignore:
                if not (C.classifier_min_overlap < IoU_RoIs[ix] < C.classifier_max_overlap):
                    IoU_RoIs_original.append(IoU_RoIs[ix])
        # pdb.set_trace()
        if len(BBs_refined_rpn) == 0: #如果RPN预测的BBdrs与BBgts的IoU都小于C.classifier_min_overlap=0.1，则RoIs数组[x1, y1, w, h, td]为空，则返回空
            print('return None, None, None, None, None, None, None')
            return None, None, None, None, None, None, None

        X_BBs68 = np.array(BBs_refined_rpn) #X.shape=[30,(4+1)]
        Y_BBs68_cls = np.array(y_class_num) #分类映射标记[0, 1]如果分2类(加上背景);Y1.shape=(30,2)
        Y_BBs68_regr = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1) #Y_BBs68_regr=(30,(4+1)+(4+1)=10)

        return np.expand_dims(X_BBs68, axis=0), np.expand_dims(Y_BBs68_cls, axis=0), np.expand_dims(Y_BBs68_regr, axis=0), IoUs, IoU_RoIs_original, gta_feature_map, ignore_feature_map
        "#返回：X_BBs68=[Samples=1,num_rois=68,(4+1)]——>[Samples=1,num_rois=68,4]"
        # Y_BBs68_cls=(Samples=1,num_rois=68,num_class=2),
        # Y_BBs68_regr=(Samples=1,num_rois=68,(4+1)+(4+1)=10)
        # IoUs.shape=(num_rois=68,1)
    #返回?<300优选RoI：X2坐标[x1, y1, w, h,td]，1*？*(4+1), Y1分类标记1*？*3，Y2坐标回归的标签和修正后的中心坐标及宽高(对应与类映射位置)1*？*8，IoUs与GT标记框的最好交并比？*1
    except Exception as e:
        s = sys.exc_info()
        print("Exception: Error '%s' happened on line %d" % (s[1], s[2].tb_lineno))
        pdb.set_trace()


def apply_regr(x, y, w, h, tx, ty, tw, th):
    try:
        cx = x + w / 2. #RoI中心水平坐标
        cy = y + h / 2. #RoI中心垂直坐标
        cx1 = tx * w + cx #修正后的RoI中心水平坐标
        cy1 = ty * h + cy #修正后的RoI中心垂直坐标
        w1 = math.exp(tw) * w #修正后的RoI水平宽度
        h1 = math.exp(th) * h #修正后的RoI垂直宽度
        x1 = cx1 - w1 / 2. #修正后的RoI左上角点水平坐标
        y1 = cy1 - h1 / 2. #修正后的RoI左上角点垂直坐标
        x1 = int(round(x1)) #修正后的RoI左上角点水平坐标圆整(像素点)
        y1 = int(round(y1)) #修正后的RoI左上角点垂直坐标圆整(像素点)
        w1 = int(round(w1)) #修正后的RoI水平宽度圆整(像素点)
        h1 = int(round(h1)) #修正后的RoI垂直宽度圆整(像素点)
        # print('Ok,return (x1, y1, w1, h1)=(%.6g,%.6g,%.6g,%.6g)' % (x1, y1, w1, h1))
        return x1, y1, w1, h1  #返回修正后的RoI左上角点坐标及宽度圆整值(像素点)

    except ValueError:
        # print('ValueError,return (x, y, w, h)=(%.6g,%.6g,%.6g,%.6g)'% (x, y, w, h))
        return x, y, w, h
    except OverflowError:
        # print('OverflowError,return (x, y, w, h)=(%.6g,%.6g,%.6g,%.6g)' % (x, y, w, h))
        return x, y, w, h
    except Exception as e:
        print(e)
        # print(end=',return (x, y, w, h)=(%.6g,%.6g,%.6g,%.6g)' % (x, y, w, h))
        return x, y, w, h

#X.shape=((4+1),rows=32,cols=80);
#计算预测锚框坐标定位和大小数组T.shape=regr.shape=((4+1)=5, rows=32,cols=80)
def apply_regr_np(X, T):
    try:
        x = X[0, :, :] #X=(4,rows,cols)特征图上每个像素点当前锚框的坐标
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]
        #DisX=X[4, :, :] #预测锚框的距离信息？？？？？？

        tx = T[0, :, :] #T=(4,rows,cols)特征图上每个像素点锚框到GT标记框的坐标回归的修正参数
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]
        #td = 4.0*T5[4, :, :] #标记锚框的距离修正？？？？？？cfg.std_scaling=4.0 *
        #DisT= np.exp(T[4, :, :]) #标记锚框的距离修正？？？？？？

        cx = x + w / 2. #取出预测锚框的中心坐标
        cy = y + h / 2.
        cx1 = tx * w + cx #取出锚框修正后的中心坐标
        cy1 = ty * h + cy
        w1 = np.exp(tw.astype(np.float64)) * w #转成64位浮点型在取指数：修正后的锚框的宽度
        h1 = np.exp(th.astype(np.float64)) * h #修正后的锚框的高度
        x1 = cx1 - w1 / 2. #修正后的锚框的左上水平坐标
        y1 = cy1 - h1 / 2. #修正后的锚框的左上垂直坐标

        x1 = np.round(x1) #对特征图上ROI框的左上坐标进行圆整
        y1 = np.round(y1)
        w1 = np.round(w1) #对特征图上ROI框的宽度进行圆整
        h1 = np.round(h1)
        # x1 = np.round(x1, decimals=2) #对特征图上ROI框的左上坐标进行圆整
        # y1 = np.round(y1, decimals=2)
        # w1 = np.round(w1, decimals=2) #对特征图上ROI框的宽度进行圆整
        # h1 = np.round(h1, decimals=2)
        # print(np.stack([x1, y1, w1, h1]))
        return np.stack([x1, y1, w1, h1]) #如果无异常，返回修正的锚框坐标及宽高，堆叠成数组4x32x80
        #return np.stack([x1, y1, w1, h1, td]) #如果无异常，返回修正的锚框坐标及宽高，堆叠成数组4x32x80
    except Exception as e: #如果异常，返回原ROI框坐标及宽高数组4x38x47
        s = sys.exc_info()
        print(e)
        # print(end="Exception: Error '%s' happened on line %d" % (s[1], s[2].tb_lineno))
        # pdb.set_trace()
        return X 

#通过快速非最大抑制non_max_suppression_fast()过滤掉重合度高的region并保留最优的。
def non_max_suppression_fast(boxes, overlap_thresh=0.7, max_boxes=300):#TODO：注意overlap_thresh=0.7 时，大量的高概率预测框被干掉，导致后续分类网络没有学习的预测框可用。
    # I changed this method with boxes already contains probabilities, so don't need prob send in this method
    #我用方框已经包含概率改变了这个方法，所以不需要在这个方法中发送概率
    #boxes = all_boxes = (16074, 4) + (16074, 1) = (16074, 5)
    # TODO: Caution!!! now the boxes actually is [x1, y1, x2, y2, prob] format!!!! with prob built in
    if len(boxes) == 0:
        return []
    # normalize to np.array 规范化为np.array
    boxes = np.array(boxes)
    # grab the coordinates of the bounding boxes 抓住边界框的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    try:
        np.testing.assert_array_less(x1, x2) #assert_array_less 两个数组必须形状一致，并且第一个数组的元素严格小于第二个数组的元素，否则就抛出异常
        np.testing.assert_array_less(y1, y2)
    except:
        print('Wrong at assert_array_less')
        # pdb.set_trace()

    if boxes.dtype.kind == "i": #如果boxes是整型数，将其变成浮点型
        boxes = boxes.astype("float")

    pick = [] #
    area = (x2 - x1) * (y2 - y1) #计算所有锚框面积
    # sorted by boxes last element which is prob
    indexes = np.argsort([i[-1] for i in boxes])  # 按boxes[-1](锚框概率)从小到大排序boxes而返回其索引
    while len(indexes) > 0:
        last = len(indexes) - 1 #最后一个概率最大的锚框在索引indexes中的索引last
        i = indexes[last]  #概率最大的锚框在boxes中的索引i
        pick.append(i)     #将boxes中的索引i放到挑选列表pick中

        # find the intersection 找出其余锚框与最大概率锚框的相交区域的左上坐标和右下坐标[xx1_int,yy1_int,xx2_int,yy2_int]
        xx1_int = np.maximum(x1[i], x1[indexes[:last]])
        yy1_int = np.maximum(y1[i], y1[indexes[:last]])
        xx2_int = np.minimum(x2[i], x2[indexes[:last]])
        yy2_int = np.minimum(y2[i], y2[indexes[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int) #计算其余锚框与最大概率锚框的相交区域宽度
        hh_int = np.maximum(0, yy2_int - yy1_int) #计算其余锚框与最大概率锚框的相交区域高度

        area_int = ww_int * hh_int #计算其余锚框与最大概率锚框的相交区域面积
        # find the union 计算其余锚框与最大概率锚框的相并区域面积
        area_union = area[i] + area[indexes[:last]] - area_int

        # compute the ratio of overlap 计算交并比
        overlap = area_int / (area_union + 1e-6) #防止出现分母为零，+ 1e-6

        # delete all indexes from the index list that have #删除交并比大于overlap_thresh=0.9的其余锚框索引
        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:  # 大概率锚框数大于max_boxes=300就终止，跳出循环。
            break

    boxes = boxes[pick]

    return boxes  #返回快速非最大抑制过的边界框boxes=[x1, y1, x2, y2, Dis，prob]

def rpn_to_roi(cls_rpn, regr_rpn, cfg, dim_ordering, use_regr=True, overlap_thresh=0.7, max_boxes=300):
    #RoI300_rpn = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
    # cls_rpn=P_rpn[0].shape=(Samples=1, rows=32, cols=80, num_anchors=5)
    # regr_rpn=P_rpn[1].shape=(Samples=1, rows=32, cols=80, 5*4=20)
    regr_rpn = regr_rpn / cfg.std_scaling #cfg.std_scaling=4.0   距离Dis也被再次缩小了4倍，在后面L281乘以4补回来
    # print('max_boxes=%d' % max_boxes)
    anchor_sizes = cfg.anchor_box_scales #预测锚框尺寸大小cfg.anchor_box_scales = [18, 28, 38, 60, 100]  在原始图上的锚框短边尺寸
    anchor_ratios = cfg.anchor_box_ratios #预测锚短边和长边相对锚框尺寸大小比例cfg.anchor_box_ratios = [[1, 1.56]]

    assert cls_rpn.shape[0] == 1 #确认是否是对一张图片进行rpn分类

    if dim_ordering == 'channels_first':
        (rows, cols) = cls_rpn.shape[2:]
    elif dim_ordering == 'channels_last':
        (rows, cols) = cls_rpn.shape[1:3] #取出特征图分类的行和列的维度cls_rpn=P_rpn[0].shape=(1, rows=32, cols=80, 5)
    curr_anchor = 0
    #初始预测锚框设为第0层，在任意特征点处遍历5*1=5个预测锚框,在特征图的各像素点，Decod解码出每个anchor的坐标回归系数和距离信息，作为第0维度，
    # 并放置在张量A中，A=(4=(x1, y1, x2, y2),rows=32,cols=80,num_anchor=5)
    if dim_ordering == 'channels_last':
        A = np.zeros((4, cls_rpn.shape[1], cls_rpn.shape[2], cls_rpn.shape[3]))
    elif dim_ordering == 'channels_first':
        A = np.zeros((4, cls_rpn.shape[2], cls_rpn.shape[3], cls_rpn.shape[1]))

    for anchor_size in anchor_sizes: #遍历锚框尺寸[12, 20, 35, 60, 100]
        for anchor_ratio in anchor_ratios: #遍历锚框缩放比例[[1, 1.6/2.5/0.41=1.56]]
            # pdb.set_trace()
            w_anchor = (anchor_size * anchor_ratio[0]) / cfg.rpn_stride #水平方向锚宽：80*1/16=5(RPN步长cfg.rpn_stride=16)
            h_anchor = (anchor_size * anchor_ratio[1]) / cfg.rpn_stride #垂直方向锚高：80*1.2/16=6(RPN步长cfg.rpn_stride=16)

            if dim_ordering == 'channels_first':
                regr = regr_rpn[0, 4 * curr_anchor:4 * curr_anchor + 4, :, :]
            elif dim_ordering == 'channels_last':
                regr = regr_rpn[0, :, :, 4 * curr_anchor:4 * curr_anchor + 4] #取出锚框回归系数regr=(Samples=1, rows=32,cols=80, 4)
                regr = np.transpose(regr, (2, 0, 1)) #锚框回归系数regr=4, rows=32,cols=80)

            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))#从一个坐标向量中返回一个坐标矩阵，表示特征图上每个像素点的横坐标和纵坐标。
            A[0, :, :, curr_anchor] = X - w_anchor / 2 #当前锚框，锚框的左上角点水平坐标
            A[1, :, :, curr_anchor] = Y - h_anchor / 2 #当前锚框，锚框的左上角点垂直坐标
            A[2, :, :, curr_anchor] = w_anchor #当前锚框，锚框的宽度
            A[3, :, :, curr_anchor] = h_anchor #当前锚框，锚框的高度

            if use_regr: #use_regr=True
                A[:, :, :, curr_anchor] = apply_regr_np(A[:, :, :, curr_anchor], regr)#A.shape=(4,rows=32,cols=80,num_anchor=5);计算回归锚框定位和大小数组4x32x80；regr=5, rows=32,cols=80,)
                #返回锚框修正后的坐标：np.stack([x1, y1, w1, h1])
            #TODO：***原始代码出现无穷大数无法处理的错误。
            A[2, :, :, curr_anchor] = np.maximum(1, A[2, :, :, curr_anchor]) #当前锚框每个像素点，对锚框的宽度限值，最小为1
            A[3, :, :, curr_anchor] = np.maximum(1, A[3, :, :, curr_anchor]) #当前锚框，对锚框的高度限值，最小为1

            A[2, :, :, curr_anchor] = np.minimum(cols, np.maximum(1, A[2, :, :, curr_anchor])) #当前锚框每个像素点，对锚框的宽度限值，最小为1,TODO：最大为cols=80
            A[3, :, :, curr_anchor] = np.minimum(rows, np.maximum(1, A[3, :, :, curr_anchor])) #当前锚框，对锚框的高度限值，最小为1,TODO：最大为rows=32
            #TODO：***解决无穷大数无法处理的错误。***解决无穷大数无法处理的错误。***解决无穷大数无法处理的错误。

            A[2, :, :, curr_anchor] += A[0, :, :, curr_anchor]  #当前锚框，计算锚框右下角水平坐标值
            A[3, :, :, curr_anchor] += A[1, :, :, curr_anchor]  #当前锚框，计算锚框右下角垂直坐标值

            A[0, :, :, curr_anchor] = np.maximum(0, A[0, :, :, curr_anchor]) #当前锚框，锚框不超出特征图边界
            A[1, :, :, curr_anchor] = np.maximum(0, A[1, :, :, curr_anchor])
            A[2, :, :, curr_anchor] = np.minimum(cols - 1, A[2, :, :, curr_anchor])
            A[3, :, :, curr_anchor] = np.minimum(rows - 1, A[3, :, :, curr_anchor])

            curr_anchor += 1 #当前锚框递增，共5层(5个锚框) A.shape=(4,rows=32,cols=80,num_anchor=5)

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0)) #all_boxes.shape = (12800, 4)
    all_probs = cls_rpn.transpose((0, 3, 1, 2)).reshape((-1)) #TODO：预测概率all_probs.shape=(12800,)

    x1 = all_boxes[:, 0] #锚框左上角点在特征图上的坐标值x  x1.shape = (12800, 4)
    y1 = all_boxes[:, 1] #锚框左上角点在特征图上的坐标值y  y1.shape = (12800, 4)
    x2 = all_boxes[:, 2] #锚框右下角点在特征图上的坐标值x  x2.shape = (12800, 4)
    y2 = all_boxes[:, 3] #锚框右下角点在特征图上的坐标值y  y2.shape = (12800, 4)

    ids = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))#找出锚框中不正常的框的坐标位置放入ids数组ids[0].shape=(8017,)

    all_boxes = np.delete(all_boxes, ids, 0) #从特征图上的锚框预测坐标回归数组中删除第0轴(即行)=ids的行(6383,4)  all_boxes.shape = (12282,4)
    all_probs = np.delete(all_probs, ids, 0) #从特征图上的锚框预测分类组中删除第0轴(即行)=ids的行(6111,) all_probs.shape=(12282,)

    if all_boxes.shape[0] != 0:
        # I guess boxes and prob are all 2d array, I will concat them。两个拼接数组的方法：np.vstack():在竖直方向上堆叠；np.hstack():在水平方向上平铺
        all_boxes = np.hstack((all_boxes, np.array([[p] for p in all_probs]))) # all_boxes=(12282, 4)+(12282, 1)=(12282, 5) TODO: 当all_boxes.shape[0] != 0 时，此运算无法执行。
        RoI300 = non_max_suppression_fast(all_boxes, overlap_thresh=overlap_thresh, max_boxes=max_boxes) #RoI300.shape=(300, 6)
        RoI300 = RoI300[:, 0: -1] #RoI300.shape=(300, 4=(x1, y1, x2, y2)) 留下最后一列预测概率prob不取出. 注意：这些RoIs是在特征图上
    else:
        RoI300 = all_boxes
    # #TODO：注意overlap_thresh=0.7 时，大量的高概率预测框被干掉，导致后续分类网络没有学习的预测框可用。
    #快速非最大抑制算法。all_boxes=(16074, 5)=[x1, y1, x2, y2, prob]；交并比阈值overlap_thresh=0.7(默认为0.9)； max_boxes=300。
    # omit the last column which is prob

    return RoI300 #根据快速非最大抑制算法，选出300个RoIs: RoI300.shape=(300, 4= (x1, y1, x2, y2)
    # 4是RPN网络在特征像素点上学习到的优选预测框到锚框的坐标(x1, y1, x2, y2)
    #并将其由resized图上RoIs变成特征图上的RoIs：RoI300_rpn = (300, (4 + 1)= (x1, y1, x2, y2))

