import numpy as np
import random, pdb
def select_samples_Multi_task(IouS_array, cfg):
    # TODO：按样本交并比大小采样***按样本交并比大小采样***按样本交并比大小采样***按样本交并比大小采样***
    index_IoUs_reverse_sorted = np.argsort(-IouS_array)

    index_pos_samples = []
    index_neg_samples = []
    index_bg_samples = []
    for indx in index_IoUs_reverse_sorted:
        if IouS_array[indx] > cfg.classifier_max_overlap:
            index_pos_samples.append(indx)
        elif IouS_array[indx] > 0.05:
            index_neg_samples.append(indx)
        else:
            index_bg_samples.append(indx)
    # TODO：按样本交并比大小采样***按样本交并比大小采样***按样本交并比大小采样***按样本交并比大小采样***
    # pdb.set_trace()
    #len(IouS_array),len(index_pos_samples),len(index_neg_samples),len(index_bg_samples)
    count_pos_samples = len(index_pos_samples)
    if cfg.num_rois > 1:  # cfg.num_rois = 32 每次?ROI数量  33/2=16.5  33//2=16
        #TODO:先选正样本
        if len(index_pos_samples) < cfg.num_rois // 2: # 当正样本数目不够cfg.num_rois=32的一半时，
            if len(index_neg_samples + index_bg_samples) < cfg.num_rois // 2:# 且当负样本数目也不够cfg.num_rois=32的一半时，with image:'%s'" % (s[1], s[2].tb_lineno, img_data['filepath']))
                if len(index_pos_samples) > 0:
                    if len(index_neg_samples + index_bg_samples) > 0:
                        selected_pos_samples = index_pos_samples + index_pos_samples[:(cfg.num_rois//2 - len(index_pos_samples))]
                    else:
                        selected_pos_samples = np.random.choice(index_pos_samples, cfg.num_rois, replace=True).tolist()  # 当重复选取正样本。
                else:
                    selected_pos_samples = []
            else:# 且当负样本数目不少于cfg.num_rois=32的一半时，
                if len(index_pos_samples) > 0:
                    selected_pos_samples = index_pos_samples  # 将所有正样本直接转成列表。
                else:
                    selected_pos_samples = []
        else:
            if len(index_neg_samples + index_bg_samples) == 0:
                if len(index_pos_samples) < cfg.num_rois:
                    selected_pos_samples = index_pos_samples + index_pos_samples[:(cfg.num_rois-len(index_pos_samples))]
                else:
                    selected_pos_samples = index_pos_samples[:cfg.num_rois]
            else:
                if len(index_pos_samples) < cfg.num_rois:
                    selected_pos_samples = index_pos_samples
                else:
                    selected_pos_samples = index_pos_samples[:cfg.num_rois]

            # 参数意思：从a中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布a1 = np.random.choice(a, size=3,
            # replace=False, p=None) replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，
            # 如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了。
        # TODO:再根据正样本数量来选负样本
        if len(index_neg_samples) >= cfg.num_rois - len(selected_pos_samples):
           selected_neg_samples = index_neg_samples[:(cfg.num_rois - len(selected_pos_samples))]
        else:
           if len(index_bg_samples) >= cfg.num_rois - len(selected_pos_samples)-len(index_neg_samples):
                selected_neg_samples = index_neg_samples + np.random.choice(index_bg_samples, cfg.num_rois - len(selected_pos_samples) - len(index_neg_samples), replace=False).tolist()
           else:
               if len(index_neg_samples) >= cfg.num_rois - len(selected_pos_samples) - len(index_neg_samples) - len(index_bg_samples):
                    selected_neg_samples = index_neg_samples + index_bg_samples + index_neg_samples[:(cfg.num_rois - len(selected_pos_samples) - len(index_neg_samples)-len(index_bg_samples))]
               else:
                   selected_neg_samples = index_neg_samples + index_bg_samples + np.random.choice(index_neg_samples, cfg.num_rois - len(selected_pos_samples) - len(index_neg_samples) - len(index_bg_samples), replace=True).tolist()

        sel_samples = selected_pos_samples + selected_neg_samples  # 正负样本序列编号整合成一个列表（前正，后负）。

        if len(sel_samples) != cfg.num_rois:
            pdb.set_trace()
    else:  # TODO: cfg.num_rois = 1  #cfg.num_rois = 32 每次?ROI数量  33/2=16.5  33//2=16  只要存在正负样本，就要选出其中一个出来训练分类网络。
        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
        selected_pos_samples = index_pos_samples.tolist()
        selected_neg_samples = index_neg_samples.tolist()
        if len(index_pos_samples) and np.random.randint(0, 2):  # 随机选取负样本或正样本的一种
            sel_samples = random.choice(selected_pos_samples)  # 原始程序：sel_samples = random.choice(index_pos_samples)
        else:
            if len(index_neg_samples):
                sel_samples = random.choice(selected_neg_samples)  # 原始程序：sel_samples = random.choice(index_neg_samples)
            else:
                sel_samples = random.choice(selected_pos_samples)
            # 用一张图片中挑选出来的RoI样本对模型进行一次训练，并修正网络参数：
            # 预处理缩放的图像img_input：X.shape=(Samples=1, rows=256, cols=640, Channels=1);
            # 图像中用于训练RoI样本框roi_input：X_BBs68.shape=(Samples=1,num_rois=68,(4+1)=5);
            # 用于训练RoI样本分类标记：Ycls_BBs68.shape=(Samples=1,num_rois=68,2);如果分2类(加上背景)
            # 用于训练RoI样本回归标记和修正的回归中心坐标：Yregr_BBs68.shape=(Samples=1,num_rois=68,4+1)+4+1))
            # train_on_batch(self, x, y, class_weight=None, sample_weight=None) 本函数在一个batch的数据上进行一次参数更新
            # 函数返回训练误差的标量值或标量值的list，与 evaluate 的情形相同。
            # 手动将一个个batch的数据送入网络中训练。model.train_on_batch(X_batch, Y_batc h)
    return sel_samples, count_pos_samples

def select_samples_Faster_RCNN(Ycls_BBs68, cfg):
    neg_samples = np.where(Ycls_BBs68[0, :, -1] == 1)  # 找出负样本所在位置neg_samples 全都是负样本怎么办？[cls_ped,cls_bg]
    pos_samples = np.where(Ycls_BBs68[0, :, -1] == 0)  # 找出正样本所在位置pos_samples

    if len(neg_samples) > 0:  # 如果负样本数量大于0
        neg_samples = neg_samples[0]
    else:
        neg_samples = []

    if len(pos_samples) > 0:  # 如果正样本数量大于0
        pos_samples = pos_samples[0]
    else:
        pos_samples = []

    count_pos_samples = len(pos_samples)
    if cfg.num_rois > 1:  # cfg.num_rois = 32 每次?ROI数量  33/2=16.5  33//2=16
        # if not ((0 < len(pos_samples) < cfg.num_rois // 2) or (0 < len(neg_samples) < cfg.num_rois // 2)):
        # 	print('Number of pos_samples is %d, number of neg_samples is %d,' % (len(pos_samples), len(neg_samples)))
        if len(pos_samples) < cfg.num_rois // 2: # 当正样本数目不够cfg.num_rois=32的一半时，
            if len(neg_samples) < cfg.num_rois // 2:# 且当负样本数目也不够cfg.num_rois=32的一半时，with image:'%s'" % (s[1], s[2].tb_lineno, img_data['filepath']))
                try:
                    if len(neg_samples) == 0:
                        #print('Number of pos_samples is %d, number of neg_samples is %d, in image:%s' % (len(pos_samples), len(neg_samples),img_data['filepath']))
                        selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois, replace=True).tolist() # 当重复选取正样本。
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois // 2, replace=True).tolist() # 当重复选取正样本。
                except:
                    #print('Number of pos_samples is %d, number of neg_samples is %d, in image:%s' % (len(pos_samples), len(neg_samples), img_data['filepath']))
                    selected_pos_samples = []
                    #pdb.set_trace()
            else:# 且当负样本数目不少于cfg.num_rois=32的一半时，
                #pdb.set_trace()
                selected_pos_samples = pos_samples.tolist()  # 将所有正样本直接转成列表。
        else:
            if len(neg_samples) == 0:
                if len(pos_samples) < cfg.num_rois:
                    selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois, replace=True).tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois, replace=False).tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois // 2, replace=False).tolist()
            # 参数意思：从a中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布a1 = np.random.choice(a, size=3,
            # replace=False, p=None) replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，
            # 如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了。
        try:
            selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples), replace=False).tolist()
            # 当负样本数目不够时不能重复选取负样本，使得此随机选取过程无法执行，为了保证正负样本总数为cfg.num_rois=32，转而执行except:后的语句
        except:
            selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples), replace=True).tolist()
            # 当负样本数目不够时，可能重复选取负样本，保证正负样本总数为cfg.num_rois=32
        sel_samples = selected_pos_samples + selected_neg_samples  # 正负样本序列编号整合成一个列表（前正，后负）。
        # sel_samples=[1, 24, 16, 0, 1, 6, 14, 13, 9, 28, 21, 13, 27, 16, 4, 3, 4, 2, 11, 22, 0, 5, 0, 18, 10, 28, 27, 23, 10, 4, 14, 17]
    else:  # TODO: cfg.num_rois = 1  #cfg.num_rois = 32 每次?ROI数量  33/2=16.5  33//2=16  只要存在正负样本，就要选出其中一个出来训练分类网络。
        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
        selected_pos_samples = pos_samples.tolist()
        selected_neg_samples = neg_samples.tolist()
        if len(pos_samples) and np.random.randint(0, 2):  # 随机选取负样本或正样本的一种
            sel_samples = random.choice(selected_pos_samples)  # 原始程序：sel_samples = random.choice(pos_samples)
        else:
            if len(neg_samples):
                sel_samples = random.choice(selected_neg_samples)  # 原始程序：sel_samples = random.choice(neg_samples)
            else:
                sel_samples = random.choice(selected_pos_samples)
            # 用一张图片中挑选出来的RoI样本对模型进行一次训练，并修正网络参数：
            # 预处理缩放的图像img_input：X.shape=(Samples=1, rows=256, cols=640, Channels=1);
            # 图像中用于训练RoI样本框roi_input：X_BBs68.shape=(Samples=1,num_rois=68,(4+1)=5);
            # 用于训练RoI样本分类标记：Ycls_BBs68.shape=(Samples=1,num_rois=68,2);如果分2类(加上背景)
            # 用于训练RoI样本回归标记和修正的回归中心坐标：Yregr_BBs68.shape=(Samples=1,num_rois=68,4+1)+4+1))
            # train_on_batch(self, x, y, class_weight=None, sample_weight=None) 本函数在一个batch的数据上进行一次参数更新
            # 函数返回训练误差的标量值或标量值的list，与 evaluate 的情形相同。
            # 手动将一个个batch的数据送入网络中训练。model.train_on_batch(X_batch, Y_batc h)
    return sel_samples, count_pos_samples