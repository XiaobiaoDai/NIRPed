import os, sys, pdb,glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import copy
np.set_printoptions(precision=6, threshold=np.inf, edgeitems=10, linewidth=260, suppress=True)
#colors = {'Ped': ['#bf77f6', '#bf77f6'], 'Peo': ['orangered', 'orangered'], 'Bic': ['hotpink', 'hotpink'], 'Mot': ['fuchsia', 'fuchsia'], 'Ign': ['lime', 'lime'], 'bg': ['seagreen', 'seagreen']}
#colors = {'Ped': ['deeppink', 'deeppink'], 'Peo': ['orangered', 'orangered'], 'Bic': ['hotpink', 'hotpink'], 'Mot': ['fuchsia', 'fuchsia'], 'Ign': ['lime', 'lime'], 'bg': ['seagreen', 'seagreen']}
colors = {'GT': ['red', 'red'], 'Ped': ['orchid', 'orchid'], 'Peo': ['orangered', 'orangered'], 'Bic': ['hotpink', 'hotpink'], 'Mot': ['fuchsia', 'fuchsia'], 'Ign': ['lime', 'lime'], 'bg': ['seagreen', 'seagreen']}
color_Obj = colors['Ped']
color_Ped = colors['Ped']
color_bg = colors['bg']
color_GT = colors['GT']
font = ImageFont.truetype('arial.ttf', size=15)
line_width = 1
edge_kept = 2     #TODO: 标记框离图像上下边界的距离。
SpaceH_boxes = 3  #TODO: 标记框间水平间隔距离。
SpaceV_boxes = 5  #TODO: 标记框间垂直间隔距离。
show_imgs_list_Ok = ['Data20210120190056_683784N850F12.png', 'Data20200624203220_687353N850F12.png', 'Data20200624203210_534617N850F12.png', 'Data20200624203325_555118N850F12.png',
					 'Data20200702195350_520497N850F12.png', 'Data20210310194832_358490N850F12.png', 'Data20210310195229_726801N850F12.png',  'Data20210310195333_668271N850F12.png',
					  'Data20210310205950_226618N850F12.png','Data20210310212134_032894N850F12.png', 'Data20210310213031_771516N850F12.png', 'Data20210315213825_319719N850F12.png',
					 'Data20210315213943_917637N850F12.png']
def Draw_tags_orderly(draw,label,Doted_text,color_cls,left,right,top,bottom,h_box,cfg,edge_kept=5):
    label_size = draw.textsize(label, font)
    if bottom < cfg.im_rows - 10 - edge_kept - label_size[1]:
        bottom_text_boundary = bottom + 20
    else:
        bottom_text_boundary = bottom - 20

    if h_box < cfg.im_rows / 4 and top - edge_kept - label_size[1] > 0:
        text_origin = np.array([int(min(max(0.5 * (left + right) - 0.5 * label_size[0], 0), cfg.im_cols - label_size[0])), int(edge_kept)])
    else:
        text_origin = np.array([int(min(max(0.5 * (left + right) - 0.5 * label_size[0], 0), cfg.im_cols - label_size[0])), int(cfg.im_rows - edge_kept - label_size[1])])

    x = text_origin[0]
    y = text_origin[1]
    y_modified = y
    dy = label_size[1] // 5
    y_range = int((cfg.im_rows - 2 * edge_kept) / dy)

    for tp in range(0, y_range):
        flagT = 1
        if tp > 0:
            if y_modified < top - label_size[1] - 10:
                y_modified = y_modified + dy
            elif y_modified >= bottom_text_boundary:
                y_modified = y_modified - dy
            if top - label_size[1] - 10 <= y_modified < bottom_text_boundary:
                if np.random.randint(0, 2) == 0:
                    y_modified = edge_kept
                else:
                    y_modified = cfg.im_rows - edge_kept - label_size[1]

        if Doted_text != []:
            for Dot_xy in Doted_text:
                dis_x = np.abs(Dot_xy[0] - x)
                dis_y = np.abs(Dot_xy[1] - y_modified)
                if x < Dot_xy[0] and y_modified < Dot_xy[1]:
                    if dis_x < label_size[0] + SpaceH_boxes and dis_y < label_size[1] + SpaceV_boxes:
                        flagT = 0
                        break
                elif x < Dot_xy[0] and y_modified >= Dot_xy[1]:
                    if dis_x < label_size[0] + SpaceH_boxes and dis_y < Dot_xy[3] + SpaceV_boxes:
                        flagT = 0
                        break
                elif x >= Dot_xy[0] and y_modified < Dot_xy[1]:
                    if dis_x < Dot_xy[2] + SpaceH_boxes and dis_y < label_size[1] + SpaceV_boxes:
                        flagT = 0
                        break
                elif x >= Dot_xy[0] and y_modified >= Dot_xy[1]:
                    if dis_x < Dot_xy[2] + SpaceH_boxes and dis_y < Dot_xy[3] + SpaceV_boxes:
                        flagT = 0
                        break
        if flagT == 1:
            text_origin = np.array([x, y_modified])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 255, 255, 20), width=line_width)
            draw.text(text_origin, label, fill=color_cls[1], font=font)
            Doted_text.append([text_origin[0], text_origin[1], label_size[0], label_size[1]])
            break

    if flagT == 0:
        text_origin = np.array([x, bottom_text_boundary])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 255, 255, 20), width=line_width)
        draw.text(text_origin, label, fill=color_cls[1], font=font)
        Doted_text.append([text_origin[0], text_origin[1], label_size[0], label_size[1]])

    if top > text_origin[1] + label_size[1]:
        draw.line((int(0.5 * (left + right)), top, int(0.5 * (left + right)), text_origin[1] + label_size[1]), fill=color_cls[0], width=line_width)
    if bottom < text_origin[1]:
        draw.line((int(0.5 * (left + right)), bottom, int(0.5 * (left + right)), text_origin[1]), fill=color_cls[0], width=line_width)
    # pdb.set_trace()
    # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 255, 255, 20))
    # draw.text(text_origin, label, fill=color_Ped[1], font=font)
    draw.rectangle([left, top, right, bottom], outline=color_cls[0], width=line_width)  # 淡红色单框：缩放到resized图上标记框
    return draw, Doted_text

def Visual_RPN_Train(X_reImg0,Ycls_rpn,img_data,cfg):
    img_new_dir = cfg.show_imgs_directory + '\VisuaOk_%s_ClsT%d_T%s' % (cfg.Datasets, 100 * cfg.classifier_min_overlap, cfg.mode)
    if not os.path.exists(img_new_dir):
        os.makedirs(img_new_dir)
    img_path = img_data['filepath']
    #img_new_path = os.path.join(img_new_dir, os.path.basename(img_path)[:-4] + '_RpnT0.jpg')
    img_id = os.path.basename(img_path)[:-4]
    times_img_train = glob.glob(img_new_dir + '/{}_RpnT*.jpg'.format(img_id))
    times_img_train = len(times_img_train)
    img_new_path = os.path.join(img_new_dir, img_id + '_Rpn_T%d_T%d.jpg' % (100*cfg.classifier_min_overlap, times_img_train))
    # 输入RoIs信息：BBdt300_rpn=(300, 4)=(x1, y1, x2, y2)
    '''输入图像信息：img_data={'filepath': 'D:\\Datasets\\Data20190326\\Data20190326192242_618127.png', 'height': 512, 'width': 1280, 'daytime': 'night', 'imageset': 'train', 
    'bboxes': [{'class': 'Ped', 'x1': 292, 'x2': 396, 'y1': 95, 'y2': 255, 'Dis': 20.4, 'Occ_Coe': 0, 'Dif': False, 'area': 16640, 'Age': 'Adult'}, 
                {'class': 'Bic', 'x1': 22, 'x2': 113, 'y1': 116, 'y2': 194, 'Dis': 43.9, 'Occ_Coe': 0, 'Dif': True, 'area': 7098, 'Age': 'Adult'}]}'''
    GTs = img_data['bboxes']
    Ycls_rpn_index = np.argwhere(Ycls_rpn == 1)
    #pos_cls_samples0 = Ycls_rpn[0, Ycls_rpn_index[:, 1], Ycls_rpn_index[:, 2], Ycls_rpn_index[:, 3]]
    # Ped_regr_index = np.argwhere(Yregr_rpn > 0)
    # pos_reg_samples0 = Yregr_rpn[0, Ped_regr_index[:, 1], Ped_regr_index[:, 2], Ped_regr_index[:, 3]]
    # cv2.imshow('{}'.format(img_data['filepath']), X_reImg)
    ratio_rows = cfg.im_rows / int(img_data['height'])
    ratio_cols = cfg.im_cols / int(img_data['width'])
    downscale = cfg.rpn_stride
    # image_with_boxes = Image.open(img_path) #.convert('L')
    image_with_boxes = Image.fromarray(np.uint8(X_reImg0)) #image_with_boxes = image_with_boxes.resize((width, height), Image.ANTIALIAS)
    draw = ImageDraw.Draw(image_with_boxes)
    # x_len, y_len = image_with_boxes.size
    # for x in range(0, x_len, downscale):
    #     draw.line(((x, 0), (x, y_len)), (50, 50, 50))
    # for y in range(0, y_len, downscale):
    #     draw.line(((0, y), (x_len, y)), (50, 50, 50))
    n_anchsizes = len(cfg.anchor_box_scales)#锚的短边尺寸个数n_anchsizes=3
    n_anchratios = len(cfg.anchor_box_ratios)#锚的比例个数n_anchratios=3
    num_anch = n_anchsizes*n_anchratios
    for index, Anchor in enumerate(Ycls_rpn_index): #TODO:***显示RPN网络的提案的标记锚框。***显示RPN网络的提案的标记锚框。***显示RPN网络的提案的标记锚框。
        if Anchor[3] < num_anch:  #TODO:***显示RPN网络的提案的背景标记锚框。
            #continue   #TODO:***不显示RPN网络的提案的背景标记锚框。
            width2anchor = int(0.5 * cfg.anchor_box_scales[Anchor[3]//n_anchratios%n_anchsizes]*cfg.anchor_box_ratios[Anchor[3]%n_anchsizes%n_anchratios][0])
            height2anchor = int(0.5 * cfg.anchor_box_scales[Anchor[3]//n_anchratios%n_anchsizes]*cfg.anchor_box_ratios[Anchor[3]%n_anchsizes%n_anchratios][1])
            draw.rectangle([downscale * (Anchor[2] + 0.5) - width2anchor,
                            downscale * (Anchor[1] + 0.5) - height2anchor,
                            downscale * (Anchor[2] + 0.5) + width2anchor,
                            downscale * (Anchor[1] + 0.5) + height2anchor], outline=color_bg[0], width=line_width)
        else:
            #continue
            width2anchor = int(0.5 * cfg.anchor_box_scales[(Anchor[3]-num_anch)//n_anchratios%n_anchsizes]*cfg.anchor_box_ratios[(Anchor[3]-num_anch)%n_anchsizes%n_anchratios][0])
            height2anchor = int(0.5 * cfg.anchor_box_scales[(Anchor[3]-num_anch)//n_anchratios%n_anchsizes]*cfg.anchor_box_ratios[(Anchor[3]-num_anch)%n_anchsizes%n_anchratios][1])
            draw.rectangle([downscale * (Anchor[2] + 0.5) - width2anchor,
                            downscale * (Anchor[1] + 0.5) - height2anchor,
                            downscale * (Anchor[2] + 0.5) + width2anchor,
                            downscale * (Anchor[1] + 0.5) + height2anchor], outline=color_Obj[0], width=line_width)
    #pdb.set_trace()
    h_boxes = np.array([GT['y2'] - GT['y1'] for GT in GTs])
    h_box_index = np.argsort(-h_boxes, axis=0)
    h_box_index = h_box_index.tolist()
    Doted_text = []
    for index in h_box_index:  #TODO:***显示RPN网络的GT标记。***显示RPN网络的GT标记。***显示RPN网络的GT标记。
        GT = GTs[index]
        top = int(ratio_rows * GT['y1'])
        left = int(ratio_cols * GT['x1'])
        bottom = int(ratio_rows * GT['y2'])
        right = int(ratio_cols * GT['x2'])
        h_box = bottom-top
        cls_GT = GT['class']
        distance = float(GT['Dis'])
        # if cls_GT in ['bicycledriver', 'bicyclist', 'Bicyclist', 'Bic', 'motorbikedriver', 'motorcyclist',  'Motorcyclist', 'Mot', 'Sed']:
        #     label = 'Ign'
        if cls_GT in ['Pedestrian', 'pedestrian', 'Ped', 'ped', 'Bic', 'Mot']:
            label = '{}{:.1f}'.format(cls_GT[:3], distance)
            if GT['Dif'] == 1:
                label = label + 'D'
            if cfg.network in ['Resnet50VIS0', 'Resnet50VIS1']:
                if GT['Occ'] == 1:
                    label = label + 'O'
                if GT['Tru'] == 1:
                    label = label + 'T'

        else:
            label = '{}'.format(cls_GT[:3])
            #continue
        draw, Doted_text = Draw_tags_orderly(draw, label, Doted_text, color_GT, left, right, top, bottom, h_box, cfg, edge_kept=edge_kept)

        # label_size = draw.textsize(label, font)
        # text_origin = np.array([int(0.5 * (left + right) - 0.5 * label_size[0]), int(0.5 * (top + bottom)) - 0.5 * label_size[1]])
        # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 255, 255, 20), width=line_width)
        # if cls_GT in ['Pedestrian', 'pedestrian', 'Ped', 'ped']:  #['Ign', 'ign', 'Ignore', 'ignore']
        #     draw.text(text_origin, label, fill='red', font=font)
        # else:
        #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        #
        # draw.rectangle([left - 1, top - 1, right + 1, bottom + 1], outline='#ffffff', width=line_width)  # 白色双框：缩放到特征图上标记框
        # draw.rectangle([left, top, right, bottom], outline='red', width=2)  # 淡红色单框：缩放到resized图上标记框
    #pdb.set_trace()
    # plt.figure("Ped")
    # plt.imshow(image_with_boxes)
    # plt.show()
    image_with_boxes.save(img_new_path)

def Visual_RPN_Predict(X_reImg0, BB_RpnP, img_data, cfg):
    # 特征图上的RoIs：BB_RpnP = (300, 4 = (x1, y1, x2, y2) )#TODO：注意在特征图上
    img_new_dir = cfg.show_imgs_directory + '\VisuaOk_%s_ClsT%d_T%s' % (cfg.Datasets, 100 * cfg.classifier_min_overlap, cfg.mode)
    if not os.path.exists(img_new_dir):
        os.makedirs(img_new_dir)
    img_path = img_data['filepath']
    #img_new_path = os.path.join(img_new_dir, os.path.basename(img_path)[:-4] + '_RpnP0.jpg')
    img_name = os.path.basename(img_path)
    img_id = img_name[:-4]
    times_img_train = glob.glob(img_new_dir + '/{}_RpnP*.jpg'.format(img_id))
    times_img_train = len(times_img_train)
    img_new_path = os.path.join(img_new_dir, img_id + '_RpnP%d.jpg' % times_img_train)
    if img_name in show_imgs_list_Ok:
        img_new_path_BBs = os.path.join(img_new_dir, img_id + '_RpnPT%d_%d_BBs.jpg' % (100*cfg.classifier_min_overlap, times_img_train))

    image_with_boxes = Image.fromarray(np.uint8(X_reImg0))
    draw = ImageDraw.Draw(image_with_boxes)

    GTs = img_data['bboxes']
    ratio_rows = cfg.im_rows / int(img_data['height'])
    ratio_cols = cfg.im_cols / int(img_data['width'])
    h_boxes = np.array([GT['y2'] - GT['y1'] for GT in GTs])
    h_box_index = np.argsort(-h_boxes, axis=0)
    h_box_index = h_box_index.tolist()
    Doted_text = []

    # for index in h_box_index:  # TODO:***显示RPN网络的GT标记。***显示RPN网络的GT标记。***显示RPN网络的GT标记。
    #     GT = GTs[index]
    #     top = int(ratio_rows * GT['y1'])
    #     left = int(ratio_cols * GT['x1'])
    #     bottom = int(ratio_rows * GT['y2'])
    #     right = int(ratio_cols * GT['x2'])
    #     h_box = bottom - top
    #     cls_GT = GT['class']
    #     distance = float(GT['Dis'])
    #     if cls_GT in ['Pedestrian', 'pedestrian', 'Ped', 'ped', 'Bic', 'Mot']:
    #         label = '{}{:.1f}'.format(cls_GT[:3], distance)
    #         if GT['Dif'] == 1:
    #             label = label + 'D'
    #         if cfg.network in ['Resnet50VIS0', 'Resnet50VIS1']:
    #             if GT['Occ'] == 1:
    #                 label = label + 'O'
    #             if GT['Tru'] == 1:
    #                 label = label + 'T'
    #     else:
    #         label = '{}'.format(cls_GT[:3])
    #         # continue
    #     draw, Doted_text = Draw_tags_orderly(draw, label, Doted_text, color_GT, left, right, top, bottom, h_box, cfg, edge_kept=edge_kept)
    # TODO: ***对标记索引按面积大小进行排序。 ***对标记索引按面积大小进行排序。 ***对标记索引按面积大小进行排序。
    # TODO: ***对标记索引按标记框高度进行排序。***对标记索引按标记框高度进行排序。***对标记索引按标记框高度进行排序。
    BB_RpnP = BB_RpnP.tolist()
    h_box = np.array([BbDt[3]-BbDt[1] for BbDt in BB_RpnP])
    h_box_index = np.argsort(-h_box, axis=0)
    h_box_index = h_box_index.tolist()
    # TODO: ***对标记索引按标记框高度进行排序。***对标记索引按标记框高度进行排序。***对标记索引按标记框高度进行排序。
    downscale = cfg.rpn_stride

    # if img_id in ['Data20210120190056_683784N850F12']:  # 输出特征图
    RoI_FP_dir = os.path.join(img_new_dir, img_id)
    if not os.path.exists(RoI_FP_dir):
        os.makedirs(RoI_FP_dir)
    if img_name in show_imgs_list_Ok:
        for index in h_box_index: #TODO:***显示RPN网络的预测标记。***显示RPN网络的预测标记。***显示RPN网络的预测标记。***显示RPN网络的预测标记。
            BbDt = BB_RpnP[index]
            # TODO: ***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。
            # TODO: ***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。
            left = int(downscale*(0.5+BbDt[0]))
            top = int(downscale*(0.5+BbDt[1]))
            right = int(downscale*(0.5+BbDt[2]))
            bottom = int(downscale*(0.5+BbDt[3]))
            RoI_FP = image_with_boxes.crop((left, top, right, bottom))  # 截取图片

            RoI_FP_path = os.path.join(RoI_FP_dir, 'RoI_FP_T%d_%d_%d.png' % (100*cfg.classifier_min_overlap, times_img_train, index))
            RoI_FP.save(RoI_FP_path)  # 保存图片Data20181220192501_020000_FP0_51
            scale = 5
            h_resize = 14*scale
            w_resize = 8*scale
            RoI_FP_fixed = RoI_FP.resize((w_resize, h_resize))
            RoI_FP_fixed_path = os.path.join(RoI_FP_dir, 'RoI_FP_fixed_T%d_%d_%d.png' % (100*cfg.classifier_min_overlap, times_img_train, index))
            RoI_FP_fixed.save(RoI_FP_fixed_path)  # 保存图片Data20181220192501_020000_FP0_51

    downscale = cfg.rpn_stride
    x_len, y_len = image_with_boxes.size
    for x in range(0, x_len, downscale):
        draw.line(((x, 0), (x, y_len)), (50, 50, 50))
    for y in range(0, y_len, downscale):
        draw.line(((0, y), (x_len, y)), (50, 50, 50))

    if img_name in show_imgs_list_Ok:
        img_BBs_Rpn = Image.new(mode='RGB', size=(x_len, y_len), color='lightgrey')
        draw_BBs_Rpn = ImageDraw.Draw(img_BBs_Rpn)

    # if times_img_train == 0 and img_id in ['Data20210120190056_683784N850F12']:  #输出特征图
    if times_img_train == 0:  #输出特征图
        img_new_path_feature_map = os.path.join(RoI_FP_dir, img_id + 'FeatureMap_T%d.png' % (100*cfg.classifier_min_overlap))
        image_with_boxes.save(img_new_path_feature_map)

    for index in h_box_index: #TODO:***显示RPN网络的预测标记。***显示RPN网络的预测标记。***显示RPN网络的预测标记。***显示RPN网络的预测标记。
        BbDt = BB_RpnP[index]
        # TODO: ***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。
        # TODO: ***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。
        left = int(downscale*(0.5+BbDt[0]))
        top = int(downscale*(0.5+BbDt[1]))
        right = int(downscale*(0.5+BbDt[2]))
        bottom = int(downscale*(0.5+BbDt[3]))
        #draw.rectangle([left, top, right, bottom], outline=color_Ped[0], width=line_width)  # 淡红色单框：缩放到resized图上标记框
        draw.rectangle([left, top, right, bottom], outline=color_Ped[0], width=3)  # 淡红色单框：缩放到resized图上标记框
        if img_name in show_imgs_list_Ok:
            draw_BBs_Rpn.rectangle([left, top, right, bottom], outline=color_Ped[0], width=3)  # 淡红色单框：缩放到resized图上标记框


    # TODO: ***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。
    image_with_boxes.save(img_new_path)
    if img_name in show_imgs_list_Ok:
        img_BBs_Rpn.save(img_new_path_BBs)

def Visual_Cls_Train(X_reImg0,X_BBs32,Ycls_BBs32,img_data, gta_feature_map, ignore_feature_map, cfg):
    # 函数roi_helpers.calc_iou返回68<300个优选锚框：X_BBs68.shape=(1, 68<300, (4+1))坐标(x1, y1, w, h, td)
    # TODO：X_BBs68=array([[[14, 10,  1,  3],[14, 10,  1,  2], [15, 10,  1,  3], [15, 10,  1,  2]]])  #TODO：注意在特征图上
    img_new_dir = cfg.show_imgs_directory + '\VisuaOk_%s_ClsT%d_T%s' % (cfg.Datasets, 100*cfg.classifier_min_overlap, cfg.mode)
    if not os.path.exists(img_new_dir):
        os.makedirs(img_new_dir)
    img_path = img_data['filepath']
    img_id = os.path.basename(img_path)[:-4]
    times_img_train = glob.glob(img_new_dir + '/{}_ClsT*.jpg'.format(img_id))
    times_img_train = len(times_img_train)
    img_new_path = os.path.join(img_new_dir, img_id + '_ClsT_T%d_%d.jpg' % (100*cfg.classifier_min_overlap, times_img_train))

    downscale = cfg.rpn_stride
    image_with_boxes = Image.fromarray(np.uint8(X_reImg0))
    draw = ImageDraw.Draw(image_with_boxes)
    # x_len, y_len = image_with_boxes.size
    # for x in range(0, x_len, downscale):
    #     draw.line(((x, 0), (x, y_len)), (50, 50, 50))
    # for y in range(0, y_len, downscale):
    #     draw.line(((0, y), (x_len, y)), (50, 50, 50))
    #pdb.set_trace()
    h_gts = downscale *gta_feature_map[:, 3]-downscale *gta_feature_map[:, 2]
    h_box_index = np.argsort(-h_gts, axis=0)
    h_box_index = h_box_index.tolist()
    Doted_text = []
    for index in h_box_index:   #TODO:***显示检测网络的GT标记。***显示检测网络的GT标记。***显示检测网络的GT标记。
        left = downscale * gta_feature_map[index, 0]
        right = downscale * gta_feature_map[index, 1]
        top = downscale * gta_feature_map[index, 2]
        bottom = downscale * gta_feature_map[index, 3]
        distance = gta_feature_map[index, 4]
        h_box = bottom - top
        # # cls_BB = BB['class']
        # # label = '{}{:.1f}'.format(cls_BB, distance)
        label = 'Ped{:.1f}'.format(distance)
        # label_size = draw.textsize(label, font)
        # text_origin = np.array([int(0.5 * (left + right) - 0.5 * label_size[0]), int(0.5 * (top + bottom))- 0.5 * label_size[1]])
        # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 255, 255, 100), width=line_width)
        # draw.text(text_origin, label, fill='red', font=font)
        # draw.rectangle([left - 1, top - 1, right + 1, bottom + 1], outline='#ffffff', width=line_width)  # 白色双框：缩放到特征图上标记框
        # draw.rectangle([left, top, right, bottom], outline='#ff0000', width=2)  # 白色双框：缩放到特征图上标记框
        draw, Doted_text = Draw_tags_orderly(draw, label, Doted_text, color_GT, left, right, top, bottom, h_box, cfg, edge_kept=edge_kept)

    h_igns = downscale * ignore_feature_map[:, 3] - downscale * ignore_feature_map[:, 2]
    h_box_index = np.argsort(-h_igns, axis=0)
    h_box_index = h_box_index.tolist()
    for index in h_box_index: #TODO:***显示检测网络的忽略ignore标记。***显示检测网络的忽略ignore标记。
        left = downscale * ignore_feature_map[index, 0]
        right = downscale * ignore_feature_map[index, 1]
        top = downscale * ignore_feature_map[index, 2]
        bottom = downscale * ignore_feature_map[index, 3]
        h_box = bottom - top
        label = 'Ign'
        # label_size = draw.textsize(label, font)
        # text_origin = np.array([int(0.5 * (left + right) - 0.5 * label_size[0]), int(0.5 * (top + bottom))- 0.5 * label_size[1]])
        # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 255, 255, 200), width=line_width)
        # draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        # draw.rectangle([left - 1, top - 1, right + 1, bottom + 1], outline='#ffffff', width=line_width)  # 白色双框：缩放到特征图上标记框
        # draw.rectangle([left, top, right, bottom], outline='#ff0000', width=2)  # 白色双框：缩放到特征图上标记框
        draw, Doted_text = Draw_tags_orderly(draw, label, Doted_text, color_GT, left, right, top, bottom, h_box, cfg, edge_kept=edge_kept)

    for Bb_indx in range(cfg.num_rois): #TODO:***显示Cls网络的预测标记。***显示Cls网络的预测标记。***显示Cls网络的预测标记。***显示Cls网络的预测标记。
        x1_Bb = int(downscale * X_BBs32[0, Bb_indx, 0])
        y1_Bb = int(downscale * X_BBs32[0, Bb_indx, 1])
        width_Bb = int(downscale * X_BBs32[0, Bb_indx, 2])
        height_Bb = int(downscale * X_BBs32[0, Bb_indx, 3])

        if Ycls_BBs32[0, Bb_indx, 1] == 1:
            draw.rectangle([x1_Bb, y1_Bb, x1_Bb + width_Bb, y1_Bb + height_Bb], outline=color_bg[0], width=line_width)  # 紫色单框：RPN预测背景
        else:
            draw.rectangle([x1_Bb, y1_Bb, x1_Bb + width_Bb, y1_Bb + height_Bb], outline=color_Ped[0], width=line_width)  # 绿色双框：RPN预测目标

    image_with_boxes.save(img_new_path)

def Visual_Cls_Predict(X_reImg0, boxes_dt, img_data, cfg):
    # img_data={'filepath': 'I:\\Datasets\\VLP16_NIR2_2020CS\\Data20200624203N850F12\\Data20200624203636_798043N850F12.png', 'height': 512, 'width': 1280, 'daytime': 'night', 'imageset': 'train',
    # 'bboxes': [{'class': 'Ped', 'x1': 799, 'x2': 847, 'y1': 123, 'y2': 201, 'Dis': 44.6, 'Occ_Coe': 0.53, 'Dif': False, 'area': 3744, 'Age': 'Adult'},
    # {'class': 'Ign', 'x1': 643, 'x2': 693, 'y1': 106, 'y2': 142, 'Dis': 80.0, 'Occ_Coe': 0, 'Dif': True, 'area': 1800, 'Age': 'Adult'}]}
    # TODO：X_BBs68=array([[[14, 10,  1,  3],[14, 10,  1,  2], [15, 10,  1,  3], [15, 10,  1,  2]]])  #TODO：注意在特征图上
    img_new_dir = cfg.show_imgs_directory + '\VisuaOk_%s_ClsT%d_T%s' % (cfg.Datasets, 100*cfg.classifier_min_overlap, cfg.mode)
    if not os.path.exists(img_new_dir):
        os.makedirs(img_new_dir)
    img_path = img_data['filepath']
    img_name = os.path.basename(img_path)
    img_id = img_name[:-4]
    times_img_train = glob.glob(img_new_dir + '/{}_ClsP*.jpg'.format(img_id))
    times_img_train = len(times_img_train)
    img_new_path = os.path.join(img_new_dir, img_id + '_ClsP_T%d_%d.jpg' % (100*cfg.classifier_min_overlap, times_img_train))

    image_with_boxes = Image.fromarray(np.uint8(X_reImg0))
    draw = ImageDraw.Draw(image_with_boxes)

    # downscale = cfg.rpn_stride
    # x_len, y_len = image_with_boxes.size
    # for x in range(0, x_len, downscale):
    #     draw.line(((x, 0), (x, y_len)), (50, 50, 50))
    # for y in range(0, y_len, downscale):
    #     draw.line(((0, y), (x_len, y)), (50, 50, 50))

    Doted_text = []
    # TODO: ***对标记索引按面积大小进行排序。 ***对标记索引按面积大小进行排序。 ***对标记索引按面积大小进行排序。
    # TODO: ***对标记索引按标记框高度进行排序。***对标记索引按标记框高度进行排序。***对标记索引按标记框高度进行排序。
    h_box = np.array([BbDt['bbox'][3]-BbDt['bbox'][1] for BbDt in boxes_dt])
    h_box_index = np.argsort(-h_box, axis=0)
    h_box_index = h_box_index.tolist()
    if img_name in show_imgs_list_Ok:
        for index in h_box_index:  # TODO:***截取RPN网络提案的感兴趣区域。***截取RPN网络提案的感兴趣区域。***截取RPN网络提案的感兴趣区域。
            '''boxes_dt=[{'class_name': 'Ped', 'confidence': '0.999853', 'bbox': [400.0, 157.0, 464.0, 337.0], 'Dis': 26.52}, 
                        {'class_name': 'Ped', 'confidence': '0.998856', 'bbox': [352.0, 179.0, 400.0, 359.0], 'Dis': 27.62}, 
                        {'class_name': 'Ped', 'confidence': '0.972915', 'bbox': [240.0, 179.0, 272.0, 269.0], 'Dis': 56.84}, 
                        {'class_name': 'Ped', 'confidence': '0.961166', 'bbox': [752.0, 157.0, 784.0, 224.0], 'Dis': 70.3}, 
                        {'class_name': 'Ped', 'confidence': '0.800621', 'bbox': [304.0, 202.0, 336.0, 292.0], 'Dis': 50.11}]'''
            BbDt = boxes_dt[index]
            # TODO: ***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。
            # TODO: ***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。
            left = int(BbDt['bbox'][0] * cfg.im_cols / int(img_data['width']))
            top = int(BbDt['bbox'][1] * cfg.im_rows / int(img_data['height']))
            right = int(BbDt['bbox'][2] * cfg.im_cols / int(img_data['width']))
            bottom = int(BbDt['bbox'][3] * cfg.im_rows / int(img_data['height']))

            BB_predicted = image_with_boxes.crop((left, top, right, bottom))  # 截取图片
            BB_predicted_dir = os.path.join(img_new_dir, img_id)
            if not os.path.exists(BB_predicted_dir):
                os.makedirs(BB_predicted_dir)
            BB_predicted_path = os.path.join(BB_predicted_dir, 'BB_predicted_T%d_%d_%d.png' % (100*cfg.classifier_min_overlap, times_img_train, index))
            BB_predicted.save(BB_predicted_path)  # 保存图片Data20181220192501_020000_FP0_51

    # TODO: ***对标记索引按标记框高度进行排序。***对标记索引按标记框高度进行排序。***对标记索引按标记框高度进行排序。
    for index in h_box_index: #TODO:***显示RPN网络的预测标记。***显示RPN网络的预测标记。***显示RPN网络的预测标记。***显示RPN网络的预测标记。
        '''boxes_dt=[{'class_name': 'Ped', 'confidence': '0.999853', 'bbox': [400.0, 157.0, 464.0, 337.0], 'Dis': 26.52}, 
        			{'class_name': 'Ped', 'confidence': '0.998856', 'bbox': [352.0, 179.0, 400.0, 359.0], 'Dis': 27.62}, 
        			{'class_name': 'Ped', 'confidence': '0.972915', 'bbox': [240.0, 179.0, 272.0, 269.0], 'Dis': 56.84}, 
        			{'class_name': 'Ped', 'confidence': '0.961166', 'bbox': [752.0, 157.0, 784.0, 224.0], 'Dis': 70.3}, 
        			{'class_name': 'Ped', 'confidence': '0.800621', 'bbox': [304.0, 202.0, 336.0, 292.0], 'Dis': 50.11}]'''
        BbDt = boxes_dt[index]
        # TODO: ***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。
        # TODO: ***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。
        left = int(BbDt['bbox'][0] * cfg.im_cols / int(img_data['width']))
        top = int(BbDt['bbox'][1] * cfg.im_rows / int(img_data['height']))
        right = int(BbDt['bbox'][2] * cfg.im_cols / int(img_data['width']))
        bottom = int(BbDt['bbox'][3] * cfg.im_rows / int(img_data['height']))

        h_box = bottom - top
        distance = BbDt['Dis']
        prob = float(BbDt['confidence'])*100
        class_name = BbDt['class_name']
        label = 'P%d%%D%.1f' % (prob, distance)
        color_cls = colors[class_name]
        # draw.rectangle([left, top, right, bottom], outline=color_Ped[0], width=3)  # 淡红色单框：缩放到resized图上标记框
        draw, Doted_text = Draw_tags_orderly(draw, label, Doted_text, color_cls, left, right, top, bottom, h_box, cfg, edge_kept=edge_kept)

    ratio_rows = cfg.im_rows / int(img_data['height'])
    ratio_cols = cfg.im_cols / int(img_data['width'])
    GTs = img_data['bboxes']
    h_boxes = np.array([GT['y2'] - GT['y1'] for GT in GTs])
    h_box_index = np.argsort(-h_boxes, axis=0)
    h_box_index = h_box_index.tolist()
    for index in h_box_index:  # TODO:***显示RPN网络的GT标记。***显示RPN网络的GT标记。***显示RPN网络的GT标记。
        GT = GTs[index]
        top = int(ratio_rows * GT['y1'])
        left = int(ratio_cols * GT['x1'])
        bottom = int(ratio_rows * GT['y2'])
        right = int(ratio_cols * GT['x2'])
        h_box = bottom - top
        cls_GT = GT['class']
        distance = float(GT['Dis'])
        if cls_GT in ['Pedestrian', 'pedestrian', 'Ped', 'ped', 'Bic', 'Mot']:
            label = '{}{:.1f}'.format(cls_GT[:3], distance)
            if GT['Dif'] == 1:
                label = label + 'D'
            if cfg.network in ['Resnet50VIS0', 'Resnet50VIS1']:
                if GT['Occ'] == 1:
                    label = label + 'O'
                if GT['Tru'] == 1:
                    label = label + 'T'
        else:
            label = '{}'.format(cls_GT[:3])
            # continue
        draw, Doted_text = Draw_tags_orderly(draw, label, Doted_text, color_GT, left, right, top, bottom, h_box, cfg, edge_kept=edge_kept)
    # TODO: ***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。***有序显示标记框，一般对标记进行检查。

    image_with_boxes.save(img_new_path)
