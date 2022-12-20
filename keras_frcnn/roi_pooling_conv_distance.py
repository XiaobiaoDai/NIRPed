from keras.layers import Layer
import keras.backend as K
import numpy as np

if K.backend() == 'tensorflow':
    import tensorflow as tf


class RoiPoolingConv(Layer): #RoiPoolingConv类，继承Layer类
    # RoiPoolingConv类初始化输入常参：pooling_regions=14, num_rois=32；
    # RoiPoolingConv类输入变量：x=[base_layers, input_rois]：base_layers=shared_layers=(Samples=1, rows=32, cols=40, filters=512)；
    # roi_input=(Samples=1, ?=32, 4)   根本就没有这个输入参数￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥
    # 输出特征图上统一大小的锚框图像：out_roi_pool=rois_output_final=(Samples=1, self.num_rois=32, self.pool_size=14, self.pool_size=14, self.nb_channels=512)
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:`(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, row_pool_size, col_pool_size, num_rois, **kwargs):
        #RoiPoolingConv类初始化输入常参：row_pool_size=14,col_pool_size=8, num_rois = 32；
        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'channels_first', 'channels_last'}, 'dim_ordering must be in {channels_first, channels_last}'

        self.row_pool_size = row_pool_size  # self.row_pool_size = row_pool_size=14
        self.col_pool_size = col_pool_size  #self.col_pool_size = col_pool_size=8
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs) 
        #钻石继承；super(RoiPoolingConv,self) 首先找到 RoiPoolingConv的父类（就是类Layer），
        #然后把类B(RoiPoolingConv)的对象 RoiPoolingConv 转换为类A(Layer)的对象

    def build(self, input_shape):#构建通道数
        #input_rois=roi_input=(Samples=1, ?=32, (4+1))  ????//
        #input_rois=base_layers=shared_layers=(Samples=1, rows=32, cols=40, filters=512)
        if self.dim_ordering == 'channels_first':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'channels_last':
            self.nb_channels = input_shape[0][3] #取出通道数self.nb_channels=filters=512

    def compute_output_shape(self, input_shape):#计算输出数据维度5D
        if self.dim_ordering == 'channels_first':
            return None, self.num_rois, self.nb_channels, self.row_pool_size, self.col_pool_size
        else:
            #return None, self.num_rois, self.row_pool_size, self.col_pool_size, self.nb_channels
            return None, self.num_rois, self.row_pool_size, self.col_pool_size, self.nb_channels
            #输出特征图上统一大小的的RoI图像：out_roi_pool =
            # rois_output_final = (Samples = 1, self.num_rois = 32, self.row_pool_size = 14, self.col_pool_size = 8, self.nb_channels = 512)
    def call(self, x, mask=None): #x=[base_layers, input_rois]主要的功能实现在call中
        #类输入变量：x=[base_layers, input_rois]：img = x[0]=base_layers=shared_layers=(Samples=1, rows=32, cols=40, filters=512)；
        #特征图上:rois = x[1]=roi_input=(Samples=1, ?=32, (4+1))
        assert(len(x) == 2) #检查条件(len(x) == 2) ，如果为真，就不做任何事。如果为假，则会抛出AssertError并且包含错误信息。

        img = x[0] #取出特征图，img=(Samples=1, rows=32, cols=40, filters=512)  (注意，这是在特征图上对图像进行操作)
        rois = x[1] #取出rois，rois=(Samples=1, num_rois=32, (4+1)=[x,y,w,h,Dis])

        input_shape = K.shape(img) #input_shape=(Samples=1, rows=32, cols=40, filters=512)

        rois_output = [] #创建一空列表，用于存放num_rois=32个RoIs。

        for roi_idx in range(self.num_rois): #self.num_rois=num_rois=32

            x = rois[0, roi_idx, 0] #取出第roi_idx个猫框的左上角坐标及宽高尺寸和距离信息
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            #Dis= rois[0, roi_idx, 4] #取出此RoI的预测距离
            
            row_length = h / float(self.row_pool_size) #假如输入维度为14×8，则self.row_pool_size = 14,h/14即为把锚框分割后的小矩形的宽
            col_length = w / float(self.col_pool_size) #self.col_pool_size = col_pool_size=8

            #num_pool_regions = self.pool_size

            #NOTE: the RoiPooling implementation differs between theano and tensorflow due to the lack of a resize op
            # in theano. The theano implementation is much less efficient and leads to long compile times
            #注意：由于在theano中缺少调整大小操作，因此在theano和tensorflow之间的RoiPooling实现不同。theano实现效率低得多，导致编译时间长

            if self.dim_ordering == 'channels_first':
                for jy in range(self.row_pool_size):
                    for ix in range(self.col_pool_size):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = K.cast(x1, 'int32')
                        x2 = K.cast(x2, 'int32')
                        y1 = K.cast(y1, 'int32')
                        y2 = K.cast(y2, 'int32')

                        x2 = x1 + K.maximum(1, x2-x1)
                        y2 = y1 + K.maximum(1, y2-y1)
                        
                        new_shape = [input_shape[0], input_shape[1], y2 - y1, x2 - x1]

                        x_crop = img[:, :, y1:y2, x1:x2] #特征图，img=(Samples=1, rows=32, cols=40, filters=512)  这是theano后端是的操作，很懵好像很不对头
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        rois_output.append(pooled_val)
#cast(x,dtype,name=None)将x的数据格式转化成dtype数据类型.例如，原来x的数据格式是bool，那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以
            elif self.dim_ordering == 'channels_last':
                x = K.cast(x, 'int32') #转化为二进制带符号的32位整数
                y = K.cast(y, 'int32')
                w = K.cast(w, 'int32')
                h = K.cast(h, 'int32')
                #在特征图img=(Samples=1, rows=32, cols=40, filters=512)上取出锚框，并对其进行resize：roi_resize=(14,8,512)
                # roi_resize = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.row_pool_size, self.col_pool_size))
                roi_resize = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.row_pool_size, self.col_pool_size))
                # 进行ROI pool，之所以需要归一化框的坐标是因为tf接口的要求
                #pooledFeatures = tf.image.crop_and_resize(image=featureMaps, boxes=boxes, box_ind=box_ind,crop_size=crop_size)

                rois_output.append(roi_resize) #输出列表尾追加统一大小尺寸的RoI图像rois_output=[(rows=14,cols=8,channels=512),(rows=14,cols=8,channels=512),……]
                #输出列表尾追加统一大小尺寸的RoI图像rois_output=[(rows=14,cols=14,channels=512),(rows=14,cols=14,channels=512),……]

        rois_output_final = K.concatenate(rois_output, axis=0) #沿axis轴将rois_output串接起来rois_output_final=(num_rois,14,8,512)
        #rois_output_final = K.reshape(rois_output_final, (1, self.num_rois, self.row_pool_size, self.pool_size, self.nb_channels))
        rois_output_final = K.reshape(rois_output_final, (1, self.num_rois, self.row_pool_size, self.col_pool_size, self.nb_channels))
        #rois_output_final=(Samples=1, self.num_rois=32, rows=14,cols=8, self.nb_channels=512)

        if self.dim_ordering == 'channels_first':
            rois_output_final = K.permute_dimensions(rois_output_final, (0, 1, 4, 2, 3))
        else:
            rois_output_final = K.permute_dimensions(rois_output_final, (0, 1, 2, 3, 4))
            #rois_output_final=(Samples=1, self.num_rois=32, rows=14,cols=14, self.nb_channels=512)
            #rois_output_final=(Samples=1, self.num_rois=32, rows=14,cols=8, self.nb_channels=512)

        return rois_output_final
        #返回特征图上统一大小的锚框图像rois_output_final=(Samples=1, self.num_rois=32, rows=14,cols=14, self.nb_channels=512)
        #返回特征图上统一大小的锚框图像rois_output_final=(Samples=1, self.num_rois=32, rows=14,cols=8, self.nb_channels=512)
        #返回特征图上统一大小的锚框图像rois_output_final=(Samples=1, self.num_rois=32, rows=7,cols=4, self.nb_channels=256)
