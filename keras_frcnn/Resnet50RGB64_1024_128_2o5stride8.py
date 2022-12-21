# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) Adapted from code contributed by BigMoyan.
'''

from __future__ import print_function
from __future__ import absolute_import

from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D,\
                          AveragePooling2D, TimeDistributed

from keras import backend as K  #我们的backend 是 TensorFlow
#Keras是一个模型级的库，提供了快速构建深度学习网络的模块。Keras并不处理如张量乘法、卷积等底层操作。这些操作依赖于某种特定的、优化良好的张量操作库。Keras依赖于处理张量的库就称为“后端引擎”。Keras提供了两种后端引擎Theano/Tensorflow，并将其函数统一封装，使得用户可以以同一个接口调用不同后端引擎的函数
#Theano是一个开源的符号主义张量操作框架，由蒙特利尔大学LISA/MILA实验室开发
#TensorFlow是一个符号主义的张量操作框架，由Google开发
from keras_frcnn.roi_pooling_conv_distance import RoiPoolingConv
from keras_frcnn.fixed_batch_normalization import FixedBatchNormalization

# keras.json 细节 { 'image_dim_ordering': 'tf', 'epsilon': 1e-07, 'floatx': 'float32', 'backend': 'tensorflow'} 可以更改以上~/.keras/keras.json中的配置：image_dim_ordering：字符串，'tf'或'th'，该选项指定了Keras将要使用的维度顺序，可通过keras.backend.image_dim_ordering()来获取当前的维度顺序。对2D数据来说，tf假定维度顺序为(rows,cols,channels)而th假定维度顺序为(channels, rows, cols)。对3D数据而言，tf假定(conv_dim1, conv_dim2, conv_dim3, channels)，th则是(channels, conv_dim1, conv_dim2, conv_dim3)；epsilon：浮点数，防止除0错误的小数字；floatx：字符串，'float16', 'float32', 'float64'之一，为浮点数精度；backend：字符串，所使用的后端，为'tensorflow'或'theano'

def get_weight_path():#定义一个获取权重路径的函数
    if K.image_data_format() == "channels_first": #获取当前的维度顺序：th则是(channels, conv_dim1, conv_dim2, conv_dim3)
        return 'resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
    else: #获取当前的维度顺序：tf假定(conv_dim1, conv_dim2, conv_dim3, channels) conv_dim1表示图片张数？
        return 'resnet50_weights_tf_dim_ordering_kernels_notop1.h5'

#获取图像经过步长为2的4次不填充的卷积后输出特征图尺寸
def get_img_output_length(width, height):  #前处理后图像：width=640, height=512
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        #filter_sizes = [7, 3, 1, 1] #  4次池化 downscale=16
        filter_sizes = [7, 3, 1]   #  8次池化  downscale=8
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride #舍余取商
        return input_length

    return get_output_length(width), get_output_length(height)

#1*1，3*3，1*1的三层stride=(1,1)卷积，最后直接输出x+input作为输出。
def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):#恒等映射块
    #x = identity_block(x, 3, [32, 32, 128], stage=2, block='b', trainable=trainable)
    #输入：input_tensor.shape=x.shape=(Samples=1, 128, 160, filters=32)
    nb_filter1, nb_filter2, nb_filter3 = filters

    if K.image_data_format() == "channels_last":
        bn_axis = 3 #bn_axis代表批量正则化的轴在输入的第3维度
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch' #残差网络卷积层基础块名
    bn_name_base = 'bn' + str(stage) + block + '_branch' #残差网络批量正则化层基础块名

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor]) 
    # 计算一个列表的输入张量的和。 相加层接受一个列表的张量,所有的张量必须有相同的输入尺寸,然后返回一个张量(和输入张量尺寸相同)。
    x = Activation('relu')(x)
    return x #返回：x.shape=(Samples=1,  128, 160, filters=nb_filter3=128)


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True): # identity block time distributed 恒等映射块时间分布
    #x = identity_block_td(x, 3, [256, 256, 1024], stage=5, block='b', trainable=trainable)
    #输入：input_tensor.shape=x.shape=(Samples=1, num_rois=32, 7, 7, filters=32)
    #kernel_size=3

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_data_format() == "channels_last":
        bn_axis = 3 #bn_axis代表批量正则化的轴在输入的第3维度
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    #时间分布的2D卷积输出：x.shape=(Samples=1, num_rois=32, 7, 7, filters=nb_filter1=256)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    #时间分布的固定批标准化输出：x.shape=(Samples=1, num_rois=32, 7, 7, filters=nb_filter1=256)
    x = Activation('relu')(x)
    #relu激活输出：x.shape=(Samples=1, num_rois=32, 7, 7, filters=nb_filter1=256)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    #时间分布的2D卷积输出：x.shape=(Samples=1, num_rois=32, 7, 7, filters=nb_filter2=256)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    #时间分布的固定批标准化输出：x.shape=(Samples=1, num_rois=32, 7, 7, filters=nb_filter1=256)
    x = Activation('relu')(x)
    #relu激活输出：x.shape=(Samples=1, num_rois=32, 7, 7, filters=nb_filter2=256)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    #时间分布的2D卷积输出：x.shape=(Samples=1, num_rois=32, 7, 7, filters=nb_filter3=1024)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)
    #时间分布的固定批标准化输出：x.shape=(Samples=1, num_rois=32, 7, 7, filters=nb_filter3=1024)
    x = Add()([x, input_tensor]) #输入shortcut与残差x相加输出：x.shape=(Samples=1, num_rois=32, 7, 7, filters=nb_filter3=1024)
    x = Activation('relu')(x) #relu激活输出：x.shape=(Samples=1, num_rois=32, 7, 7, filters=nb_filter3=1024)

    return x  #返回：x.shape=(Samples=1, num_rois=32, 7, 7, filters=nb_filter3=1024)

#1*1 3*3 1*1 的三层卷积模块,stride=(2*2)，与identity block不同的是，conv block创建了shortcut = bn(conv(input) )+ 输出x，作为输出。而indentity block直接加上input，没有再次conv和bn的过程。
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True): #定义前向通道的卷积层块
    #x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1), trainable=trainable)
    #输入input_tensor.shape=x.shape=(Samples=1,128, 160, filters=16)
    nb_filter1, nb_filter2, nb_filter3 = filters #nb_filter1=32, nb_filter2=32, nb_filter3=128
    # if K.image_dim_ordering() == 'tf':
    if K.image_data_format() == "channels_last":
        bn_axis = 3 #bn_axis代表批量正则化的轴在输入的第3维度
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch' #残差网络卷积层基础块名
    bn_name_base = 'bn' + str(stage) + block + '_branch'  #残差网络批量正则化层基础块名

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    #残差2D卷积层res2a_branch2a输出：x.shape=(Samples=1, 128, 160, filters=nb_filter1=32)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    #残差固定批量标准化层bn2a_branch2a输出：x.shape=(Samples=1, 128, 160, filters=nb_filter1=32)
    x = Activation('relu')(x)
    #残差relu激活层activation_2输出：x.shape=(Samples=1, 128, 160, filters=nb_filter1=32)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',trainable=trainable)(x)
    #残差2D卷积层res2a_branch2b输出：x.shape=(Samples=1, 128, 160, filters=nb_filter2=32)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    #残差固定批量标准化层bn2a_branch2b输出：x.shape=(Samples=1, 128, 160, filters=nb_filter2=32)
    x = Activation('relu')(x)
    #残差relu激活层activation_3输出：x.shape=(Samples=1, 128, 160, filters=nb_filter2=32)
    
    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    #残差2D卷积层res2a_branch2c输出：x.shape=(Samples=1, 128, 160, filters=nb_filter3=32)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    #残差固定批量标准化层bn2a_branch2c输出：x.shape=(Samples=1, 128, 160, filters=nb_filter3=32)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(input_tensor)
    #输入原图2D卷积层res2a_branch1输出：x.shape=(Samples=1, 128, 160, filters=nb_filter3=32)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    #输入原图固定批量标准化层bn2a_branch1输出：x.shape=(Samples=1, 128, 160, filters=nb_filter3=32)
    
    x = Add()([x, shortcut]) 
    #求和层add_1输出：x.shape=(Samples=1, 128, 160, filters=nb_filter3=32)：残差F(x)=x和原图像x=shortcut求和x=H(x)=F(x)+x的计算
    x = Activation('relu')(x) #relu激活层activation_4输出：x.shape=(Samples=1, 128, 160, filters=nb_filter3=32)
    return x #返回：x.shape=(Samples=1, 128, 160, filters=nb_filter3=32)


def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    # conv block time distributed 时间分布卷积块
    #x = conv_block_td(x, 3, [256, 256, 1024], stage=5, block='a', input_shape=input_shape, strides=(2, 2),trainable=trainable) 
    #输入变量：input_tensor.shape=x.shape=(Samples=1, self.num_rois=32, rows=14, cols=14, self.nb_channels=512)
    #input_shape=(num_rois=32, rows=14, cols=14, filters=512)/(num_rois=32, rows=14, cols=8, filters=512)
    nb_filter1, nb_filter2, nb_filter3 = filters #filters=[256, 256, 64]
    if K.image_data_format() == "channels_last":
        bn_axis = 3 #bn_axis代表批量正则化的轴在输入的第3维度
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    #输出：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=256)/(Samples=1, num_rois=32, rows=7, cols=4, filters=256)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    #输出：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=256)/(Samples=1, num_rois=32, rows=7, cols=4, filters=256)
    x = Activation('relu')(x)
    #输出：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=256)/(Samples=1, num_rois=32, rows=7, cols=4, filters=256)
    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    #输出：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=256)/(Samples=1, num_rois=32, rows=7, cols=4, filters=256)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    #输出：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=256)/(Samples=1, num_rois=32, rows=7, cols=4, filters=256)
    x = Activation('relu')(x)
    #输出：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=256)/(Samples=1, num_rois=32, rows=7, cols=4, filters=256)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    #输出：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=64)/(Samples=1, num_rois=32, rows=7, cols=4, filters=64)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)
    #输出：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=64)/(Samples=1, num_rois=32, rows=7, cols=4, filters=64)
    shortcut = TimeDistributed(Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    #输出：shortcut.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=64)/(Samples=1, num_rois=32, rows=7, cols=4, filters=64)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)
    #输出：shortcut.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=64)/(Samples=1, num_rois=32, rows=7, cols=4, filters=64)

    x = Add()([x, shortcut]) #输入shortcut与残差x相加输出：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=64)/(Samples=1, num_rois=32, rows=7, cols=4, filters=64)
    x = Activation('relu')(x) #relu激活输出：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=64)/(Samples=1, num_rois=32, rows=7, cols=4, filters=64)
    return x #返回：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=64)/(Samples=1, num_rois=32, rows=7, cols=4, filters=64)

#定义共享网络层(不可训练)
def nn_base(input_tensor=None, trainable=False):
    #共享基础网络，输入为：缩放后的整张图片input_tensor.shape=img_input.shape=(Samples=1, 512, 640, channels=1)，trainable=True表示可训练。
    # Determine proper input shape确定正确的输入形状
    # if K.image_dim_ordering() == 'th':
    if K.image_data_format() == "channels_first":
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):#不是K.is_keras_tensor重新定义输入图像张量shape
            img_input = Input(tensor=input_tensor, shape=input_shape) 
            #tf假定img_input.shape()=（samples，first_axis_to_pad，second_axis_to_pad, channels） 
        else:#是K.is_keras_tensor输入图像张量=input_tensor
            img_input = input_tensor 
            #输入层input_1的维度：img_input.shape=input_tensor.shape=img_input.shape=(Samples=1, 512, 640, channels=1)

    # if K.image_dim_ordering() == 'tf':
    if K.image_data_format() == "channels_last":
        bn_axis = 3 #bn_axis代表批量正则化的轴在输入的第3维度
    else:
        bn_axis = 1
    #keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format=None)。
    x = ZeroPadding2D((3, 3))(img_input)#输入层：在first_axis_to_pad(图像的高)和second_axis_to_pad(图像的宽)两轴的起始和结束位置填充3排0
    #2D零填充层zero_padding2d_1输出：x.shape=(Samples=1,518, 646, channels=1)

    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x) #TODO：***第1 次最大池化.
    #x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x)
    #2D卷积层conv1输出：x.shape=(Samples=1,256, 320, filters=16)
    x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    #固定批标准化层bn_conv1输出：x.shape=(Samples=1,256, 320, filters=16)
    x = Activation('relu')(x)
    #relu激活层activation_1输出：x.shape=(Samples=1,256, 320, filters=16)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)   #TODO：***第2 次最大池化.
    #高、宽轴步长均为2的3x3最大池化MaxPooling2D层max_pooling2d_1输出：x.shape=(Samples=1,128, 160, filters=16)
    #下面是残差F(x)和原图像x求和H(x)=F(x)+x的计算
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=trainable)
    #返回：x.shape=(Samples=1, 128, 160, filters=nb_filter3=32)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=trainable) #跟conv_block的区别在于这个过程输出通道数没有任何变化
    #返回：x.shape=(Samples=1, 128, 160, filters=nb_filter3=32)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)
    #返回：x.shape=(Samples=1, 128, 160, filters=nb_filter3=32)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=trainable) #TODO：***第3 次最大池化.
    #返回：x.shape=(Samples=1, 64, 80, filters=nb_filter3=128)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
    
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)
    #返回：x.shape=(Samples=1, 64, 80, filters=nb_filter3=128)
    
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=(1, 1), trainable=trainable) #TODO：***第4 次最大池化。去除此次下采样，将特征图缩放系数由16->8
    #返回：x.shape=(Samples=1, rows=32, cols=40, filters=nb_filter3=256)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)
    #返回：x.shape=(Samples=1, rows=32, cols=40, filters=nb_filter3=512)
    
    return x #返回基础网络最后输出为卷积后的特征图base_layers.shape=x.shape=(Samples=1, rows=32, cols=40, filters=512)


def classifier_layers(x, input_shape, trainable=False):
    #输入变量：x.shape=out_roi_pool.shape=(Samples=1, self.num_rois=32, rows=14, cols=14, self.nb_channels=512)
    #input_shape=(num_rois=32, rows=14, cols=14, filters=512)/(num_rois=32, rows=14, cols=8, filters=512)
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    # (hence a smaller stride in the region that follows the ROI pool)
    if K.backend() == 'tensorflow':
        x = conv_block_td(x, 3, [512, 512, 128], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
        #返回：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=64)/(Samples=1, num_rois=32, rows=7, cols=4, filters=64)
    elif K.backend() == 'theano':
        x = conv_block_td(x, 3, [512, 512, 128], stage=5, block='a', input_shape=input_shape, strides=(1, 1), trainable=trainable)

    x = identity_block_td(x, 3, [512, 512, 128], stage=5, block='b', trainable=trainable)
    #输出：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=64)/(Samples=1, num_rois=32, rows=7, cols=4, filters=64)
    x = identity_block_td(x, 3, [512, 512, 128], stage=5, block='c', trainable=trainable)
    #输出：x.shape=(Samples=1, num_rois=32, rows=7, cols=7, filters=64)/(Samples=1, num_rois=32, rows=7, cols=4, filters=64)
    #x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)#该包装器可以把一个层应用到输入的每一个时间步上
    x = TimeDistributed(AveragePooling2D((7, 4)), name='avg_pool')(x)#该包装器可以把一个层应用到输入的每一个时间步上
    #输出：x.shape=(Samples=1, num_rois=32, rows=1, cols=1, filters=64)

    return x #输出：x.shape=(Samples=1, num_rois=32, rows=1, cols=1, filters=64)

#定义RPN网络,trainable=False??
def rpn(base_layers, num_anchors):
    #输入变量：base_layers=shared_layers=(shape=(Samples=1, rows=32, cols=40, filters=512));特征图上每个像素点上的锚框个数:num_anchors=3*3=9
    x = Convolution2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    #输出：x.shape=(Samples=1, rows=32, cols=40, filters=256)
    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    #输出为RPN锚框分类预测概率：x_class .shape=(Samples=1, rows=32, cols=40, filters=num_anchors=9)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    #输出为RPN锚框回归坐标预测修正参数：x_regr.shape=(Samples=1, rows=32, cols=40, filters=(4+1)*num_anchors=45)增加距离信息

    #return [x_class, x_regr, base_layers] #为什么返回去就变成了[x_class, x_regr]？？？？？？？？？？？？？？？？？？
    return [x_class, x_regr, base_layers] #为什么返回去就变成了[x_class, x_regr]？？？？？？？？？？？？？？？？？？
    # 返回一个列表rpn=[x_class, x_regr, base_layers]：
    # RPN锚框分类预测概率：x_class  =(Samples=1, rows=32, cols=40, filters=num_anchors=9)
    # RPN锚框回归坐标预测修正参数：x_regr =(Samples=1, rows=32,cols=40, filters=(4+1)*num_anchors=45)
    # 基础网络输出——特征图层: base_layers =(Samples=1, rows=32, cols=40, filters=512)
    
#定义classifier网络
def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    #输入为变量列表[img_input, roi_input]：base_layers.shape=(Samples=1, rows=32, cols=40, channels=512)；
    #input_rois.shape=roi_input.shape=(Samples=1, ?=32, (4+1))
    #输入为常参：num_rois=32；nb_classes=21？？？？？
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    #在theano上的编译时间往往非常高，因此我们使用较小的ROI池区域来解决
    if K.backend() == 'tensorflow':
        row_pool_size = 14
        col_pool_size = 8
        #input_shape = (num_rois, 14, 14, 512) #num_rois=cfg.num_rois=32 512为nn_base输出通道
        input_shape = (num_rois, row_pool_size, col_pool_size, 1024) #num_rois=cfg.num_rois=32 512为nn_base输出通道
    elif K.backend() == 'theano':
        row_pool_size = 7
        col_pool_size = 7
        input_shape = (num_rois, 1024, row_pool_size, col_pool_size)

    out_roi_pool = RoiPoolingConv(row_pool_size, col_pool_size, num_rois)([base_layers, input_rois]) #调用RoI池卷积类，([base_layers, input_rois])类初始化变量
    #RoiPoolingConv类初始化输入常参：row_pool_size=14, col_pool_size=8, num_rois=32；
    #RoiPoolingConv类输入变量：x=[base_layers, input_rois]：base_layers.shape=shared_layers.shape=(Samples=1, rows=32, cols=40, filters=512)；
    #input_rois=(Samples=1, num_rois=32, (4+1))
    #输出特征图上统一大小的的RoI图像：out_roi_pool=rois_output_final=(Samples=1, self.num_rois=32, rows=14, cols=8, self.nb_channels=512)

    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)#对RoI分类
    #输入变量：out_roi_pool.shape=(Samples=1, num_rois=32, rows=14, cols=14, self.nb_channels=512)/(Samples=1, num_rois=32, rows=14, cols=8, self.nb_channels=512)
    #input_shape=(num_rois=32, rows=14, cols=14, filters=512)/input_shape=(num_rois=32, rows=14, cols=8, filters=512)
    #返回：out.shape=(Samples=1, num_rois=32, rows=1, cols=1, filters=64)
    out = TimeDistributed(Flatten())(out) #返回：out.shape=(Samples=1, num_rois=32, 1*1*filters=1*1*64=64)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    #输出：out_class.shape=(Samples=1, num_rois=32, nb_classes=2)
    # note: no regression target for bg class 注意：没有bg类的回归目标
    out_regr = TimeDistributed(Dense((4+1) * (nb_classes - 1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    #输出：out_regr.shape=(Samples=1, num_rois=32, (4+1)* (nb_classes - 1)=(4+1))
    return [out_class, out_regr]
    #返回：out_class.shape=(Samples=1, num_rois=32, nb_classes=2)；
    # out_regr.shape=(Samples=1, num_rois=32, (4+1) * (nb_classes - 1)=(4+1))
