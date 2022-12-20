# coding: utf-8

# In[1]:

from keras.layers import Layer, InputSpec
from keras import initializers, regularizers #Initializer是所有初始化方法的父类，不能直接使用，如果想要定义自己的初始化方法，请继承此类。
#正则项 正则项在优化过程中层的参数或层的激活值添加惩罚项，这些惩罚项将与损失函数一起作为网络的最终优化目标 惩罚项基于层进行惩罚，目前惩罚项的接口与层有关，但 Dense, Conv1D, Conv2D, Conv3D 具有共同的接口。 这些层有三个关键字参数以施加正则项： kernel_regularizer ：施加在权重上的正则项，为 keras.regularizer
from keras import backend as K
#Batch Normalization在卷积神经网络中，是对每个核卷积出来的一个batchsize中所有图片的feature map上的值进行归一化后再进行激活，
#所以叫批标准化。 
#【Tips】BN层的作用 ：1）加速收敛 （2）控制过拟合，可以少用或不用Dropout和正则 （3）降低网络对初始化权重不敏感 （4）允许使用较大的学习率 
#可以把它看做一个自适应重参数化的方法，主要解决训练非常深的模型的困难。当然也不是万能的，对RNN来说，Batch Normalization并没有起到好的效果。 
#主要是把BN变换，置于网络激活函数层的前面。在没有采用BN的时候，激活函数层是这样的：Y=g(WX+b)
#也就是我们希望一个激活函数，比如sigmoid函数s(x)的自变量x是经过BN处理后的结果。因此前向传导的计算公式就应该是：Y=g(BN(WX+b))
#其实因为偏置参数b经过BN层后其实是没有用的，最后也会被均值归一化，当然BN层后面还有个β参数作为偏置项，所以b这个参数就可以不用了。
#因此最后把BN层+激活函数层就变成了：Y=g(BN(WX))
class FixedBatchNormalization(Layer):#固定的批标准化
    #定义(初始化)BN所需参数
    def __init__(self, epsilon=1e-3, axis=-1,#axis为-1,即channel_last
                 weights=None, beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):# **kwargs表示后续可能还有不确定参数

        self.supports_masking = True
        self.beta_init = initializers.get(beta_init) #？？？
        self.gamma_init = initializers.get(gamma_init) #？？？
        self.epsilon = epsilon #？？？
        self.axis = axis #批标准化的轴为通道轴
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        super(FixedBatchNormalization, self).__init__(**kwargs)#super() 函数是用于调用父类(超类)的一个方法。
#定义BN权重不可训练，参数固定
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name),
                                     trainable=False)
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name),
                                    trainable=False)
        self.running_mean = self.add_weight(shape=shape, initializer='zero',
                                            name='{}_running_mean'.format(self.name),
                                            trainable=False)
        self.running_std = self.add_weight(shape=shape, initializer='one',
                                           name='{}_running_std'.format(self.name),
                                           trainable=False)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True
#最后还是看一下模型的classifier部分的关键：ROIPOOLINGCONV
#定义BN方法
    def call(self, x, mask=None):

        assert self.built, 'Layer must be built before being called'
        input_shape = K.int_shape(x)

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        #判断是否对axis是否为-1,即channel_last，对数据BN
        if sorted(reduction_axes) == range(K.ndim(x))[:-1]:
            x_normed = K.batch_normalization(x, self.running_mean, self.running_std,self.beta, self.gamma,epsilon=self.epsilon)
        else:
            # need broadcasting
            broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
            broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            x_normed = K.batch_normalization(
                x, broadcast_running_mean, broadcast_running_std,
                broadcast_beta, broadcast_gamma,
                epsilon=self.epsilon)

        return x_normed

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None}
        base_config = super(FixedBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
#定义了类似Keras中的Batch Normalization的参数，在后续的resnet有使用如：

