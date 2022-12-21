from keras import backend as K
from keras.metrics import categorical_crossentropy

if K.image_data_format() == 'channels_last':
	import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0  # lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def rpn_loss_regr(num_anchors): #num_anchors=32
	def rpn_loss_regr_fixed_num(y_true, y_pred): #回归标记值：y_true, 回归预测值：y_pred如何而来？？？？？？？
		#y_true=Y[1]=np.copy(y_rpn_regr)=[Samples=1, rows=32, cols=40, 9*(4+1)+9*(4+1)=90]  后9*(4+1)=(tx, ty, tw, th, td)
		# 前9*(4+1)个是锚框与标记框的交并比(即分类：1表示正锚框，0表示负锚框/背景)；后9*(4+1)个是锚框与标记框的回归参数和距离
		"""#y_pred=x_regr=(Samples=1, rows=32, cols=40, filters=(4+1)*num_anchors=45)——>(Samples=1, rows=32, cols=40, filters=4 *num_anchors=36)"""
		# TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int32 of argument 'x'.
		y_true1 = K.cast(y_true, 'float32')
		if K.image_data_format() == 'channels_first':
			x = y_true1[:, 4 * num_anchors:, :, :] - y_pred
			x_abs = K.abs(x)
			x_bool = K.less_equal(x_abs, 1.0)
			return lambda_rpn_regr * K.sum(y_true1[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)))\
				   / K.sum(epsilon + y_true1[:, :4 * num_anchors, :, :])
		else:
			x = y_true1[:, :, :, 4 * num_anchors:] - y_pred #计算锚框回归值与预测值的差值：前9*(4+1)个是锚框与标记框的交并比(即分类：1表示正锚框，0表示负锚框/背景)
			x_abs = K.abs(x) #求锚框回归值与预测值的差值的绝对值
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)  #K.less_equal()求小于等于1的数(即预测很靠近标记的哪些预测值)，返回一个与x_abs相同维度的布尔矩阵
			#返回L1：只对正样本计算，负样本将被x_bool干掉
			return lambda_rpn_regr * K.sum(y_true1[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) \
				   / K.sum(epsilon + y_true1[:, :, :, :4 * num_anchors])

	return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors): #num_anchors=3*3=9
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		# 标记Y=[np.copy(y_rpn_cls), np.copy(y_rpn_regr)]
		# 锚框标记:Y=[np.copy(y_rpn_cls), np.copy(y_rpn_regr)]
		# 锚框分类标记信息:Y[0]=np.copy(y_rpn_cls)=[Samples=1,rows=32,cols=40,(3*3)+(3*3)=18] 前一个(3*3)表示锚框(每个特征像素点有9个)的有效性(1表示有效，0表示无效)；
		# 后一个(3*3)表示锚框与标记框的交并比(即分类：1表示正锚框，0表示负锚框/背景)
		#y_true=Y[1]=np.copy(y_rpn_cls)=[Samples=1,rows=32,cols=40,(3*3)+(3*3)=18]
		#y_pred=x_class .shape=(Samples=1,rows=32,cols=40, filters=num_anchors=9)
		y_true1 = K.cast(y_true, 'float32')
		if K.image_data_format() == 'channels_last':
			return lambda_rpn_class * K.sum(y_true1[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true1[:, :, :, num_anchors:])) / K.sum(epsilon + y_true1[:, :, :, :num_anchors])
		else:
			return lambda_rpn_class * K.sum(y_true1[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true1[:, num_anchors:, :, :])) / K.sum(epsilon + y_true1[:, :num_anchors, :, :])

	return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):
		# 标记Y=[np.copy(y_rpn_cls), np.copy(y_rpn_regr)]
		# 锚框标记Y=[np.copy(y_rpn_cls), np.copy(y_rpn_regr)]
		# 分类信息:Y[0]=np.copy(y_rpn_cls)=[1,32,40,(3*3)+(3*3)=18] 前一个(3*3)表示锚框(每个特征像素点有9个)的有效性(1表示有效，0表示无效)；
		# 后一个(3*3)表示锚框与标记框的交并比(即分类：1表示正锚框，0表示负锚框/背景)
		#y_true=Y[1]=np.copy(y_rpn_regr)=[1,32,40,9*(4+1)+9*(4+1)=90]
		#y_pred=x_class .shape=(Samples=1, 32, 40, filters=num_anchors=9)
		num5_regr = 5  #锚框与标记框的回归参数(tx, ty, tw, th, td)共5个
		y_true1 = K.cast(y_true, 'float32')
		x = y_true1[:, :, num5_regr*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * K.sum(y_true1[:, :, :num5_regr*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true1[:, :, :num5_regr*num_classes])
	return class_loss_regr_fixed_num

def class_loss_dis(num_classes):
	def class_loss_dis_fixed_num(y_true, y_pred):
		# 标记Y=[np.copy(y_rpn_cls), np.copy(y_rpn_regr)]
		# 锚框标记Y=[np.copy(y_rpn_cls), np.copy(y_rpn_regr)]
		# 分类信息:Y[0]=np.copy(y_rpn_cls)=[1,32,40,(3*3)+(3*3)=18] 前一个(3*3)表示锚框(每个特征像素点有9个)的有效性(1表示有效，0表示无效)；
		# 后一个(3*3)表示锚框与标记框的交并比(即分类：1表示正锚框，0表示负锚框/背景)
		#y_true=Y[1]=np.copy(y_rpn_regr)=[1,32,40,9*(4+1)+9*(4+1)=90]
		#y_pred=x_class .shape=(Samples=1, 32, 40, filters=num_anchors=9)
		num5_regr = 5  #锚框与标记框的回归参数(tx, ty, tw, th, td)共5个
		y_true1 = K.cast(y_true, 'float32')
		x = y_true1[:, :, num5_regr*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * K.sum(y_true1[:, :, :num5_regr*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :num5_regr*num_classes])
	return class_loss_dis_fixed_num


def class_loss_cls(y_true, y_pred):
	# TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int32 of argument 'x'.
	y_true1 = K.cast(y_true, 'float32')
	return lambda_cls_class * K.mean(categorical_crossentropy(y_true1[0, :, :], y_pred[0, :, :]))
