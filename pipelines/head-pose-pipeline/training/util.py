import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import cv2


class DecayLearningRate(tf.keras.callbacks.Callback):
	def __init__(self, startEpoch):
		self.startEpoch = startEpoch
		# self.isFine = isFine
	
	def on_train_begin(self, logs={}):
		# if self.isFine:
		# 	self.build_ht_copy_W()
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		
		if epoch in self.startEpoch:
			if epoch == 0:
				ratio = 1
			else:
				ratio = 0.1
			LR = K.get_value(self.model.optimizer.lr)
			K.set_value(self.model.optimizer.lr,LR*ratio)
		
		return

	def on_epoch_end(self, epoch, logs={}):	
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		# if self.isFine:
		# 	for i_m, sub_model in enumerate(self.model.layers):
		# 		if i_m>1:
		# 			if sub_model.name != 'ssr_Cap_model' and sub_model.name != 'ssr_F_Cap_model':
		# 				old_W = self.ht[i_m]
		# 				new_W = sub_model.get_weights()
		# 				delta_W = self.list_operation(new_W,old_W,-1)

		# 				sub_W = self.list_operation(old_W,delta_W,0.1)
		# 				sub_model.set_weights(sub_W)

		# 	self.build_ht_copy_W()
		return


def random_crop(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    h = x.shape[0]
    w = x.shape[1]
    out = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    out = cv2.resize(out, (h,w), interpolation=cv2.INTER_CUBIC)
    return out


def random_crop_black(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn,size=1)[0]
    dy_shift = np.random.randint(dn,size=1)[0]
    out = x*0
    out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    
    return out


def random_crop_white(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn,size=1)[0]
    dy_shift = np.random.randint(dn,size=1)[0]
    out = x*0+255
    out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    
    return out


def augment_data(image):
	rand_r = np.random.random()
	if  rand_r < 0.25:
		dn = np.random.randint(15,size=1)[0]+1
		image = random_crop(image,dn)

	elif rand_r >= 0.25 and rand_r < 0.5:
		dn = np.random.randint(15,size=1)[0]+1
		image = random_crop_black(image,dn)

	elif rand_r >= 0.5 and rand_r < 0.75:
		dn = np.random.randint(15,size=1)[0]+1
		image = random_crop_white(image,dn)

	if np.random.random() > 0.3:
		image = tf.contrib.keras.preprocessing.image.random_zoom(image, [0.8,1.2], row_axis=0, col_axis=1, channel_axis=2)

	return image


def data_generator_pose(X,Y,batch_size):

    while True:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y = Y[idxs]
        p,q = [],[]
        for i in range(len(X)):
            p.append(X[i])
            q.append(Y[i])
            if len(p) == batch_size:
                yield augment_data(np.array(p)),np.array(q)
                p,q = [],[]
        if p:
            yield augment_data(np.array(p)),np.array(q)
            p,q = [],[]
