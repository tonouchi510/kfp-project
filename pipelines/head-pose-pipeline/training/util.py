import tensorflow as tf
from tensorflow.keras import backend as K


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
            K.set_value(self.model.optimizer.lr, LR * ratio)
            print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, LR * ratio))

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
