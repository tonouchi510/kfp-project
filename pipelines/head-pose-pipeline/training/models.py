import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from capsulelayers import CapsuleLayer
from capsulelayers import MatMulLayer
from loupe_keras import NetVLAD

sys.setrecursionlimit(2**20)
np.random.seed(2**10)

# Custom layers
# Note - Usage of Lambda layers prevent the convertion
# and the optimizations by the underlying math engine (tensorflow in this case)


class SSRLayer(tf.keras.layers.Layer):
    def __init__(self, s1, s2, s3, lambda_d, **kwargs):
        super(SSRLayer, self).__init__(**kwargs)
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.lambda_d = lambda_d
        self.trainable = False

    def call(self, inputs):
        x = inputs

        a = x[0][:, :, 0] * 0
        b = x[0][:, :, 0] * 0
        c = x[0][:, :, 0] * 0

        s1 = self.s1
        s2 = self.s2
        s3 = self.s3
        lambda_d = self.lambda_d

        di = s1 // 2
        dj = s2 // 2
        dk = s3 // 2

        V = 99

        for i in range(0, s1):
            a = a + (i - di + x[6]) * x[0][:, :, i]
        a = a / (s1 * (1 + lambda_d * x[3]))

        for j in range(0, s2):
            b = b + (j - dj + x[7]) * x[1][:, :, j]
        b = b / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4]))

        for k in range(0, s3):
            c = c + (k - dk + x[8]) * x[2][:, :, k]
        c = c / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4])) / (s3 * (1 + lambda_d * x[5]))

        pred = (a + b + c) * V

        return pred

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 3)

    def get_config(self):
        config = {
            "s1": self.s1,
            "s2": self.s2,
            "s3": self.s3,
            "lambda_d": self.lambda_d,
        }
        base_config = super(SSRLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeatSliceLayer(tf.keras.layers.Layer):
    def __init__(self, start_index, end_index, **kwargs):
        super(FeatSliceLayer, self).__init__(**kwargs)
        self.start_index = start_index
        self.end_index = end_index
        self.trainable = False

    def call(self, inputs):
        return inputs[:, self.start_index:self.end_index]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.end_index - self.start_index)

    def get_config(self):
        config = {"start_index": self.start_index, "end_index": self.end_index}
        base_config = super(FeatSliceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MomentsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MomentsLayer, self).__init__(**kwargs)
        self.trainable = False

    def call(self, inputs):
        _, var = tf.nn.moments(inputs, axes=-1)
        return var

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class MatrixMultiplyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MatrixMultiplyLayer, self).__init__(**kwargs)
        self.trainable = False

    def call(self, inputs):
        x1, x2 = inputs
        # TODO: add some asserts on the inputs
        # it is expected the shape of inputs are
        # arranged to be able to perform the matrix multiplication
        return tf.matmul(x1, x2)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], input_shapes[1][-1])


class MatrixNormLayer(tf.keras.layers.Layer):
    def __init__(self, tile_count, **kwargs):
        super(MatrixNormLayer, self).__init__(**kwargs)
        self.trainable = False
        self.tile_count = tile_count

    def call(self, input):
        sum = K.sum(input, axis=-1, keepdims=True)
        tiled = K.tile(sum, (1, 1, self.tile_count))
        return tiled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.tile_count)

    def get_config(self):
        config = {"tile_count": self.tile_count}
        base_config = super(MatrixNormLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PrimCapsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PrimCapsLayer, self).__init__(**kwargs)
        self.trainable = False

    def call(self, inputs):
        x1, x2, norm = inputs
        return tf.matmul(x1, x2) / norm

    def compute_output_shape(self, input_shapes):
        return input_shapes[-1]


class AggregatedFeatureExtractionLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsule, **kwargs):
        super(AggregatedFeatureExtractionLayer, self).__init__(**kwargs)
        self.trainable = False
        self.num_capsule = num_capsule

    def call(self, input):
        s1_a = 0
        s1_b = self.num_capsule // 3
        feat_s1_div = input[:, s1_a:s1_b, :]
        s2_a = self.num_capsule // 3
        s2_b = 2 * self.num_capsule // 3
        feat_s2_div = input[:, s2_a:s2_b, :]
        s3_a = 2 * self.num_capsule // 3
        s3_b = self.num_capsule
        feat_s3_div = input[:, s3_a:s3_b, :]

        return [feat_s1_div, feat_s2_div, feat_s3_div]

    def compute_output_shape(self, input_shape):
        last_dim = input_shape[-1]
        partition = self.num_capsule // 3
        return [
            (input_shape[0], partition, last_dim),
            (input_shape[0], partition, last_dim),
            (input_shape[0], partition, last_dim),
        ]

    def get_config(self):
        config = {"num_capsule": self.num_capsule}
        base_config = super(AggregatedFeatureExtractionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BaseFSANet(object):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        self._channel_axis = 3 if K.image_data_format() == "channels_last" else 1

        if self._channel_axis == 1:
            logging.debug("image_dim_ordering = 'th'")
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule / 3) * self.dim_capsule
        self.map_xy_size = int(8 * image_size / 64)

        self.is_fc_model = False
        self.is_noS_model = False
        self.is_varS_model = False

    def _convBlock(self, x, num_filters, activation, kernel_size=(3, 3)):
        x = tf.keras.layers.SeparableConv2D(num_filters, kernel_size, padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.Activation(activation)(x)
        return x

    def ssr_G_model_build(self, img_inputs):
        # -------------------------------------------------------------------------------------------------------------------------
        x = self._convBlock(img_inputs, num_filters=16, activation="relu")
        x_layer1 = tf.keras.layers.AveragePooling2D((2, 2))(x)
        x = self._convBlock(x_layer1, num_filters=32, activation="relu")
        x = self._convBlock(x, num_filters=32, activation="relu")
        x_layer2 = tf.keras.layers.AveragePooling2D((2, 2))(x)
        x = self._convBlock(x_layer2, num_filters=64, activation="relu")
        x = self._convBlock(x, num_filters=64, activation="relu")
        x_layer3 = tf.keras.layers.AveragePooling2D((2, 2))(x)
        x = self._convBlock(x_layer3, num_filters=128, activation="relu")
        x_layer4 = self._convBlock(x, num_filters=128, activation="relu")
        # -------------------------------------------------------------------------------------------------------------------------
        s = self._convBlock(img_inputs, num_filters=16, activation="tanh")
        s_layer1 = tf.keras.layers.MaxPooling2D((2, 2))(s)
        s = self._convBlock(s_layer1, num_filters=32, activation="tanh")
        s = self._convBlock(s, num_filters=32, activation="tanh")
        s_layer2 = tf.keras.layers.MaxPooling2D((2, 2))(s)
        s = self._convBlock(s_layer2, num_filters=64, activation="tanh")
        s = self._convBlock(s, num_filters=64, activation="tanh")
        s_layer3 = tf.keras.layers.MaxPooling2D((2, 2))(s)
        s = self._convBlock(s_layer3, num_filters=128, activation="tanh")
        s_layer4 = self._convBlock(s, num_filters=128, activation="tanh")
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer4 = tf.keras.layers.Conv2D(64, (1, 1), activation="tanh")(s_layer4)
        x_layer4 = tf.keras.layers.Conv2D(64, (1, 1), activation="relu")(x_layer4)

        feat_s1_pre = tf.keras.layers.Multiply()([s_layer4, x_layer4])
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer3 = tf.keras.layers.Conv2D(64, (1, 1), activation="tanh")(s_layer3)
        x_layer3 = tf.keras.layers.Conv2D(64, (1, 1), activation="relu")(x_layer3)

        feat_s2_pre = tf.keras.layers.Multiply()([s_layer3, x_layer3])
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer2 = tf.keras.layers.Conv2D(64, (1, 1), activation="tanh")(s_layer2)
        x_layer2 = tf.keras.layers.Conv2D(64, (1, 1), activation="relu")(x_layer2)

        feat_s3_pre = tf.keras.layers.Multiply()([s_layer2, x_layer2])
        # -------------------------------------------------------------------------------------------------------------------------

        # Spatial Pyramid Pooling
        # feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)
        # feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
        # feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
        # feat_s1_pre = Globaltf.keras.layers.AveragePooling2D()(feat_s1_pre)
        # feat_s2_pre = Globaltf.keras.layers.AveragePooling2D()(feat_s2_pre)
        feat_s3_pre = tf.keras.layers.AveragePooling2D((2, 2))(
            feat_s3_pre
        )  # make sure (8x8x64) feature maps

        return tf.keras.models.Model(
            inputs=img_inputs,
            outputs=[feat_s1_pre, feat_s2_pre, feat_s3_pre],
            name="ssr_G_model",
        )

    def ssr_F_model_build(self, feat_dim, name_F):
        input_s1_pre = tf.keras.layers.Input((feat_dim,))
        input_s2_pre = tf.keras.layers.Input((feat_dim,))
        input_s3_pre = tf.keras.layers.Input((feat_dim,))

        def _process_input(stage_index, stage_num, num_classes, input_s_pre):
            feat_delta_s = FeatSliceLayer(0, 4)(input_s_pre)
            delta_s = tf.keras.layers.Dense(
                num_classes, activation="tanh", name=f"delta_s{stage_index}"
            )(feat_delta_s)

            feat_local_s = FeatSliceLayer(4, 8)(input_s_pre)
            local_s = tf.keras.layers.Dense(
                units=num_classes,
                activation="tanh",
                name=f"local_delta_stage{stage_index}",
            )(feat_local_s)

            feat_pred_s = FeatSliceLayer(8, 16)(input_s_pre)
            feat_pred_s = tf.keras.layers.Dense(
                stage_num * num_classes, activation="relu"
            )(feat_pred_s)
            pred_s = tf.keras.layers.Reshape((num_classes, stage_num))(feat_pred_s)

            return delta_s, local_s, pred_s

        delta_s1, local_s1, pred_s1 = _process_input(
            1, self.stage_num[0], self.num_classes, input_s1_pre
        )
        delta_s2, local_s2, pred_s2 = _process_input(
            2, self.stage_num[1], self.num_classes, input_s2_pre
        )
        delta_s3, local_s3, pred_s3 = _process_input(
            3, self.stage_num[2], self.num_classes, input_s3_pre
        )

        return tf.keras.models.Model(
            inputs=[input_s1_pre, input_s2_pre, input_s3_pre],
            outputs=[
                pred_s1,
                pred_s2,
                pred_s3,
                delta_s1,
                delta_s2,
                delta_s3,
                local_s1,
                local_s2,
                local_s3,
            ],
            name=name_F,
        )

    def ssr_FC_model_build(self, feat_dim, name_F):
        input_s1_pre = tf.keras.layers.Input((feat_dim,))
        input_s2_pre = tf.keras.layers.Input((feat_dim,))
        input_s3_pre = tf.keras.layers.Input((feat_dim,))

        def _process_input(stage_index, stage_num, num_classes, input_s_pre):
            feat_delta_s = tf.keras.layers.Dense(2 * num_classes, activation="tanh")(
                input_s_pre
            )
            delta_s = tf.keras.layers.Dense(
                num_classes, activation="tanh", name=f"delta_s{stage_index}"
            )(feat_delta_s)

            feat_local_s = tf.keras.layers.Dense(2 * num_classes, activation="tanh")(
                input_s_pre
            )
            local_s = tf.keras.layers.Dense(
                units=num_classes,
                activation="tanh",
                name=f"local_delta_stage{stage_index}",
            )(feat_local_s)

            feat_pred_s = tf.keras.layers.Dense(
                stage_num * num_classes, activation="relu"
            )(input_s_pre)
            pred_s = tf.keras.layers.Reshape((num_classes, stage_num))(feat_pred_s)

            return delta_s, local_s, pred_s

        delta_s1, local_s1, pred_s1 = _process_input(
            1, self.stage_num[0], self.num_classes, input_s1_pre
        )
        delta_s2, local_s2, pred_s2 = _process_input(
            2, self.stage_num[1], self.num_classes, input_s2_pre
        )
        delta_s3, local_s3, pred_s3 = _process_input(
            3, self.stage_num[2], self.num_classes, input_s3_pre
        )

        return tf.keras.models.Model(
            inputs=[input_s1_pre, input_s2_pre, input_s3_pre],
            outputs=[
                pred_s1,
                pred_s2,
                pred_s3,
                delta_s1,
                delta_s2,
                delta_s3,
                local_s1,
                local_s2,
                local_s3,
            ],
            name=name_F,
        )

    def ssr_feat_S_model_build(self, m_dim):
        input_preS = tf.keras.layers.Input((self.map_xy_size, self.map_xy_size, 64))

        if self.is_varS_model:
            feat_preS = MomentsLayer()(input_preS)
        else:
            feat_preS = tf.keras.layers.Conv2D(
                1, (1, 1), padding="same", activation="sigmoid"
            )(input_preS)

        feat_preS = tf.keras.layers.Reshape((-1,))(feat_preS)

        SR_matrix = tf.keras.layers.Dense(
            m_dim * (self.map_xy_size * self.map_xy_size * 3), activation="sigmoid"
        )(feat_preS)
        SR_matrix = tf.keras.layers.Reshape(
            (m_dim, (self.map_xy_size * self.map_xy_size * 3))
        )(SR_matrix)

        return tf.keras.models.Model(
            inputs=input_preS, outputs=[SR_matrix, feat_preS], name="feat_S_model"
        )

    def ssr_S_model_build(self, num_primcaps, m_dim):
        input_s1_preS = tf.keras.layers.Input((self.map_xy_size, self.map_xy_size, 64))
        input_s2_preS = tf.keras.layers.Input((self.map_xy_size, self.map_xy_size, 64))
        input_s3_preS = tf.keras.layers.Input((self.map_xy_size, self.map_xy_size, 64))

        feat_S_model = self.ssr_feat_S_model_build(m_dim)

        SR_matrix_s1, feat_s1_preS = feat_S_model(input_s1_preS)
        SR_matrix_s2, feat_s2_preS = feat_S_model(input_s2_preS)
        SR_matrix_s3, feat_s3_preS = feat_S_model(input_s3_preS)

        feat_pre_concat = tf.keras.layers.Concatenate()(
            [feat_s1_preS, feat_s2_preS, feat_s3_preS]
        )
        SL_matrix = tf.keras.layers.Dense(
            int(num_primcaps / 3) * m_dim, activation="sigmoid"
        )(feat_pre_concat)
        SL_matrix = tf.keras.layers.Reshape((int(num_primcaps / 3), m_dim))(SL_matrix)

        S_matrix_s1 = MatrixMultiplyLayer(name="S_matrix_s1")([SL_matrix, SR_matrix_s1])
        S_matrix_s2 = MatrixMultiplyLayer(name="S_matrix_s2")([SL_matrix, SR_matrix_s2])
        S_matrix_s3 = MatrixMultiplyLayer(name="S_matrix_s3")([SL_matrix, SR_matrix_s3])

        # Very important!!! Without this training won't converge.
        # norm_S_s1 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s1)
        norm_S_s1 = MatrixNormLayer(tile_count=64)(S_matrix_s1)
        norm_S_s2 = MatrixNormLayer(tile_count=64)(S_matrix_s2)
        norm_S_s3 = MatrixNormLayer(tile_count=64)(S_matrix_s3)

        feat_s1_pre = tf.keras.layers.Reshape(
            (self.map_xy_size * self.map_xy_size, 64)
        )(input_s1_preS)
        feat_s2_pre = tf.keras.layers.Reshape(
            (self.map_xy_size * self.map_xy_size, 64)
        )(input_s2_preS)
        feat_s3_pre = tf.keras.layers.Reshape(
            (self.map_xy_size * self.map_xy_size, 64)
        )(input_s3_preS)

        feat_pre_concat = tf.keras.layers.Concatenate(axis=1)(
            [feat_s1_pre, feat_s2_pre, feat_s3_pre]
        )

        # Warining: don't use keras's 'K.dot'. It is very weird when high dimension is used.
        # https://github.com/keras-team/keras/issues/9779
        # Make sure 'tf.matmul' is used
        # primcaps = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix,feat_pre_concat, norm_S])
        primcaps_s1 = PrimCapsLayer()([S_matrix_s1, feat_pre_concat, norm_S_s1])
        primcaps_s2 = PrimCapsLayer()([S_matrix_s2, feat_pre_concat, norm_S_s2])
        primcaps_s3 = PrimCapsLayer()([S_matrix_s3, feat_pre_concat, norm_S_s3])

        primcaps = tf.keras.layers.Concatenate(axis=1)(
            [primcaps_s1, primcaps_s2, primcaps_s3]
        )

        return tf.keras.models.Model(
            inputs=[input_s1_preS, input_s2_preS, input_s3_preS],
            outputs=primcaps,
            name="ssr_S_model",
        )

    def ssr_noS_model_build(self, **kwargs):

        input_s1_preS = tf.keras.layers.Input((self.map_xy_size, self.map_xy_size, 64))
        input_s2_preS = tf.keras.layers.Input((self.map_xy_size, self.map_xy_size, 64))
        input_s3_preS = tf.keras.layers.Input((self.map_xy_size, self.map_xy_size, 64))

        primcaps_s1 = tf.keras.layers.Reshape(
            (self.map_xy_size * self.map_xy_size, 64)
        )(input_s1_preS)
        primcaps_s2 = tf.keras.layers.Reshape(
            (self.map_xy_size * self.map_xy_size, 64)
        )(input_s2_preS)
        primcaps_s3 = tf.keras.layers.Reshape(
            (self.map_xy_size * self.map_xy_size, 64)
        )(input_s3_preS)

        primcaps = tf.keras.layers.Concatenate(axis=1)(
            [primcaps_s1, primcaps_s2, primcaps_s3]
        )
        return tf.keras.models.Model(
            inputs=[input_s1_preS, input_s2_preS, input_s3_preS],
            outputs=primcaps,
            name="ssr_S_model",
        )

    def __call__(self):
        logging.debug("Creating model...")
        img_inputs = tf.keras.layers.Input(self._input_shape)

        # Build various models
        ssr_G_model = self.ssr_G_model_build(img_inputs)

        if self.is_noS_model:
            ssr_S_model = self.ssr_noS_model_build()
        else:
            ssr_S_model = self.ssr_S_model_build(
                num_primcaps=self.num_primcaps, m_dim=self.m_dim
            )

        ssr_aggregation_model = self.ssr_aggregation_model_build(
            (self.num_primcaps, 64)
        )

        if self.is_fc_model:
            ssr_F_Cap_model = self.ssr_FC_model_build(self.F_shape, "ssr_F_Cap_model")
        else:
            ssr_F_Cap_model = self.ssr_F_model_build(self.F_shape, "ssr_F_Cap_model")

        # Wire them up
        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Cap_list = ssr_aggregation_model(ssr_primcaps)
        ssr_F_Cap_list = ssr_F_Cap_model(ssr_Cap_list)

        pred_pose = SSRLayer(
            s1=self.stage_num[0],
            s2=self.stage_num[1],
            s3=self.stage_num[2],
            lambda_d=self.lambda_d,
            name="pred_pose",
        )(ssr_F_Cap_list)

        return tf.keras.models.Model(inputs=img_inputs, outputs=pred_pose)


# Capsule FSANetworks


class BaseCapsuleFSANet(BaseFSANet):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(BaseCapsuleFSANet, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )

    def ssr_aggregation_model_build(self, shape_primcaps):
        input_primcaps = tf.keras.layers.Input(shape_primcaps)
        capsule = CapsuleLayer(
            self.num_capsule, self.dim_capsule, routings=self.routings, name="caps"
        )(input_primcaps)

        feat_s1_div, feat_s2_div, feat_s3_div = AggregatedFeatureExtractionLayer(
            num_capsule=self.num_capsule
        )(capsule)

        feat_s1_div = tf.keras.layers.Reshape((-1,))(feat_s1_div)
        feat_s2_div = tf.keras.layers.Reshape((-1,))(feat_s2_div)
        feat_s3_div = tf.keras.layers.Reshape((-1,))(feat_s3_div)

        return tf.keras.models.Model(
            inputs=input_primcaps,
            outputs=[feat_s1_div, feat_s2_div, feat_s3_div],
            name="ssr_Cap_model",
        )


class FSA_net_Capsule(BaseCapsuleFSANet):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_Capsule, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_varS_model = False


class FSA_net_Var_Capsule(BaseCapsuleFSANet):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_Var_Capsule, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_varS_model = True


class FSA_net_noS_Capsule(BaseCapsuleFSANet):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_noS_Capsule, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_noS_model = True


class FSA_net_Capsule_FC(FSA_net_Capsule):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_Capsule_FC, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_fc_model = True


class FSA_net_Var_Capsule_FC(FSA_net_Var_Capsule):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_Var_Capsule_FC, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_fc_model = True


class FSA_net_noS_Capsule_FC(FSA_net_noS_Capsule):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_noS_Capsule_FC, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_fc_model = True


# NetVLAD models


class BaseNetVLADFSANet(BaseFSANet):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(BaseNetVLADFSANet, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )

    def ssr_aggregation_model_build(self, shape_primcaps):
        input_primcaps = tf.keras.layers.Input(shape_primcaps)

        agg_feat = NetVLAD(
            feature_size=64,
            max_samples=self.num_primcaps,
            cluster_size=self.num_capsule,
            output_dim=self.num_capsule * self.dim_capsule,
        )(input_primcaps)
        agg_feat = tf.keras.layers.Reshape((self.num_capsule, self.dim_capsule))(
            agg_feat
        )

        feat_s1_div, feat_s2_div, feat_s3_div = AggregatedFeatureExtractionLayer(
            num_capsule=self.num_capsule
        )(agg_feat)

        feat_s1_div = tf.keras.layers.Reshape((-1,))(feat_s1_div)
        feat_s2_div = tf.keras.layers.Reshape((-1,))(feat_s2_div)
        feat_s3_div = tf.keras.layers.Reshape((-1,))(feat_s3_div)

        return tf.keras.models.Model(
            inputs=input_primcaps,
            outputs=[feat_s1_div, feat_s2_div, feat_s3_div],
            name="ssr_Agg_model",
        )


class FSA_net_NetVLAD(BaseNetVLADFSANet):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_NetVLAD, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_varS_model = False


class FSA_net_Var_NetVLAD(BaseNetVLADFSANet):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_Var_NetVLAD, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_varS_model = True


class FSA_net_noS_NetVLAD(BaseNetVLADFSANet):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_noS_NetVLAD, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_noS_model = True


class FSA_net_NetVLAD_FC(FSA_net_NetVLAD):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_NetVLAD_FC, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_fc_model = True


class FSA_net_Var_NetVLAD_FC(FSA_net_Var_NetVLAD):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_Var_NetVLAD_FC, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_fc_model = True


class FSA_net_noS_NetVLAD_FC(FSA_net_noS_NetVLAD):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_noS_NetVLAD_FC, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_fc_model = True


# // Metric models


class BaseMetricFSANet(BaseFSANet):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(BaseMetricFSANet, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )

    def ssr_aggregation_model_build(self, shape_primcaps):
        input_primcaps = tf.keras.layers.Input(shape_primcaps)

        metric_feat = MatMulLayer(16, type=1)(input_primcaps)
        metric_feat = MatMulLayer(3, type=2)(metric_feat)

        feat_s1_div, feat_s2_div, feat_s3_div = AggregatedFeatureExtractionLayer(
            num_capsule=self.num_capsule
        )(metric_feat)

        feat_s1_div = tf.keras.layers.Reshape((-1,))(feat_s1_div)
        feat_s2_div = tf.keras.layers.Reshape((-1,))(feat_s2_div)
        feat_s3_div = tf.keras.layers.Reshape((-1,))(feat_s3_div)

        return tf.keras.models.Model(
            inputs=input_primcaps,
            outputs=[feat_s1_div, feat_s2_div, feat_s3_div],
            name="ssr_Metric_model",
        )


class FSA_net_Metric(BaseMetricFSANet):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_Metric, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_varS_model = False


class FSA_net_Var_Metric(BaseMetricFSANet):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_Var_Metric, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_varS_model = True


class FSA_net_noS_Metric(BaseMetricFSANet):
    def __init__(self, image_size, num_classes, stage_num, lambda_d, S_set):
        super(FSA_net_noS_Metric, self).__init__(
            image_size, num_classes, stage_num, lambda_d, S_set
        )
        self.is_noS_model = True


class SSR_net:
    def __init__(self, image_size, stage_num, lambda_local, lambda_d):

        self._channel_axis = -1
        self._input_shape = (image_size, image_size, 3)
        self.stage_num = stage_num
        self.lambda_local = lambda_local
        self.lambda_d = lambda_d

    def __call__(self):
        logging.debug("Creating model...")

        inputs = tf.keras.layers.Input(shape=self._input_shape)

        # -------------------------------------------------------------------------------------------------------------------------
        x = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        x = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x_layer1 = tf.keras.layers.AveragePooling2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(32, (3, 3))(x_layer1)
        x = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x_layer2 = tf.keras.layers.AveragePooling2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(32, (3, 3))(x_layer2)
        x = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x_layer3 = tf.keras.layers.AveragePooling2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(32, (3, 3))(x_layer3)
        x = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(x)
        x = tf.keras.layers.Activation("relu")(x)
        # -------------------------------------------------------------------------------------------------------------------------
        s = tf.keras.layers.Conv2D(16, (3, 3))(inputs)
        s = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(s)
        s = tf.keras.layers.Activation("tanh")(s)
        s_layer1 = tf.keras.layers.MaxPooling2D(2, 2)(s)
        s = tf.keras.layers.Conv2D(16, (3, 3))(s_layer1)
        s = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(s)
        s = tf.keras.layers.Activation("tanh")(s)
        s_layer2 = tf.keras.layers.MaxPooling2D(2, 2)(s)
        s = tf.keras.layers.Conv2D(16, (3, 3))(s_layer2)
        s = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(s)
        s = tf.keras.layers.Activation("tanh")(s)
        s_layer3 = tf.keras.layers.MaxPooling2D(2, 2)(s)
        s = tf.keras.layers.Conv2D(16, (3, 3))(s_layer3)
        s = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(s)
        s = tf.keras.layers.Activation("tanh")(s)

        # -------------------------------------------------------------------------------------------------------------------------
        # Classifier block
        s_layer4 = tf.keras.layers.Conv2D(10, (1, 1), activation="relu")(s)
        s_layer4 = tf.keras.layers.Flatten()(s_layer4)
        s_layer4_mix = tf.keras.layers.Dropout(0.2)(s_layer4)
        s_layer4_mix = tf.keras.layers.Dense(
            units=self.stage_num[0], activation="relu"
        )(s_layer4_mix)

        x_layer4 = tf.keras.layers.Conv2D(10, (1, 1), activation="relu")(x)
        x_layer4 = tf.keras.layers.Flatten()(x_layer4)
        x_layer4_mix = tf.keras.layers.Dropout(0.2)(x_layer4)
        x_layer4_mix = tf.keras.layers.Dense(
            units=self.stage_num[0], activation="relu"
        )(x_layer4_mix)

        feat_a_s1_pre = tf.keras.layers.Multiply()([s_layer4, x_layer4])
        delta_s1 = tf.keras.layers.Dense(1, activation="tanh", name="delta_s1")(
            feat_a_s1_pre
        )

        feat_a_s1 = tf.keras.layers.Multiply()([s_layer4_mix, x_layer4_mix])
        feat_a_s1 = tf.keras.layers.Dense(2 * self.stage_num[0], activation="relu")(
            feat_a_s1
        )
        pred_a_s1 = tf.keras.layers.Dense(
            units=self.stage_num[0], activation="relu", name="pred_age_stage1"
        )(feat_a_s1)
        # feat_local_s1 = Lambda(lambda x: x/10)(feat_a_s1)
        # feat_a_s1_local = Dropout(0.2)(pred_a_s1)
        local_s1 = tf.keras.layers.Dense(
            units=self.stage_num[0], activation="tanh", name="local_delta_stage1"
        )(feat_a_s1)
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer2 = tf.keras.layers.Conv2D(10, (1, 1), activation="relu")(s_layer2)
        s_layer2 = tf.keras.layers.MaxPooling2D(4, 4)(s_layer2)
        s_layer2 = tf.keras.layers.Flatten()(s_layer2)
        s_layer2_mix = tf.keras.layers.Dropout(0.2)(s_layer2)
        s_layer2_mix = tf.keras.layers.Dense(self.stage_num[1], activation="relu")(
            s_layer2_mix
        )

        x_layer2 = tf.keras.layers.Conv2D(10, (1, 1), activation="relu")(x_layer2)
        x_layer2 = tf.keras.layers.AveragePooling2D(4, 4)(x_layer2)
        x_layer2 = tf.keras.layers.Flatten()(x_layer2)
        x_layer2_mix = tf.keras.layers.Dropout(0.2)(x_layer2)
        x_layer2_mix = tf.keras.layers.Dense(self.stage_num[1], activation="relu")(
            x_layer2_mix
        )

        feat_a_s2_pre = tf.keras.layers.Multiply()([s_layer2, x_layer2])
        delta_s2 = tf.keras.layers.Dense(1, activation="tanh", name="delta_s2")(
            feat_a_s2_pre
        )

        feat_a_s2 = tf.keras.layers.Multiply()([s_layer2_mix, x_layer2_mix])
        feat_a_s2 = tf.keras.layers.Dense(2 * self.stage_num[1], activation="relu")(
            feat_a_s2
        )
        pred_a_s2 = tf.keras.layers.Dense(
            units=self.stage_num[1], activation="relu", name="pred_age_stage2"
        )(feat_a_s2)
        # feat_local_s2 = Lambda(lambda x: x/10)(feat_a_s2)
        # feat_a_s2_local = Dropout(0.2)(pred_a_s2)
        local_s2 = tf.keras.layers.Dense(
            units=self.stage_num[1], activation="tanh", name="local_delta_stage2"
        )(feat_a_s2)
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer1 = tf.keras.layers.Conv2D(10, (1, 1), activation="relu")(s_layer1)
        s_layer1 = tf.keras.layers.MaxPooling2D(8, 8)(s_layer1)
        s_layer1 = tf.keras.layers.Flatten()(s_layer1)
        s_layer1_mix = tf.keras.layers.Dropout(0.2)(s_layer1)
        s_layer1_mix = tf.keras.layers.Dense(self.stage_num[2], activation="relu")(
            s_layer1_mix
        )

        x_layer1 = tf.keras.layers.Conv2D(10, (1, 1), activation="relu")(x_layer1)
        x_layer1 = tf.keras.layers.AveragePooling2D(8, 8)(x_layer1)
        x_layer1 = tf.keras.layers.Flatten()(x_layer1)
        x_layer1_mix = tf.keras.layers.Dropout(0.2)(x_layer1)
        x_layer1_mix = tf.keras.layers.Dense(self.stage_num[2], activation="relu")(
            x_layer1_mix
        )

        feat_a_s3_pre = tf.keras.layers.Multiply()([s_layer1, x_layer1])
        delta_s3 = tf.keras.layers.Dense(1, activation="tanh", name="delta_s3")(
            feat_a_s3_pre
        )

        feat_a_s3 = tf.keras.layers.Multiply()([s_layer1_mix, x_layer1_mix])
        feat_a_s3 = tf.keras.layers.Dense(2 * self.stage_num[2], activation="relu")(
            feat_a_s3
        )
        pred_a_s3 = tf.keras.layers.Dense(
            units=self.stage_num[2], activation="relu", name="pred_age_stage3"
        )(feat_a_s3)
        # feat_local_s3 = Lambda(lambda x: x/10)(feat_a_s3)
        # feat_a_s3_local = Dropout(0.2)(pred_a_s3)
        local_s3 = tf.keras.layers.Dense(
            units=self.stage_num[2], activation="tanh", name="local_delta_stage3"
        )(feat_a_s3)
        # -------------------------------------------------------------------------------------------------------------------------

        def merge_age(x, s1, s2, s3, lambda_local, lambda_d):
            a = x[0][:, 0] * 0
            b = x[0][:, 0] * 0
            c = x[0][:, 0] * 0
            # A = s1 * s2 * s3
            V = 101

            for i in range(0, s1):
                a = a + (i + lambda_local * x[6][:, i]) * x[0][:, i]
            a = K.expand_dims(a, -1)
            a = a / (s1 * (1 + lambda_d * x[3]))

            for j in range(0, s2):
                b = b + (j + lambda_local * x[7][:, j]) * x[1][:, j]
            b = K.expand_dims(b, -1)
            b = b / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4]))

            for k in range(0, s3):
                c = c + (k + lambda_local * x[8][:, k]) * x[2][:, k]
            c = K.expand_dims(c, -1)
            c = c / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4])) / (s3 * (1 + lambda_d * x[5]))

            age = (a + b + c) * V
            return age

        pred_a = tf.keras.layers.Lambda(
            merge_age,
            arguments={
                "s1": self.stage_num[0],
                "s2": self.stage_num[1],
                "s3": self.stage_num[2],
                "lambda_local": self.lambda_local,
                "lambda_d": self.lambda_d,
            },
            name="pred_a",
        )(
            [
                pred_a_s1,
                pred_a_s2,
                pred_a_s3,
                delta_s1,
                delta_s2,
                delta_s3,
                local_s1,
                local_s2,
                local_s3,
            ]
        )

        model = tf.keras.models.Model(inputs=inputs, outputs=pred_a)

        return model


class SSR_net_MT:
    def __init__(self, image_size, num_classes, stage_num, lambda_d):

        self._channel_axis = -1
        self._input_shape = (image_size, image_size, 3)
        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = tf.keras.layers.Input(self._input_shape)
        # -------------------------------------------------------------------------------------------------------------------------
        x = tf.keras.layers.SeparableConv2D(16, (3, 3), padding="same")(img_inputs)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x_layer1 = tf.keras.layers.AveragePooling2D((2, 2))(x)
        x = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same")(x_layer1)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x_layer2 = tf.keras.layers.AveragePooling2D((2, 2))(x)
        x = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same")(x_layer2)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x_layer3 = tf.keras.layers.AveragePooling2D((2, 2))(x)
        x = tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same")(x_layer3)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x_layer4 = tf.keras.layers.Activation("relu")(x)
        # -------------------------------------------------------------------------------------------------------------------------
        s = tf.keras.layers.SeparableConv2D(16, (3, 3), padding="same")(img_inputs)
        s = tf.keras.layers.BatchNormalization(axis=-1)(s)
        s = tf.keras.layers.Activation("tanh")(s)
        s_layer1 = tf.keras.layers.MaxPooling2D((2, 2))(s)
        s = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same")(s_layer1)
        s = tf.keras.layers.BatchNormalization(axis=-1)(s)
        s = tf.keras.layers.Activation("tanh")(s)
        s = tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same")(s)
        s = tf.keras.layers.BatchNormalization(axis=-1)(s)
        s = tf.keras.layers.Activation("tanh")(s)
        s_layer2 = tf.keras.layers.MaxPooling2D((2, 2))(s)
        s = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same")(s_layer2)
        s = tf.keras.layers.BatchNormalization(axis=-1)(s)
        s = tf.keras.layers.Activation("tanh")(s)
        s = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same")(s)
        s = tf.keras.layers.BatchNormalization(axis=-1)(s)
        s = tf.keras.layers.Activation("tanh")(s)
        s_layer3 = tf.keras.layers.MaxPooling2D((2, 2))(s)
        s = tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same")(s_layer3)
        s = tf.keras.layers.BatchNormalization(axis=-1)(s)
        s = tf.keras.layers.Activation("tanh")(s)
        s = tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same")(s)
        s = tf.keras.layers.BatchNormalization(axis=-1)(s)
        s_layer4 = tf.keras.layers.Activation("tanh")(s)

        # -------------------------------------------------------------------------------------------------------------------------
        # Classifier block
        s_layer4 = tf.keras.layers.Conv2D(64, (1, 1), activation="tanh")(s_layer4)
        s_layer4 = tf.keras.layers.MaxPooling2D((2, 2))(s_layer4)

        x_layer4 = tf.keras.layers.Conv2D(64, (1, 1), activation="relu")(x_layer4)
        x_layer4 = tf.keras.layers.AveragePooling2D((2, 2))(x_layer4)

        feat_s1_pre = tf.keras.layers.Multiply()([s_layer4, x_layer4])
        feat_s1_pre = tf.keras.layers.Flatten()(feat_s1_pre)
        feat_delta_s1 = tf.keras.layers.Dense(2 * self.num_classes, activation="tanh")(
            feat_s1_pre
        )
        delta_s1 = tf.keras.layers.Dense(
            self.num_classes, activation="tanh", name="delta_s1"
        )(feat_delta_s1)

        feat_local_s1 = tf.keras.layers.Dense(2 * self.num_classes, activation="tanh")(
            feat_s1_pre
        )
        local_s1 = tf.keras.layers.Dense(
            units=self.num_classes, activation="tanh", name="local_delta_stage1"
        )(feat_local_s1)

        feat_pred_s1 = tf.keras.layers.Dense(
            self.stage_num[0] * self.num_classes, activation="relu"
        )(feat_s1_pre)
        pred_a_s1 = tf.keras.layers.Reshape((self.num_classes, self.stage_num[0]))(
            feat_pred_s1
        )
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer3 = tf.keras.layers.Conv2D(64, (1, 1), activation="tanh")(s_layer3)
        s_layer3 = tf.keras.layers.MaxPooling2D((2, 2))(s_layer3)

        x_layer3 = tf.keras.layers.Conv2D(64, (1, 1), activation="relu")(x_layer3)
        x_layer3 = tf.keras.layers.AveragePooling2D((2, 2))(x_layer3)

        feat_s2_pre = tf.keras.layers.Multiply()([s_layer3, x_layer3])
        feat_s2_pre = tf.keras.layers.Flatten()(feat_s2_pre)
        feat_delta_s2 = tf.keras.layers.Dense(2 * self.num_classes, activation="tanh")(
            feat_s2_pre
        )
        delta_s2 = tf.keras.layers.Dense(
            self.num_classes, activation="tanh", name="delta_s2"
        )(feat_delta_s2)

        feat_local_s2 = tf.keras.layers.Dense(2 * self.num_classes, activation="tanh")(
            feat_s2_pre
        )
        local_s2 = tf.keras.layers.Dense(
            units=self.num_classes, activation="tanh", name="local_delta_stage2"
        )(feat_local_s2)

        feat_pred_s2 = tf.keras.layers.Dense(
            self.stage_num[1] * self.num_classes, activation="relu"
        )(feat_s2_pre)
        pred_a_s2 = tf.keras.layers.Reshape((self.num_classes, self.stage_num[1]))(
            feat_pred_s2
        )
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer2 = tf.keras.layers.Conv2D(64, (1, 1), activation="tanh")(s_layer2)
        s_layer2 = tf.keras.layers.MaxPooling2D((2, 2))(s_layer2)

        x_layer2 = tf.keras.layers.Conv2D(64, (1, 1), activation="relu")(x_layer2)
        x_layer2 = tf.keras.layers.AveragePooling2D((2, 2))(x_layer2)

        feat_s3_pre = tf.keras.layers.Multiply()([s_layer2, x_layer2])
        feat_s3_pre = tf.keras.layers.Flatten()(feat_s3_pre)
        feat_delta_s3 = tf.keras.layers.Dense(2 * self.num_classes, activation="tanh")(
            feat_s3_pre
        )
        delta_s3 = tf.keras.layers.Dense(
            self.num_classes, activation="tanh", name="delta_s3"
        )(feat_delta_s3)

        feat_local_s3 = tf.keras.layers.Dense(2 * self.num_classes, activation="tanh")(
            feat_s3_pre
        )
        local_s3 = tf.keras.layers.Dense(
            units=self.num_classes, activation="tanh", name="local_delta_stage3"
        )(feat_local_s3)

        feat_pred_s3 = tf.keras.layers.Dense(
            self.stage_num[2] * self.num_classes, activation="relu"
        )(feat_s3_pre)
        pred_a_s3 = tf.keras.layers.Reshape((self.num_classes, self.stage_num[2]))(
            feat_pred_s3
        )
        # -------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x, s1, s2, s3, lambda_d):
            a = x[0][:, :, 0] * 0
            b = x[0][:, :, 0] * 0
            c = x[0][:, :, 0] * 0

            di = s1 // 2
            dj = s2 // 2
            dk = s3 // 2

            V = 99
            # lambda_d = 0.9

            for i in range(0, s1):
                a = a + (i - di + x[6]) * x[0][:, :, i]
            # a = K.expand_dims(a,-1)
            a = a / (s1 * (1 + lambda_d * x[3]))

            for j in range(0, s2):
                b = b + (j - dj + x[7]) * x[1][:, :, j]
            # b = K.expand_dims(b,-1)
            b = b / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4]))

            for k in range(0, s3):
                c = c + (k - dk + x[8]) * x[2][:, :, k]
            # c = K.expand_dims(c,-1)
            c = c / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4])) / (s3 * (1 + lambda_d * x[5]))

            pred = (a + b + c) * V

            return pred

        pred_pose = tf.keras.layers.Lambda(
            SSR_module,
            arguments={
                "s1": self.stage_num[0],
                "s2": self.stage_num[1],
                "s3": self.stage_num[2],
                "lambda_d": self.lambda_d,
            },
            name="pred_pose",
        )(
            [
                pred_a_s1,
                pred_a_s2,
                pred_a_s3,
                delta_s1,
                delta_s2,
                delta_s3,
                local_s1,
                local_s2,
                local_s3,
            ]
        )

        model = tf.keras.models.Model(inputs=img_inputs, outputs=pred_pose)

        return model


class SSR_net_ori_MT:
    def __init__(self, image_size, num_classes, stage_num, lambda_d):

        self._channel_axis = -1
        self._input_shape = (image_size, image_size, 3)
        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = tf.keras.layers.Input(self._input_shape)
        # -------------------------------------------------------------------------------------------------------------------------
        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(img_inputs)
        x = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x_layer1 = tf.keras.layers.AveragePooling2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x_layer1)
        x = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x_layer2 = tf.keras.layers.AveragePooling2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x_layer2)
        x = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x_layer3 = tf.keras.layers.AveragePooling2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x_layer3)
        x = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(x)
        x_layer4 = tf.keras.layers.Activation("relu")(x)
        # -------------------------------------------------------------------------------------------------------------------------
        s = tf.keras.layers.Conv2D(16, (3, 3), padding="same")(img_inputs)
        s = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(s)
        s = tf.keras.layers.Activation("tanh")(s)
        s_layer1 = tf.keras.layers.MaxPooling2D(2, 2)(s)
        s = tf.keras.layers.Conv2D(16, (3, 3), padding="same")(s_layer1)
        s = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(s)
        s = tf.keras.layers.Activation("tanh")(s)
        s_layer2 = tf.keras.layers.MaxPooling2D(2, 2)(s)
        s = tf.keras.layers.Conv2D(16, (3, 3), padding="same")(s_layer2)
        s = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(s)
        s = tf.keras.layers.Activation("tanh")(s)
        s_layer3 = tf.keras.layers.MaxPooling2D(2, 2)(s)
        s = tf.keras.layers.Conv2D(16, (3, 3), padding="same")(s_layer3)
        s = tf.keras.layers.BatchNormalization(axis=self._channel_axis)(s)
        s_layer4 = tf.keras.layers.Activation("tanh")(s)

        # -------------------------------------------------------------------------------------------------------------------------
        # Classifier block
        s_layer4 = tf.keras.layers.Conv2D(64, (1, 1), activation="tanh")(s_layer4)
        s_layer4 = tf.keras.layers.MaxPooling2D((2, 2))(s_layer4)

        x_layer4 = tf.keras.layers.Conv2D(64, (1, 1), activation="relu")(x_layer4)
        x_layer4 = tf.keras.layers.AveragePooling2D((2, 2))(x_layer4)

        feat_s1_pre = tf.keras.layers.Multiply()([s_layer4, x_layer4])
        feat_s1_pre = tf.keras.layers.Flatten()(feat_s1_pre)
        feat_delta_s1 = tf.keras.layers.Dense(2 * self.num_classes, activation="tanh")(
            feat_s1_pre
        )
        delta_s1 = tf.keras.layers.Dense(
            self.num_classes, activation="tanh", name="delta_s1"
        )(feat_delta_s1)

        feat_local_s1 = tf.keras.layers.Dense(2 * self.num_classes, activation="tanh")(
            feat_s1_pre
        )
        local_s1 = tf.keras.layers.Dense(
            units=self.num_classes, activation="tanh", name="local_delta_stage1"
        )(feat_local_s1)

        feat_pred_s1 = tf.keras.layers.Dense(
            self.stage_num[0] * self.num_classes, activation="relu"
        )(feat_s1_pre)
        pred_a_s1 = tf.keras.layers.Reshape((self.num_classes, self.stage_num[0]))(
            feat_pred_s1
        )
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer3 = tf.keras.layers.Conv2D(64, (1, 1), activation="tanh")(s_layer3)
        s_layer3 = tf.keras.layers.MaxPooling2D((2, 2))(s_layer3)

        x_layer3 = tf.keras.layers.Conv2D(64, (1, 1), activation="relu")(x_layer3)
        x_layer3 = tf.keras.layers.AveragePooling2D((2, 2))(x_layer3)

        feat_s2_pre = tf.keras.layers.Multiply()([s_layer3, x_layer3])
        feat_s2_pre = tf.keras.layers.Flatten()(feat_s2_pre)
        feat_delta_s2 = tf.keras.layers.Dense(2 * self.num_classes, activation="tanh")(
            feat_s2_pre
        )
        delta_s2 = tf.keras.layers.Dense(
            self.num_classes, activation="tanh", name="delta_s2"
        )(feat_delta_s2)

        feat_local_s2 = tf.keras.layers.Dense(2 * self.num_classes, activation="tanh")(
            feat_s2_pre
        )
        local_s2 = tf.keras.layers.Dense(
            units=self.num_classes, activation="tanh", name="local_delta_stage2"
        )(feat_local_s2)

        feat_pred_s2 = tf.keras.layers.Dense(
            self.stage_num[1] * self.num_classes, activation="relu"
        )(feat_s2_pre)
        pred_a_s2 = tf.keras.layers.Reshape((self.num_classes, self.stage_num[1]))(
            feat_pred_s2
        )
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer2 = tf.keras.layers.Conv2D(64, (1, 1), activation="tanh")(s_layer2)
        s_layer2 = tf.keras.layers.MaxPooling2D((2, 2))(s_layer2)

        x_layer2 = tf.keras.layers.Conv2D(64, (1, 1), activation="relu")(x_layer2)
        x_layer2 = tf.keras.layers.AveragePooling2D((2, 2))(x_layer2)

        feat_s3_pre = tf.keras.layers.Multiply()([s_layer2, x_layer2])
        feat_s3_pre = tf.keras.layers.Flatten()(feat_s3_pre)
        feat_delta_s3 = tf.keras.layers.Dense(2 * self.num_classes, activation="tanh")(
            feat_s3_pre
        )
        delta_s3 = tf.keras.layers.Dense(
            self.num_classes, activation="tanh", name="delta_s3"
        )(feat_delta_s3)

        feat_local_s3 = tf.keras.layers.Dense(2 * self.num_classes, activation="tanh")(
            feat_s3_pre
        )
        local_s3 = tf.keras.layers.Dense(
            units=self.num_classes, activation="tanh", name="local_delta_stage3"
        )(feat_local_s3)

        feat_pred_s3 = tf.keras.layers.Dense(
            self.stage_num[2] * self.num_classes, activation="relu"
        )(feat_s3_pre)
        pred_a_s3 = tf.keras.layers.Reshape((self.num_classes, self.stage_num[2]))(
            feat_pred_s3
        )
        # -------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x, s1, s2, s3, lambda_d):
            a = x[0][:, :, 0] * 0
            b = x[0][:, :, 0] * 0
            c = x[0][:, :, 0] * 0

            di = s1 // 2
            dj = s2 // 2
            dk = s3 // 2

            V = 99
            # lambda_d = 0.9

            for i in range(0, s1):
                a = a + (i - di + x[6]) * x[0][:, :, i]
            # a = K.expand_dims(a,-1)
            a = a / (s1 * (1 + lambda_d * x[3]))

            for j in range(0, s2):
                b = b + (j - dj + x[7]) * x[1][:, :, j]
            # b = K.expand_dims(b,-1)
            b = b / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4]))

            for k in range(0, s3):
                c = c + (k - dk + x[8]) * x[2][:, :, k]
            # c = K.expand_dims(c,-1)
            c = c / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4])) / (s3 * (1 + lambda_d * x[5]))

            pred = (a + b + c) * V

            return pred

        pred_pose = tf.keras.layers.Lambda(
            SSR_module,
            arguments={
                "s1": self.stage_num[0],
                "s2": self.stage_num[1],
                "s3": self.stage_num[2],
                "lambda_d": self.lambda_d,
            },
            name="pred_pose",
        )(
            [
                pred_a_s1,
                pred_a_s2,
                pred_a_s3,
                delta_s1,
                delta_s2,
                delta_s3,
                local_s1,
                local_s2,
                local_s3,
            ]
        )

        model = tf.keras.models.Model(inputs=img_inputs, outputs=pred_pose)

        return model
