import itertools

import tensorflow as tf
from keras import layers, Model
from keras.layers import Flatten, Concatenate, Layer, Dense
from keras.regularizers import l2

from model.config.feature_config_without_multimodal_feature import build_input, build_embed_features, \
    build_spare_features, build_dense_features, build_lr_layer, dnn

try:
    from tensorflow.python.ops.init_ops import Zeros, Ones, Constant, TruncatedNormal, \
        glorot_normal_initializer as glorot_normal, \
        glorot_uniform_initializer as glorot_uniform
except ImportError:
    from tensorflow.python.ops.init_ops_v2 import Zeros, Ones, Constant, TruncatedNormal, glorot_normal, glorot_uniform


class SENetLayer(Layer):
    def __init__(self, reduction_ratio=3, seed=1024, l2_reg=0.001, **kwargs):
        super(SENetLayer, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.seed = seed
        self.l2_reg = l2_reg

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('`SENetLayer` layer should be called \
                on a list of at least 2 inputs')

        self.field_size = len(input_shape)
        self.embedding_size = input_shape[0].as_list()[-1]
        reduction_size = max(1, int(self.field_size // self.reduction_ratio))

        # 定义两个全连接层
        self.W_1 = self.add_weight(name="W_1", shape=(self.field_size,
                                                      reduction_size), initializer=glorot_normal(self.seed),
                                   regularizer=l2(self.l2_reg), )
        self.W_2 = self.add_weight(name="W_2", shape=(reduction_size,
                                                      self.field_size), initializer=glorot_normal(self.seed),
                                   regularizer=l2(self.l2_reg))
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        """inputs: 是一个长度为field_size的列表，其中每个元素的形状为：
            (None, 1, embedding_size)
        """
        inputs = Concatenate(axis=1)(inputs)  # (None, field_size, embedding_size)
        x = tf.reduce_mean(inputs, axis=-1)  # (None, field_size)

        # (None, field_size) * (field_size, reduction_size) =
        # (None, reduction_size)
        A_1 = tf.tensordot(x, self.W_1, axes=(-1, 0))
        A_1 = tf.nn.relu(A_1)
        # (None, reduction_size) * (reduction_size, field_size) =
        # (None, field_size)
        A_2 = tf.tensordot(A_1, self.W_2, axes=(-1, 0))
        A_2 = tf.nn.relu(A_2)
        A_2 = tf.expand_dims(A_2, axis=2)  # (None, field_size, 1)

        res = tf.multiply(inputs, A_2)  # (None, field_size, embedding_size)
        # 切分成数组，方便后续特征交叉
        res = tf.split(res, self.field_size, axis=1)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"reduction_ratio": self.reduction_ratio, "seed": self.seed}
        base_config = super(SENetLayer, self).get_config()
        base_config.update(config)
        return base_config


class BilinearInteractionLayer(Layer):
    def __init__(self, bilinear_type="interaction", seed=1024, l2_reg=0.001, **kwargs):
        super(BilinearInteractionLayer, self).__init__(**kwargs)
        self.bilinear_type = bilinear_type
        self.seed = seed
        self.l2_reg = l2_reg

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('`OuterProduct` layer should be called \
                on a list of at least 2 inputs')

        embedding_size = input_shape[0].as_list()[-1]
        field_size = len(input_shape)

        # 所有交叉特征共享一个交叉矩阵
        if self.bilinear_type == 'all':
            self.W = self.add_weight(name='bilinear_weight',
                                     shape=(embedding_size, embedding_size),
                                     initializer=glorot_normal(self.seed), regularizer=l2(self.l2_reg))
        # 每个特征使用一个交叉矩阵
        elif self.bilinear_type == "each":
            self.W_list = [self.add_weight(name='bilinear_weight_' + str(i),
                                           shape=(embedding_size, embedding_size),
                                           initializer=glorot_normal(self.seed), regularizer=l2(self.l2_reg)) for i in
                           range(field_size)]
        # 每组交叉特征使用一个交叉矩阵
        elif self.bilinear_type == 'interaction':
            self.W_list = [self.add_weight(name='bilinear_weight_' + str(i) + "_" + str(j),
                                           shape=(embedding_size, embedding_size),
                                           initializer=glorot_normal(self.seed), regularizer=l2(self.l2_reg))
                           for i, j in itertools.combinations(range(field_size), 2)]
        else:
            raise NotImplementedError
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        """inputs: 是一个长度为field_size的列表，其中每个元素的形状为：
            (None, 1, embedding_size)
        """
        if self.bilinear_type == 'all':
            # 计算点积, 遍历所有的特征分别与交叉矩阵计算内积
            # inputs[i]: (None, 1, embedding_size)
            vdotw_list = [tf.tensordot(inputs[i], self.W, axes=(-1, 0))
                          for i in range(len(inputs))]
            # 计算哈达玛积，遍历两两特征组合，计算哈达玛积
            p = [tf.multiply(vdotw_list[i], inputs[j]) for i, j in
                 itertools.combinations(range(len(inputs)), 2)]
        elif self.bilinear_type == 'each':
            # 每个特征都有一个交叉矩阵，self.W_list[i]
            vdotw_list = [tf.tensordot(inputs[i], self.W_list[i],
                                       axes=(-1, 0)) for i in range(len(inputs))]
            p = [tf.multiply(vdotw_list[i], inputs[j]) for i, j in
                 itertools.combinations(range(len(inputs)), 2)]
        elif self.bilinear_type == 'interaction':
            p = [tf.multiply(tf.tensordot(inputs[v[0]], w, axes=(-1, 0)), inputs[v[1]]) for v, w in
                 zip(itertools.combinations(range(len(inputs)), 2), self.W_list)]
        else:
            raise NotImplementedError
        # (None, field_size * (field_size - 1) / 2, embedding_size)
        output = tf.concat(p, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        field_size = input_shape[1]
        embedding_size = input_shape[2]
        return (None, int(field_size * (field_size - 1) // 2), embedding_size)

    def get_config(self):
        config = {"bilinear_type": self.bilinear_type, "seed": self.seed}
        base_config = super(BilinearInteractionLayer, self).get_config()
        base_config.update(config)
        return base_config


def fibinet(input_config, spare_features_config, dense_features_config, hidden_units=[64, 32, 16], dropout_rate=0.5,
            bilinear_type='each', k_reg=0.01):
    feature_input, feature_map, input_map = build_input(input_config)
    embed_features = build_embed_features(8, spare_features_config, feature_map)
    spare_features = build_spare_features(spare_features_config, feature_map)
    dense_features = build_dense_features(dense_features_config, feature_map)
    lr_layer = build_lr_layer(k_reg)
    liner_output = lr_layer(layers.concatenate(spare_features))

    senet_out_embedding_list = SENetLayer()([tf.expand_dims(ef, axis=1) for ef in embed_features])

    bilinear_out = BilinearInteractionLayer(bilinear_type)([tf.expand_dims(ef, axis=1) for ef in embed_features])
    bilinear_out_se = BilinearInteractionLayer(bilinear_type)(senet_out_embedding_list)

    dnn_bilinear_inputs = Flatten()(bilinear_out)
    dnn_bilinear_se_inputs = Flatten()(bilinear_out_se)

    x = []
    x.append(dnn_bilinear_inputs)
    x.append(dnn_bilinear_se_inputs)
    x = layers.concatenate(x + dense_features)
    dnn_output = Dense(1, use_bias=False)(dnn(x, hidden_units, dropout_rate, k_reg))
    # output = layers.Dense(1, activation='sigmoid')(layers.concatenate([x,liner_layer]))
    output = tf.math.sigmoid(tf.keras.layers.Add()([liner_output, dnn_output]))
    output = tf.reshape(output, (-1, 1))
    model = Model(input_map, output)
    return model
