import numpy as np

import tensorflow as tf
from keras.layers import Lambda, Dense, concatenate, BatchNormalization, Dropout, ReLU


try:
    from tensorflow.python.ops.init_ops import Zeros, Ones, Constant, TruncatedNormal, \
        glorot_normal_initializer as glorot_normal, \
        glorot_uniform_initializer as glorot_uniform
except ImportError:
    from tensorflow.python.ops.init_ops_v2 import Zeros, Ones, Constant, TruncatedNormal, glorot_normal, glorot_uniform

from keras.layers import Layer

from model.config.feature_config_without_multimodal_feature import input_config, build_input, build_embed_features, \
    build_dense_features


class InBatchSoftmaxLayer(Layer):
    def __init__(self, sampler_config, temperature=1.0, **kwargs):
        self.sampler_config = sampler_config
        self.temperature = temperature
        self.item_count = self.sampler_config['item_count']

        super(InBatchSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InBatchSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_item_idx, training=None, **kwargs):
        user_vec, item_vec, item_idx = inputs_with_item_idx
        if item_idx.dtype != tf.int64:
            item_idx = tf.cast(item_idx, tf.int64)
        user_vec /= self.temperature
        logits = tf.matmul(user_vec, item_vec, transpose_b=True)
        loss = self.inbatch_softmax_cross_entropy_with_logits(logits, self.item_count, item_idx)
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def inbatch_softmax_cross_entropy_with_logits(logits, item_count, item_idx):
        Q = tf.gather(tf.constant(item_count / np.sum(item_count), 'float32'),
                      tf.squeeze(item_idx, axis=1))
        try:
            logQ = tf.reshape(tf.math.log(Q), (1, -1))
            logits -= logQ  # subtract_log_q
            labels = tf.linalg.diag(tf.ones_like(logits[0]))
        except AttributeError:
            logQ = tf.reshape(tf.log(Q), (1, -1))
            logits -= logQ  # subtract_log_q
            labels = tf.diag(tf.ones_like(logits[0]))

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        return loss

    def get_config(self, ):
        config = {'sampler_config': self.sampler_config, 'temperature': self.temperature}
        base_config = super(InBatchSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def inner_product(x, y, temperature=1.0):
    return Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0], x[1])) / temperature)([x, y])


def DSSM(input_config,user_spare_faeture_config,user_dense_faeture_config,item_spare_faeture_config,item_dense_faeture_config, layer_sizes, temperature=0.05,
         dropout_rate=0.5 ):
    feature_input, feature_map, input_map = build_input(input_config)
    user_embed_features = build_embed_features(8, user_spare_faeture_config, feature_map)
    user_dense_features = build_dense_features(user_dense_faeture_config, feature_map)
    # concate起来经过dnn后得到user_embedding
    x = concatenate(user_embed_features + user_dense_features)
    for units in layer_sizes[:-1]:
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)
    x = Dense(layer_sizes[-1])(x)
    x = BatchNormalization()(x)
    x = ReLU(name="user_embedding")(x)
    item_embed_features = build_embed_features(8, item_spare_faeture_config, feature_map)
    item_dense_features = build_dense_features(item_dense_faeture_config, feature_map)
    y = concatenate(item_embed_features + item_dense_features)
    for units in layer_sizes[:-1]:
        y = Dense(units)(y)
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Dropout(dropout_rate)(y)
    y = Dense(layer_sizes[-1])(y)
    y = BatchNormalization()(y)
    y = ReLU(name="item_embedding",)(y)
    score=tf.keras.layers.Dot(axes=1,normalize=True)([x,y])
    score=score/temperature
    output = tf.math.sigmoid(score)
    output = tf.reshape(output, (-1, 1))
    model = tf.keras.Model(input_map, output)
    return model


def export_user_tower(model,user_spare_faeture_config,user_dense_faeture_config):
    user_tower = tf.keras.Model(
        [model.get_layer(name).input for name in user_spare_faeture_config + user_dense_faeture_config],
        model.get_layer('user_embedding').output)
    return user_tower


def export_item_tower(model,item_spare_faeture_config,item_dense_faeture_config):
    item_tower = tf.keras.Model(
        [model.get_layer(name).input for name in item_spare_faeture_config + item_dense_faeture_config],
        model.get_layer('item_embedding').output)
    return item_tower



