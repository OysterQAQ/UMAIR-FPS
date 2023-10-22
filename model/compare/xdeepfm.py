import tensorflow as tf
from keras import layers

from model.config.feature_config_without_multimodal_feature import build_input, build_embed_features, \
    build_spare_features, build_dense_features, build_lr_layer, dnn
from model.layer.cross_net import CIN


def xdeepfm(input_config, spare_features_config, dense_features_config, hidden_units=[64, 32, 16], dropout_rate=0.5,
            k_reg=0.001):
    feature_input, feature_map, input_map = build_input(input_config)
    embed_features = build_embed_features(8, spare_features_config, feature_map)
    spare_features = build_spare_features(spare_features_config, feature_map)
    dense_features = build_dense_features(dense_features_config, feature_map)
    # 构建线性回归层
    lr_layer = build_lr_layer(k_reg)
    liner_output = lr_layer(layers.concatenate(spare_features))
    # 构建CIN
    CIN_layer = CIN(l2_reg=k_reg)(tf.stack(embed_features, axis=1))
    exFM_logit = layers.Dense(1, use_bias=False)(CIN_layer)
    # 构建DNN
    dnn_input = layers.concatenate(dense_features + embed_features)
    dnn_output = dnn(dnn_input, hidden_units, dropout_rate, k_reg)
    dnn_output = layers.Dense(1, use_bias=False)(dnn_output)
    output = layers.Add()([liner_output, dnn_output])
    # dnn_output = x
    # 汇总输出
    output = tf.math.sigmoid(tf.keras.layers.Add()([exFM_logit, output]))
    output = tf.reshape(output, (-1, 1))
    # 构建模型
    model = tf.keras.Model(input_map, output)

    return model
