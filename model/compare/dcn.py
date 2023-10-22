import tensorflow as tf
from keras import layers

from model.config.feature_config_without_multimodal_feature import build_input, build_embed_features, \
    build_spare_features, build_dense_features, build_lr_layer, dnn
from model.layer.cross_net import CrossNet


def dcn(input_config, spare_features_config, dense_features_config, hidden_units=[64, 32, 16], dropout_rate=0.5,
        k_reg=0.01):
    feature_input, feature_map, input_map = build_input(input_config)
    embed_features = build_embed_features(8, spare_features_config, feature_map)
    spare_features = build_spare_features(spare_features_config, feature_map)
    dense_features = build_dense_features(dense_features_config, feature_map)
    lr_layer = build_lr_layer(k_reg)
    liner_output = lr_layer(layers.concatenate(spare_features))
    # dnn与cross共享输入
    dnn_input = layers.concatenate(dense_features + embed_features)
    cross_input = layers.concatenate(dense_features + embed_features)

    dnn_output = dnn(dnn_input, hidden_units, dropout_rate, k_reg)
    dcn_output = CrossNet(k_reg=k_reg)(cross_input)
    stack = layers.Concatenate(axis=1)([dcn_output, dnn_output])
    dcn_dnn_output = layers.Dense(1, use_bias=False)(stack)
    output = tf.math.sigmoid(layers.Add()([dcn_dnn_output, liner_output]))
    output = tf.reshape(output, (-1, 1))
    model = tf.keras.Model(input_map, output)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), ]
                  )
    return model
