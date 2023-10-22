import tensorflow as tf
from keras import backend as K
from keras import layers, Model
from keras.layers import Layer, Dense, Activation, Dropout, BatchNormalization, \
    Reshape
from keras.regularizers import l2


try:
    from tensorflow.python.ops.init_ops import Zeros, Ones, Constant, TruncatedNormal, \
        glorot_normal_initializer as glorot_normal, \
        glorot_uniform_initializer as glorot_uniform
except ImportError:
    from tensorflow.python.ops.init_ops_v2 import Zeros, Ones, Constant, TruncatedNormal, glorot_normal, glorot_uniform

from model.config.feature_config_without_multimodal_feature import build_input, build_embed_features, \
    build_spare_features, build_lr_layer, dnn
class CrossNet(Layer):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **layer_num**: Positive integer, the cross layer number
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """

    def __init__(self, layer_num=2, parameterization='vector', l2_reg=0, seed=1024, **kwargs):
        self.layer_num = layer_num
        self.parameterization = parameterization
        self.l2_reg = l2_reg
        self.seed = seed
        print('CrossNet parameterization:', self.parameterization)
        super(CrossNet, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = int(input_shape[-1])
        if self.parameterization == 'vector':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, 1),
                                            initializer=glorot_normal(
                                                seed=self.seed),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True) for i in range(self.layer_num)]
        elif self.parameterization == 'matrix':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, dim),
                                            initializer=glorot_normal(
                                                seed=self.seed),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True) for i in range(self.layer_num)]
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        # Be sure to call this somewhere!
        super(CrossNet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))
                dot_ = tf.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = tf.einsum('ij,bjk->bik', self.kernels[i], x_l)  # W * xi  (bs, dim, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

    def get_config(self, ):

        config = {'layer_num': self.layer_num, 'parameterization': self.parameterization,
                  'l2_reg': self.l2_reg, 'seed': self.seed}
        base_config = super(CrossNet, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape


class DNN(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [Activation(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = Activation(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            try:
                fc = self.activation_layers[i](fc, training=training)
            except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                print("make sure the activation function use training flag properly", e)
                fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RegulationModule(Layer):
    """Regulation module used in EDCN.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size,field_size * embedding_size)``.

      Arguments
        - **tau** : Positive float, the temperature coefficient to control
        distribution of field-wise gating unit.

      References
        - [Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models.](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf)
    """

    def __init__(self, tau=1.0, **kwargs):
        if tau == 0:
            raise ValueError("RegulationModule tau can not be zero.")
        self.tau = 1.0 / tau
        super(RegulationModule, self).__init__(**kwargs)

    def build(self, input_shape):
        self.field_size = int(input_shape[1])
        self.embedding_size = int(input_shape[2])
        self.g = self.add_weight(
            shape=(1, self.field_size, 1),
            initializer=Ones(),
            name=self.name + '_field_weight')

        # Be sure to call this somewhere!
        super(RegulationModule, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        feild_gating_score = tf.nn.softmax(self.g * self.tau, 1)
        E = inputs * feild_gating_score
        return tf.reshape(E, [-1, self.field_size * self.embedding_size])

    def compute_output_shape(self, input_shape):
        return (None, self.field_size * self.embedding_size)

    def get_config(self):
        config = {'tau': self.tau}
        base_config = super(RegulationModule, self).get_config()
        base_config.update(config)
        return base_config


class BridgeModule(Layer):
    """Bridge Module used in EDCN

      Input shape
        - A list of two 2D tensor with shape: ``(batch_size, units)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.

    Arguments
        - **bridge_type**: The type of bridge interaction, one of 'pointwise_addition', 'hadamard_product', 'concatenation', 'attention_pooling'

        - **activation**: Activation function to use.

      References
        - [Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models.](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf)

    """

    def __init__(self, bridge_type='hadamard_product', activation='relu', **kwargs):
        self.bridge_type = bridge_type
        self.activation = activation

        super(BridgeModule, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError(
                'A `BridgeModule` layer should be called '
                'on a list of 2 inputs')

        self.dnn_dim = int(input_shape[0][-1])
        if self.bridge_type == "concatenation":
            self.dense = Dense(self.dnn_dim, self.activation)
        elif self.bridge_type == "attention_pooling":
            self.dense_x = DNN([self.dnn_dim, self.dnn_dim], self.activation, output_activation='softmax')
            self.dense_h = DNN([self.dnn_dim, self.dnn_dim], self.activation, output_activation='softmax')

        super(BridgeModule, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        x, h = inputs
        if self.bridge_type == "pointwise_addition":
            return x + h
        elif self.bridge_type == "hadamard_product":
            return x * h
        elif self.bridge_type == "concatenation":
            return self.dense(tf.concat([x, h], axis=-1))
        elif self.bridge_type == "attention_pooling":
            a_x = self.dense_x(x)
            a_h = self.dense_h(h)
            return a_x * x + a_h * h

    def compute_output_shape(self, input_shape):
        return (None, self.dnn_dim)

    def get_config(self):
        base_config = super(BridgeModule, self).get_config().copy()
        config = {
            'bridge_type': self.bridge_type,
            'activation': self.activation
        }
        config.update(base_config)
        return config


def edcn(input_config, spare_features_config, dense_features_config, hidden_units=[64, 32, 16], dropout_rate=0.5,
         cross_num=3,
         cross_parameterization='vector',
         bridge_type='concatenation',
         tau=1.0,
         l2_reg_cross=1e-5,
         k_reg=0.01):
    """Instantiates the Enhanced Deep&Cross Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param cross_num: positive integet,cross layer number
    :param cross_parameterization: str, ``"vector"`` or ``"matrix"``, how to parameterize the cross network.
    :param bridge_type: The type of bridge interaction, one of ``"pointwise_addition"``, ``"hadamard_product"``, ``"concatenation"`` , ``"attention_pooling"``
    :param tau: Positive float, the temperature coefficient to control distribution of field-wise gating unit
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_cross: float. L2 regularizer strength applied to cross net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not DNN
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """
    if cross_num == 0:
        raise ValueError("Cross layer num must > 0")

    print('EDCN brige type: ', bridge_type)

    feature_input, feature_map, input_map = build_input(input_config)

    spare_features = build_spare_features(spare_features_config, feature_map)
    # dense_features = build_dense_features(dense_features_config, feature_map)

    embed_features = build_embed_features(8, spare_features_config + dense_features_config, feature_map)
    deep_in = RegulationModule(tau)(tf.stack(embed_features, axis=1))
    cross_in = RegulationModule(tau)(tf.stack(embed_features, axis=1))

    field_size = len(embed_features)
    embedding_size = int(embed_features[0].shape[-1])
    cross_dim = field_size * embedding_size
    lr_layer = build_lr_layer(k_reg)
    liner_output = lr_layer(layers.concatenate(spare_features))
    for i in range(cross_num):
        cross_out = CrossNet(3, parameterization=cross_parameterization,
                             l2_reg=l2_reg_cross)(cross_in)
        deep_out = dnn(deep_in, [cross_dim], dropout_rate, k_reg)
        # deep_out = DNN([cross_dim], dnn_activation, l2_reg_dnn,
        #                dnn_dropout, dnn_use_bn, seed=seed)(deep_in)
        # print(cross_out, deep_out)
        bridge_out = BridgeModule(bridge_type)([cross_out, deep_out])
        if i + 1 < cross_num:
            bridge_out_list = Reshape([field_size, embedding_size])(bridge_out)
            deep_in = RegulationModule(tau)(bridge_out_list)
            cross_in = RegulationModule(tau)(bridge_out_list)

    stack_out = layers.concatenate([cross_out, deep_out, bridge_out])
    final_logit = Dense(1, use_bias=False)(stack_out)

    output = tf.sigmoid(layers.Add()([final_logit, liner_output]))
    output = tf.reshape(output, (-1, 1))
    model = Model(inputs=input_map, outputs=output)

    return model
