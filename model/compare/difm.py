import tensorflow as tf
# from utils import ouput_model_arch_to_image
from keras import backend as K
from keras import layers, Model
from keras.layers import Flatten, Layer, Add, Lambda, Dense

from model.config.feature_config_without_multimodal_feature import build_input, build_embed_features, \
    build_spare_features, build_dense_features, build_lr_layer, dnn
from model.layer.fm import InteractingLayer

class FM(layers.Layer):
    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        """
        inputs: 是一个列表，列表中每个元素的维度为：(None, 1, emb_dim)， 列表长度
            为field_num
        """
        # print(inputs.shape)
        # for input in inputs:

        # concated_embeds_value = tf.stack(inputs, axis=1)  # (None,field_num,emb_dim)
        concated_embeds_value = inputs
        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keepdims=True))  # (None, 1, emb_dim)
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value,
            axis=1, keepdims=True)  # (None, 1, emb_dim)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)  # (None,1)
        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        return super().get_config()

def concat_func(inputs, axis=-1, mask=False):
    if len(inputs) == 1:
        input = inputs[0]
        if not mask:
            input = NoMask()(input)
        return input
    return Concat(axis, supports_masking=mask)(inputs)


class Concat(Layer):
    def __init__(self, axis, supports_masking=True, **kwargs):
        super(Concat, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = supports_masking

    def call(self, inputs):
        return tf.concat(inputs, axis=self.axis)

    def compute_mask(self, inputs, mask=None):
        if not self.supports_masking:
            return None
        if mask is None:
            mask = [inputs_i._keras_mask if hasattr(inputs_i, "_keras_mask") else None for inputs_i in inputs]
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all([m is None for m in mask]):
            return None
        # Make a list of masks while making sure
        # the dimensionality of each mask
        # is the same as the corresponding input.
        masks = []
        for input_i, mask_i in zip(inputs, mask):
            if mask_i is None:
                # Input is unmasked. Append all 1s to masks,
                masks.append(tf.ones_like(input_i, dtype='bool'))
            elif K.ndim(mask_i) < K.ndim(input_i):
                # Mask is smaller than the input, expand it
                masks.append(tf.expand_dims(mask_i, axis=-1))
            else:
                masks.append(mask_i)
        concatenated = K.concatenate(masks, axis=self.axis)
        return K.all(concatenated, axis=-1, keepdims=False)

    def get_config(self, ):
        config = {'axis': self.axis, 'supports_masking': self.supports_masking}
        base_config = super(Concat, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NoMask(Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None


class _Add(Layer):
    def __init__(self, **kwargs):
        super(_Add, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(_Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if len(inputs) == 0:
            return tf.constant([[0.0]])

        return Add()(inputs)


def add_func(inputs):
    if not isinstance(inputs, list):
        return inputs
    if len(inputs) == 1:
        return inputs[0]
    return _Add()(inputs)


def difm(input_config, spare_features_config, dense_features_config, hidden_units=[64, 32, 16], dropout_rate=0.5,
         att_embedding_size=8, att_head_num=8, att_res=True, k_reg=0.01):
    """Instantiates the DIFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param att_embedding_size: integer, the embedding size in multi-head self-attention network.
    :param att_head_num: int. The head number in multi-head  self-attention network.
    :param att_res: bool. Whether or not use standard residual connections before output.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    feature_input, feature_map, input_map = build_input(input_config)
    embed_features = build_embed_features(8, spare_features_config, feature_map)
    spare_features = build_spare_features(spare_features_config, feature_map)
    dense_features = build_dense_features(dense_features_config, feature_map)

    att_out = InteractingLayer(att_embedding_size, att_head_num, att_res, scaling=True)(
        concat_func([tf.expand_dims(ef, axis=1) for ef in embed_features], axis=1))
    att_out = Flatten()(att_out)
    m_vec = Dense(len(spare_features), use_bias=False)(att_out)
    dnn_input = layers.concatenate(dense_features + embed_features)

    dnn_output = dnn(dnn_input, hidden_units, dropout_rate, k_reg)

    m_bit = Dense(len(spare_features), use_bias=False)(dnn_output)

    input_aware_factor = add_func([m_vec, m_bit])  # the complete input-aware factor m_x

    lr_layer = build_lr_layer(k_reg)
    liner_output = lr_layer(layers.concatenate(spare_features))

    fm_input = tf.stack(embed_features, axis=1)
    refined_fm_input = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=-1))(
        [fm_input, input_aware_factor])
    fm_logit = FM()(refined_fm_input)

    # output = layers.Dense(1, activation='sigmoid')(layers.concatenate([linear_logit, fm_logit]))
    output = tf.math.sigmoid(tf.keras.layers.Add()([liner_output, fm_logit]))
    output = tf.reshape(output, (-1, 1))
    model = Model(inputs=input_map, outputs=output)
    return model
