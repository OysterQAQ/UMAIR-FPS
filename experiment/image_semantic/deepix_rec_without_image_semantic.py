import tensorflow as tf
from keras import Model
from keras import regularizers
from keras.layers import Dense, concatenate, Attention, GlobalAveragePooling1D, Add, BatchNormalization, ReLU,Layer,GlobalAveragePooling2D,Softmax
from keras.regularizers import l2
from .feature_config_without_image_semantic import build_input, build_embed_features, build_spare_features, \
    build_dense_features, \
    build_lr_layer, dnn
from model.layer.cross_net import CrossNetMix
from model.layer.fm import FM


def atten_for_multimodal(user_embedding, item_multi_modal,dropout):
    attention = Attention(use_scale=False, dropout=dropout)
    # 得到最终加权后的物品特征
    atten = attention([user_embedding, item_multi_modal])
    # pooling
    query_encoding = GlobalAveragePooling1D()(
        atten)
    return  query_encoding



def user_aware_senet_layer(user_embedding, item_multi_modal):

    #先对多模态特征做压缩
    squeeze = tf.math.reduce_mean(item_multi_modal,axis=-1)
    #两层mlp
    #先缩放到用户特征维度
    user_embedding_dim=user_embedding.shape[-1]
    filed_num=squeeze.shape[-1]
    squeeze = Dense(user_embedding_dim)(squeeze)
    #和用户特征点乘做联系
    squeeze_with_user_embedding = tf.multiply(squeeze, user_embedding)
    squeeze_with_user_embedding = ReLU()(squeeze_with_user_embedding)
    #再缩放到filed维度
    squeeze_with_user_embedding = Dense(filed_num)(squeeze_with_user_embedding)
    #和注意力匹配 这里把relu改成softmax
    squeeze_with_user_embedding = Softmax()(squeeze_with_user_embedding)
    print(squeeze_with_user_embedding.shape)
    print(item_multi_modal.shape)
    print( tf.expand_dims(squeeze_with_user_embedding, axis=-1).shape)
    output=tf.multiply(item_multi_modal, tf.expand_dims(squeeze_with_user_embedding, axis=-1))
    output=tf.keras.layers.GlobalAveragePooling1D()(
       output)
    return output





# passthrough是特征缩放
#
def user_aware_multi_modal_cross_network(input_config, user_spare_faeture_config, user_dense_faeture_config,
                                         item_spare_faeture_config,
                                         item_dense_faeture_config, k_reg=0.0001, scale_k_reg=0.0001, user_hidden_units=[128, 64],
                                         user_dropout_rate=0.5, item_hidden_units=[128, 64],
                                         item_dropout_rate=0.5, hidden_units=[64, 32, 16], dropout_rate=0.5,
                                         cross_layer_num=2, cross_layer_k_reg=None, residual=False, fm=False,
                                         modal_dim=64, modal_embed_bn=False,mmoe=False,expert_type_list=["atten","senet"],passthrough=False,passthrough_with_split=False,atten_dropout=0.5,has_style_feature=True):
    # 构建输入层（及其预处理逻辑）
    feature_input, feature_map, input_map = build_input(input_config)
    # 用户离散特征embed层
    user_embed_features = build_embed_features(8, user_spare_faeture_config, feature_map)
    # 用户密集特征层
    user_dense_features = build_dense_features(user_dense_faeture_config, feature_map)

    # 构建用户特征
    x = concatenate(user_embed_features + user_dense_features)


    # 物品离散特征embed层
    item_embed_features = build_embed_features(8, item_spare_faeture_config, feature_map)
    # 物品密集特征层
    item_dense_features = build_dense_features(item_dense_faeture_config, feature_map)

    # 构建物品特征
    # 将item_embed_features与user_dense_features中的create concate起来作为综合模态
    y = concatenate(item_embed_features)


    # 之后与 'item_semantics_feature' 'item_style_feature''item_tag_feature' 作为item四大模态数据 需要embedding成user_embedding的维度然后concate起来

    if passthrough:
        if passthrough_with_split:
            user_embedding = dnn(x, user_hidden_units, user_dropout_rate, k_reg=k_reg)
            item_spare_modal_embedding = dnn(y, item_hidden_units, item_dropout_rate, k_reg=k_reg)
            # 物品语义特征
            item_semantics_modal_embedding = Dense(modal_dim, activation='relu',kernel_regularizer=regularizers.l2(scale_k_reg) if scale_k_reg is not None else None)(item_dense_features[0])
            if modal_embed_bn:
                item_semantics_modal_embedding = BatchNormalization()(item_semantics_modal_embedding)
            # 物品风格特征
            item_style_modal_embedding = Dense(modal_dim, activation='relu',kernel_regularizer=regularizers.l2(scale_k_reg) if scale_k_reg is not None else None)(item_dense_features[1])
            if modal_embed_bn:
                item_style_modal_embedding = BatchNormalization()(item_style_modal_embedding)
            # 物品的标签的语义特征
            item_tag_semantics_modal_embedding = Dense(modal_dim, activation='relu',kernel_regularizer=regularizers.l2(scale_k_reg) if scale_k_reg is not None else None)(item_dense_features[2])
            if modal_embed_bn:
                item_tag_semantics_modal_embedding = BatchNormalization()(item_tag_semantics_modal_embedding)
            multi_modal_feature=[item_spare_modal_embedding, item_semantics_modal_embedding, item_style_modal_embedding,
             item_tag_semantics_modal_embedding, user_embedding]
            z = concatenate(multi_modal_feature)
        else:
            z =  concatenate([x, y] + item_dense_features)

    else:
        user_embedding = dnn(x, user_hidden_units, user_dropout_rate, k_reg=k_reg)
        item_spare_modal_embedding = dnn(y, item_hidden_units, item_dropout_rate, k_reg=k_reg)
        # 物品语义特征
        # item_semantics_modal_embedding = Dense(modal_dim, activation='relu',kernel_regularizer=regularizers.l2(scale_k_reg) if scale_k_reg is not None else None)(item_dense_features[0])
        # if modal_embed_bn:
        #     item_semantics_modal_embedding = BatchNormalization()(item_semantics_modal_embedding)
        # 物品风格特征
        item_style_modal_embedding = Dense(modal_dim, activation='relu', kernel_regularizer=regularizers.l2(
            scale_k_reg) if scale_k_reg is not None else None)(item_dense_features[1])
        if modal_embed_bn:
            item_style_modal_embedding = BatchNormalization()(item_style_modal_embedding)
        # 物品的标签的语义特征
        item_tag_semantics_modal_embedding = Dense(modal_dim, activation='relu',kernel_regularizer=regularizers.l2(scale_k_reg) if scale_k_reg is not None else None)(item_dense_features[1])
        if modal_embed_bn:
            item_tag_semantics_modal_embedding = BatchNormalization()(item_tag_semantics_modal_embedding)
        item_multi_modal = tf.stack(
            [item_spare_modal_embedding, item_style_modal_embedding,
             item_tag_semantics_modal_embedding], 1)
        if mmoe:
            # 多专家网络
            expert_outs = []
            for i, expert_type in enumerate(expert_type_list):
                if expert_type == "atten":
                    expert_outs.append(atten_for_multimodal(user_embedding, item_multi_modal,atten_dropout))
                elif expert_type == "senet":
                    expert_outs.append(user_aware_senet_layer(user_embedding, item_multi_modal))

            # 多专家注意力门控
            user_aware_multi_modal_feature = atten_for_multimodal(user_embedding, tf.stack(expert_outs, 1),dropout=0.0)
        else:
            user_aware_multi_modal_feature = atten_for_multimodal(user_embedding, item_multi_modal,atten_dropout)
            # user_aware_multi_modal_feature=user_aware_senet_layer(user_embedding, item_multi_modal)
        z = concatenate([user_aware_multi_modal_feature, user_embedding])

    # 物品特征与用户特征 concatenate进入特征交叉

    cross_net_work_output = CrossNetMix(layer_num=cross_layer_num, k_reg=cross_layer_k_reg)(z)
    # 残差链接
    if residual:
        cross_net_work_output = Add()([cross_net_work_output, z])
        cross_net_work_output = BatchNormalization()(cross_net_work_output)
        cross_net_work_output = ReLU()(cross_net_work_output)
    # 交叉结束后进入dnn输出
    dnn_output = dnn(cross_net_work_output, hidden_units, dropout_rate)
    dnn_output = Dense(1, use_bias=False)(dnn_output)
    # 用户与项目的离散特征进入lr层
    spare_feature = build_spare_features(user_spare_faeture_config + item_spare_faeture_config, feature_map)
    lr_layer = build_lr_layer(k_reg=k_reg)
    lr_output = lr_layer(concatenate(spare_feature))
    result = [lr_output, dnn_output]

    # embed后的离散特征拼接后进入fm
    if fm:
        fm_output = FM()(item_embed_features + user_embed_features)
        result.append(fm_output)

    # dnn输出与lr输出相加后进入sigmoid输出最终结果
    output = tf.math.sigmoid(Add()(result))
    output = tf.reshape(output, (-1, 1))
    model = Model(input_map, output)
    return model

# model = user_aware_multi_model_cross_network(input_config, user_spare_faeture_config, user_dense_faeture_config,
#                                              item_spare_faeture_config,
#                                              item_dense_faeture_config)
#
# model.compile(optimizer=Adam(learning_rate=1e-3),
#               loss="binary_crossentropy",
#               metrics=[AUC(name='auc'), Recall(), Precision(), ]
#               )
# model.summary()
# dataset = tf.data.Dataset.load('/Volumes/Data/oysterqaq/Desktop/pixivic-rec-ds')
# dataset = dataset.batch(256)
# for k, v in dataset.take(1):
#     print(v)
#
# model.fit(dataset, epochs=1)
