import tensorflow as tf
from keras import layers
from keras import regularizers

input_config = {
    'category': [
        {'feature': 'sanity_level', 'dtype': 'int32', 'num_tokens': 10, 'vocab': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
        {'feature': 'restrict', 'dtype': 'int32', 'num_tokens': 3, 'vocab': [0, 1, 3]},
        {'feature': 'x_restrict', 'dtype': 'int32', 'num_tokens': 3, 'vocab': [0, 1, 3]},
    ],
    # hash分桶
    # 后续添加统一的hash层
    'hash': [
        # {'feature': 'type', 'num_bins': 3, 'dtype': 'string', 'hash_layer_name': 'type_hash',
        #  'embed_layer_name': 'type_embed', },
        # {'feature': 'title', 'num_bins': 100, 'dtype': 'string', 'hash_layer_name': 'title_hash',
        #  'embed_layer_name': 'title_embed', },
        # {'feature': 'caption', 'num_bins': 1000, 'dtype': 'string', 'hash_layer_name': 'caption_hash',
        #  'embed_layer_name': 'caption_embed', },
        # {'feature': 'tools', 'num_bins': 10, 'dtype': 'string', 'hash_layer_name': 'tools_hash',
        #  'embed_layer_name': 'tools_embed', },
        {'feature': 'tag_0', 'num_bins': 8000, 'dtype': 'string', 'hash_layer_name': 'tag_hash',
         'embed_layer_name': 'tag_embed', },
        {'feature': 'tag_1', 'num_bins': 8000, 'dtype': 'string', 'hash_layer_name': 'tag_hash',
         'embed_layer_name': 'tag_embed', },
        {'feature': 'tag_2', 'num_bins': 8000, 'dtype': 'string', 'hash_layer_name': 'tag_hash',
         'embed_layer_name': 'tag_embed', },
        {'feature': 'tag_3', 'num_bins': 8000, 'dtype': 'string', 'hash_layer_name': 'tag_hash',
         'embed_layer_name': 'tag_embed', },

        {'feature': 'long_term_interest_tag_0', 'num_bins': 8000, 'dtype': 'string', 'hash_layer_name': 'tag_hash',
         'embed_layer_name': 'tag_embed', },
        {'feature': 'long_term_interest_tag_1', 'num_bins': 8000, 'dtype': 'string', 'hash_layer_name': 'tag_hash',
         'embed_layer_name': 'tag_embed', },
        {'feature': 'long_term_interest_tag_2', 'num_bins': 8000, 'dtype': 'string', 'hash_layer_name': 'tag_hash',
         'embed_layer_name': 'tag_embed', },
        {'feature': 'long_term_interest_tag_3', 'num_bins': 8000, 'dtype': 'string', 'hash_layer_name': 'tag_hash',
         'embed_layer_name': 'tag_embed', },
        {'feature': 'short_term_interest_tag_0', 'num_bins': 8000, 'dtype': 'string', 'hash_layer_name': 'tag_hash',
         'embed_layer_name': 'tag_embed', },
        {'feature': 'short_term_interest_tag_1', 'num_bins': 8000, 'dtype': 'string', 'hash_layer_name': 'tag_hash',
         'embed_layer_name': 'tag_embed', },
        {'feature': 'short_term_interest_tag_2', 'num_bins': 8000, 'dtype': 'string', 'hash_layer_name': 'tag_hash',
         'embed_layer_name': 'tag_embed', },
        {'feature': 'short_term_interest_tag_3', 'num_bins': 8000, 'dtype': 'string', 'hash_layer_name': 'tag_hash',
         'embed_layer_name': 'tag_embed', },

        {'feature': 'illust_id', 'dtype': 'int32', 'num_bins': 1000000, 'hash_layer_name': 'illust_id_hash',
         'embed_layer_name': 'illust_id_embed', },
        {'feature': 'user_id', 'dtype': 'int32', 'num_bins': 100000, 'hash_layer_name': 'user_id_hash',
         'embed_layer_name': 'user_id_embed', },
        {'feature': 'artist_id', 'dtype': 'int32', 'num_bins': 10000, 'hash_layer_name': 'artist_id_hash',
         'embed_layer_name': 'artist_id_embed', },

        {'feature': 'username', 'num_bins': 1000, 'dtype': 'string', 'hash_layer_name': 'username_hash',
         'embed_layer_name': 'username_embed', },
    ],
    # 数值分桶
    'int_bucket': [
        # {'feature': 'page_count', 'dtype': 'int32', 'bin_boundaries': [5, 10],  # 'embedding_dims': 10
        #  },
        {'feature': 'width', 'dtype': 'int32', 'bin_boundaries': [400, 800, 1600]},
        {'feature': 'height', 'dtype': 'int32', 'bin_boundaries': [400, 800, 1600]},
        {'feature': 'total_view', 'dtype': 'int32',
         'bin_boundaries': [100, 250, 500, 1000, 5000, 10000, 40000, 60000, 100000]},
        {'feature': 'total_bookmarks', 'dtype': 'int32',
         'bin_boundaries': [100, 250, 500, 1000, 1500, 2000, 3000, 6000, 10000]},
    ],
    # 数值类型（归一化）
    'num': [
        # {'feature': 'illust_id', 'dtype': 'float32'},
        # {'feature': 'create_date', 'dtype': 'float32'},
        #   {'feature': 'page_count', 'dtype': 'float32'},  # 需要分桶
        #   {'feature': 'width', 'dtype': 'float32'},  # 需要分桶
        #   {'feature': 'height', 'dtype': 'float32'},  # 需要分桶
        # {'feature': 'total_view', 'dtype': 'float32'},  # 需要分桶
        #  {'feature': 'total_bookmarks', 'dtype': 'float32'},  # 需要分桶
        # {'feature': 'artist_id', 'dtype': 'float32'},
        # {'feature': 'user_id', 'dtype': 'float32'},
        # {'feature': 'create_time', 'dtype': 'float32'},
    ],
    # 手动交叉
    'cross': [

    ],
    # 原始稠密特征
    'dense': [
        {'feature': 'long_term_interest_tags_feature', 'dtype': 'float32', 'dim': 512},
        {'feature': 'short_term_interest_tags_feature', 'dtype': 'float32', 'dim': 512},
        {'feature': 'img_semantics_feature', 'dtype': 'float32', 'dim': 1024},

        {'feature': 'tag_semantics_feature', 'dtype': 'float32', 'dim': 512},

    ]
}

item_dense_faeture_config = [
    # 'create_date',
    'img_semantics_feature',
    'tag_semantics_feature',
]

user_dense_faeture_config = [
    'long_term_interest_tags_feature',
    'short_term_interest_tags_feature',
    # 'create_time',
]

spec_embed_name = {
    'tag_0': 'tag',
    'tag_1': 'tag',
    'tag_2': 'tag',
    'tag_3': 'tag',
    'long_term_interest_tag_0': 'tag',
    'long_term_interest_tag_1': 'tag',
    'long_term_interest_tag_2': 'tag',
    'long_term_interest_tag_3': 'tag',

    'short_term_interest_tag_0': 'tag',
    'short_term_interest_tag_1': 'tag',
    'short_term_interest_tag_2': 'tag',
    'short_term_interest_tag_3': 'tag',

}

spare_features_config = [
    'restrict', 'sanity_level', 'x_restrict', 'tag_0', 'tag_1', 'tag_2', 'tag_3',
    'total_bookmarks', 'total_view', 'width', 'height',
    'illust_id', 'artist_id', 'user_id',
    'username',
    'long_term_interest_tag_0',
    'long_term_interest_tag_1',
    'long_term_interest_tag_2',
    'long_term_interest_tag_3',

    'short_term_interest_tag_0',
    'short_term_interest_tag_1',
    'short_term_interest_tag_2',
    'short_term_interest_tag_3',

]
dense_features_config = [
    # 'create_date',
    'img_semantics_feature',
    'img_style_feature',
    'tag_semantics_feature',
    'long_term_interest_tags_feature',
    'short_term_interest_tags_feature',
    # 'create_time',

]

item_spare_faeture_config = [
    'restrict', 'sanity_level', 'x_restrict', 'tag_0', 'tag_1', 'tag_2', 'tag_3',
    'total_bookmarks', 'total_view', 'width', 'height',
    'illust_id', 'artist_id',
]

user_spare_faeture_config = [
    'user_id',
    'username',
    'long_term_interest_tag_0',
    'long_term_interest_tag_1',
    'long_term_interest_tag_2',
    'long_term_interest_tag_3',

    'short_term_interest_tag_0',
    'short_term_interest_tag_1',
    'short_term_interest_tag_2',
    'short_term_interest_tag_3',

]


def build_input(input_config):
    feature_input = []
    feature_map = {}
    input_map = {}
    embedding_layer_map = {}
    hash_layer_map = {}
    # 构建连续数值型特征输入
    for num_feature in input_config.get('num', []):
        layer = tf.keras.Input(shape=(1,), dtype=num_feature['dtype'], name=num_feature[
            'feature'])
        input_map[num_feature['feature']] = layer
        feature_input.append(layer)  # tf.feature_column.numeric_column(num_feature['feature']))
        feature_map[num_feature['feature']] = layer
    # 构建分类特征输入
    for cate_feature in input_config.get('category', []):
        layer = layers.Input(shape=(1,), dtype=cate_feature['dtype'], name=cate_feature['feature'])
        input_map[cate_feature['feature']] = layer
        # 是否数字型
        if cate_feature.get('num_tokens') is None:
            if cate_feature['embed_layer_name'] is None:
                embed_layer = layers.StringLookup(vocabulary=cate_feature['vocabulary'], output_mode="one_hot",
                                                  num_oov_indices=0)
            else:
                if embedding_layer_map[cate_feature['embed_layer_name']] is None:
                    embed_layer = layers.StringLookup(vocabulary=cate_feature['vocabulary'], output_mode="one_hot",
                                                      name=cate_feature['embed_layer_name'],
                                                      num_oov_indices=0)
                    embedding_layer_map[cate_feature['embed_layer_name']] = embed_layer
                else:
                    embed_layer = embedding_layer_map[cate_feature['embed_layer_name']]
            layer = embed_layer(layer)
            input_dim = len(cate_feature['vocabulary'])
        else:
            if cate_feature.get('embed_layer_name', None) is None:
                embed_layer = layers.CategoryEncoding(num_tokens=cate_feature['num_tokens'], output_mode="one_hot")
            else:
                if embedding_layer_map.get(cate_feature['embed_layer_name']) is None:
                    embed_layer = layers.CategoryEncoding(num_tokens=cate_feature['num_tokens'], output_mode="one_hot",
                                                          name=cate_feature['embed_layer_name'])
                    embedding_layer_map[cate_feature['embed_layer_name']] = embed_layer
                else:
                    embed_layer = embedding_layer_map[cate_feature['embed_layer_name']]
            layer = embed_layer(layer)
            input_dim = cate_feature['num_tokens']
        # 是否需要embedding
        # if cate_feature.get('embedding_dims') is not None:
        #     layer = layers.Dense(cate_feature['embedding_dims'], use_bias=False)(layer)
        feature_input.append(layer)
        feature_map[cate_feature['feature']] = layer
    # 需要hash分桶的特征
    for hash_feature in input_config.get('hash', []):
        layer = tf.keras.Input(shape=(1,), dtype=hash_feature['dtype'], name=hash_feature['feature'])
        input_map[hash_feature['feature']] = layer
        if hash_layer_map.get(hash_feature['hash_layer_name']) is None:
            hash_layer = layers.Hashing(num_bins=hash_feature['num_bins'], output_mode='one_hot',
                                        name=hash_feature['hash_layer_name'])
            hash_layer_map[hash_feature['hash_layer_name']] = hash_layer
        else:
            hash_layer = hash_layer_map[hash_feature['hash_layer_name']]
        layer = hash_layer(layer)
        if hash_feature.get('embedding_dims') is not None:
            if embedding_layer_map.get(hash_feature['embed_layer_name']) is None:
                embed_layer = layers.Dense(hash_feature['embedding_dims'], use_bias=False,
                                           name=hash_feature['embed_layer_name'])
                embedding_layer_map[hash_feature['embed_layer_name']] = embed_layer
            else:
                embed_layer = embedding_layer_map[hash_feature['embed_layer_name']]

            layer = embed_layer(layer)
        feature_input.append(layer)
        feature_map[hash_feature['feature']] = layer
    # 连续数值分桶
    for bucket_feature in input_config.get('int_bucket', []):
        layer = tf.keras.Input(shape=(1,), dtype=bucket_feature['dtype'], name=bucket_feature['feature'])
        input_map[bucket_feature['feature']] = layer
        layer = layers.Discretization(bin_boundaries=bucket_feature['bin_boundaries'],
                                      output_mode='one_hot', )(layer)
        if bucket_feature.get('embedding_dims') is not None:
            embedding = layers.Dense(bucket_feature['embedding_dims'], use_bias=False)
            layer = embedding(layer)
        feature_input.append(layer)
        feature_map[bucket_feature['feature']] = layer
    for dense_feature in input_config.get('dense', []):
        layer = tf.keras.Input(shape=(dense_feature['dim'],), dtype=dense_feature['dtype'],
                               name=dense_feature['feature'])
        input_map[dense_feature['feature']] = layer
        feature_input.append(layer)
        feature_map[dense_feature['feature']] = layer
    cross_cate_map = {}
    # 构建交叉特征
    # for cross_feature in input_config.get('cross', []):
    #     col = []
    #     col = col + build_input(cross_feature['features'])
    #     # layer = layers.experimental.preprocessing.HashedCrossing(num_bins=cross_feature['num_bins'],
    #     #                                                                   output_mode='one_hot', sparse=True)(
    #     #     (tuple(col)))
    #     layer=tf.feature_column.indicator_column(tf.feature_column.crossed_column(col, 10000))
    #     feature_input.append(layer)
    #     feature_input_map[cross_feature['feature']] = layer

    return feature_input, feature_map, input_map


def build_embed_features(embedding_dims, spare_features_config, feature_input_map):
    embed_features = []
    embed_layer_map = {}
    for feature_name in spare_features_config:
        if spec_embed_name.get(feature_name) is None:
            embed_layer_name = feature_name + "_embed_" + str(embedding_dims)
        else:
            embed_layer_name = spec_embed_name[feature_name] + "_embed_" + str(embedding_dims)
        if embed_layer_map.get(embed_layer_name) is None:
            embedding = layers.Dense(embedding_dims, use_bias=False)
            embed_layer_map[embed_layer_name] = embedding
        else:
            embedding = embed_layer_map[embed_layer_name]
        embed_features.append(embedding(feature_input_map[feature_name]))

    return embed_features


def build_spare_features(spare_features_config, feature_input_map):
    spare_features = []
    for feature_name in spare_features_config:
        spare_features.append(feature_input_map[feature_name])
    return spare_features


def build_dense_features(dense_features_config, feature_input_map):
    dense_features = []
    for feature_name in dense_features_config:
        dense_features.append(feature_input_map[feature_name])
    return dense_features


def build_lr_layer(k_reg=0.01):
    lr_layer = layers.Dense(1, use_bias=False, kernel_regularizer=regularizers.l2(k_reg) if k_reg is not None else None)
    return lr_layer


def dnn(x, hidden_units=[64, 32, 16], dropout_rate=0.5, k_reg=0.01):
    for units in hidden_units:
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(k_reg) if k_reg is not None else None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if (dropout_rate is not None):
            x = layers.Dropout(dropout_rate)(x)
    return x
