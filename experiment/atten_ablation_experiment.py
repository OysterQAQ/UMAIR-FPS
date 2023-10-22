import tensorflow as tf
from keras.metrics import AUC, Recall, Precision
from keras.optimizers import Adam

from model.config.feature_config import input_config, user_spare_faeture_config, user_dense_faeture_config, \
    item_spare_faeture_config, item_dense_faeture_config
from model.deepix_rec import user_aware_multi_modal_cross_network

dataset = tf.data.Dataset.load('/Volumes/Data/oysterqaq/Desktop/pixivic-rec-ds')
dataset = dataset.batch(128)
dataset = dataset.shuffle(16, reshuffle_each_iteration=True)
dataset = dataset.prefetch(buffer_size=10)

#无贡献度机制

model = user_aware_multi_modal_cross_network(input_config, user_spare_faeture_config, user_dense_faeture_config,
                                             item_spare_faeture_config,
                                             item_dense_faeture_config,k_reg=None,mmoe=False,passthrough=True,passthrough_with_split=True)
#单注意力机制实现贡献度
model = user_aware_multi_modal_cross_network(input_config, user_spare_faeture_config, user_dense_faeture_config,
                                             item_spare_faeture_config,
                                             item_dense_faeture_config,k_reg=None,mmoe=False)
#单senet实现贡献度
model = user_aware_multi_modal_cross_network(input_config, user_spare_faeture_config, user_dense_faeture_config,
                                             item_spare_faeture_config,
                                             item_dense_faeture_config,k_reg=None,mmoe=False,atten=False)

#MMOE思想 多注意力机制 3*atten
model = user_aware_multi_modal_cross_network(input_config, user_spare_faeture_config, user_dense_faeture_config,
                                             item_spare_faeture_config,
                                             item_dense_faeture_config,k_reg=None,mmoe=True,expert_type_list=["atten","atten","atten"])
#MMOE思想 多senet机制
model = user_aware_multi_modal_cross_network(input_config, user_spare_faeture_config, user_dense_faeture_config,
                                             item_spare_faeture_config,
                                             item_dense_faeture_config,k_reg=None,mmoe=True,expert_type_list=["senet","senet","senet"])


#MMOE思想 注意力和senet机制混合
model = user_aware_multi_modal_cross_network(input_config, user_spare_faeture_config, user_dense_faeture_config,
                                             item_spare_faeture_config,
                                             item_dense_faeture_config,k_reg=None,mmoe=True,expert_type_list=["senet","atten"])

