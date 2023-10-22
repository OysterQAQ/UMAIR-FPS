import tensorflow as tf
from keras.metrics import AUC, Recall, Precision
from keras.optimizers import Adam
import sys
sys.path.append("..")
from experiment.image_style.feature_config_without_style import input_config, user_spare_faeture_config, user_dense_faeture_config, \
    item_spare_faeture_config, item_dense_faeture_config
from experiment.image_style.deepix_rec_without_style import user_aware_multi_modal_cross_network

dataset = tf.data.Dataset.load('/Volumes/Home/oysterqaq/dataset/pixivic-rec-ds')
dataset = dataset.batch(128)
dataset = dataset.shuffle(16, reshuffle_each_iteration=True)
dataset = dataset.prefetch(buffer_size=10)
#风格模态消融
#重新写个输入数据构造


model = user_aware_multi_modal_cross_network(input_config, user_spare_faeture_config, user_dense_faeture_config,
                                             item_spare_faeture_config,
                                             item_dense_faeture_config,k_reg=None,mmoe=False)
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss="binary_crossentropy",
              metrics=[AUC(name='auc'), Recall(), Precision(), ]
              )
model.fit(dataset, epochs=1)

test_dataset = tf.data.Dataset.load('/sys-backup/pixivic_rec_dataset/pixivic-rec-ds_test')
test_dataset = test_dataset.batch(512)
test_dataset = test_dataset.shuffle(512, reshuffle_each_iteration=True)
test_dataset = test_dataset.prefetch(buffer_size=10)
model.evaluate(
    test_dataset
)
from experiment.image_semantic.feature_config_without_image_semantic import input_config, user_spare_faeture_config, user_dense_faeture_config, \
    item_spare_faeture_config, item_dense_faeture_config
from experiment.image_semantic.deepix_rec_without_image_semantic import user_aware_multi_modal_cross_network



from experiment.label_semantic.feature_config_without_label_semantic import input_config, user_spare_faeture_config, user_dense_faeture_config, \
    item_spare_faeture_config, item_dense_faeture_config
from experiment.label_semantic.deepix_rec_without_label_semantic import user_aware_multi_modal_cross_network
