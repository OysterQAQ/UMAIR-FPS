import tensorflow as tf





# 定义一个简单的模型示例（可以替换为你自己的模型）
class RankingModel(tf.keras.Model):
    def __init__(self):
        super(RankingModel, self).__init__()
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        return self.dense(inputs)


# 创建模型实例
model = RankingModel()

# 定义正样本和负样本的输入数据（示例数据）
positive_data = tf.constant([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=tf.float32)
negative_data = tf.constant([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=tf.float32)

# 使用模型生成正样本和负样本的分数
score_positive = model(positive_data)
score_negative = model(negative_data)

# 定义最大边际排名损失
margin = 1.0
loss = tf.reduce_mean(tf.maximum(0.0, margin - (score_positive - score_negative)))

# 创建优化器
optimizer = tf.optimizers.Adam(learning_rate=0.001)

# 在训练过程中，使用GradientTape记录梯度并应用优化器
for epoch in range(100):
    #从数据集取batch 一个样本有三个图
    with tf.GradientTape() as tape:
        score_positive = model(positive_data)
        score_negative = model(negative_data)
        loss_value = tf.reduce_mean(tf.maximum(0.0, margin - (score_positive - score_negative)))
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print(f'Epoch {epoch + 1}, Loss: {loss_value.numpy()}')
