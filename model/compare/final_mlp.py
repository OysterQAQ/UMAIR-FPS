import tensorflow as tf
from model.config.feature_config_without_multimodal_feature import build_input, build_embed_features, \
    build_spare_features, build_dense_features, build_lr_layer, dnn

from keras import layers

class MLP(tf.keras.layers.Layer):
    def __init__(self, num_hidden, dim_hidden, dim_out=None, dropout=0.0, batch_norm=True, name="MLP"):
        super().__init__(name=name)

        self.dim_out = dim_out
        self.blocks = tf.keras.Sequential(name="MLP")
        for i in range(num_hidden - 1):
            self.blocks.add(tf.keras.layers.Dense(dim_hidden[i]))

            if batch_norm:
                self.blocks.add(tf.keras.layers.BatchNormalization())

            self.blocks.add(tf.keras.layers.ReLU())
            self.blocks.add(tf.keras.layers.Dropout(dropout))

        if dim_out:
            self.blocks.add(tf.keras.layers.Dense(dim_out))
        else:
            self.blocks.add(tf.keras.layers.Dense(dim_hidden[-1]))

    def call(self, inputs, training=None):
        return self.blocks(inputs, training=training)


class FeatureSelection(tf.keras.layers.Layer):
    def __init__(self, dim_feature, dim_gate, num_hidden=1, dim_hidden=64, dropout=0.0):
        super().__init__()

        self.gate_1 = MLP(
            num_hidden=num_hidden,
            dim_hidden=[dim_hidden]*num_hidden,
            dim_out=dim_feature,
            dropout=dropout,
            batch_norm=False,
            name="feature_selection_gate_1",
        )
        self.gate_1_bias = self.add_weight(shape=(1, dim_gate), initializer="ones", trainable=True)

        self.gate_2 = MLP(
            num_hidden=num_hidden,
            dim_hidden=[dim_hidden]*num_hidden,
            dim_out=dim_feature,
            dropout=dropout,
            batch_norm=False,
            name="feature_selection_gate_2",
        )
        self.gate_2_bias = self.add_weight(shape=(1, dim_gate), initializer="ones", trainable=True)

    def call(self, embeddings, training=False):
        # embeddings is of shape (batch_size, dim_feature)
        gate_score_1 = self.gate_1(self.gate_1_bias, training=training)  # (1, dim_feature)
        out_1 = 2.0 * tf.nn.sigmoid(gate_score_1) * embeddings  # (bs, dim_feature)

        gate_score_2 = self.gate_2(self.gate_2_bias, training=training)  # (1, dim_feature)
        out_2 = 2.0 * tf.nn.sigmoid(gate_score_2) * embeddings  # (bs, dim_feature)

        return out_1, out_2  # (bs, dim_feature), (bs, dim_feature)


class Aggregation(tf.keras.layers.Layer):
    def __init__(self, dim_latent_1, dim_latent_2, num_heads=1):
        super().__init__()

        self.num_heads = num_heads
        self.dim_head_1 = dim_latent_1 // num_heads
        self.dim_head_2 = dim_latent_2 // num_heads

        self.w_1 = self.add_weight(shape=(self.dim_head_1, num_heads, 1), initializer="glorot_uniform", trainable=True)
        self.w_2 = self.add_weight(shape=(self.dim_head_2, num_heads, 1), initializer="glorot_uniform", trainable=True)
        self.w_12 = self.add_weight(
            shape=(num_heads, self.dim_head_1, self.dim_head_2, 1), initializer="glorot_uniform", trainable=True
        )
        self.bias = self.add_weight(shape=(1, num_heads, 1), initializer="zeros", trainable=True)

    def call(self, latent_1, latent_2):
        # bilinear aggregation of the two latent representations
        # y = b + w_1.T o_1 + w_2.T o_2 + o_1.T W_3 o_2
        latent_1 = tf.reshape(latent_1, (-1, self.num_heads, self.dim_head_1))  # (bs, num_heads, dim_head_1)
        latent_2 = tf.reshape(latent_2, (-1, self.num_heads, self.dim_head_2))  # (bs, num_heads, dim_head_2)

        first_order = tf.einsum("bhi,iho->bho", latent_1, self.w_1)  # (bs, num_heads, 1)
        first_order += tf.einsum("bhi,iho->bho", latent_2, self.w_2)  # (bs, num_heads, 1)
        second_order = tf.einsum("bhi,hijo,bhj->bho", latent_1, self.w_12, latent_2)  # (bs, num_heads, 1)

        out = tf.reduce_sum(first_order + second_order + self.bias, 1)  # (bs, 1)

        return out


class FinalMLP(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=32,
        dim_hidden_fs=64,
        num_hidden_1=2,
        dim_hidden_1=64,
        num_hidden_2=2,
        dim_hidden_2=64,
        num_heads=1,
        dropout=0.0,
        name="FinalMLP",
    ):
        super().__init__(name=name)

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        # embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            input_length=dim_input,
            name="embedding",
        )

        # feature selection layer that projects a learnable vector to the flatened embedded feature space
        self.feature_selection = FeatureSelection(
            dim_feature=dim_input * dim_embedding,
            dim_gate=dim_input,
            dim_hidden=dim_hidden_fs,
            dropout=dropout,
        )

        # branch 1
        self.interaction_1 = MLP(
            num_hidden=num_hidden_1,
            dim_hidden=[dim_hidden_1]*num_hidden_1,
            dropout=dropout,
            name="MLP_1",
        )
        # branch 2
        self.interaction_2 = MLP(
            num_hidden=num_hidden_2,
            dim_hidden=[dim_hidden_2]*num_hidden_2,
            dropout=dropout,
            name="MLP_2",
        )

        # final aggregation layer
        self.aggregation = Aggregation(dim_latent_1=dim_hidden_1, dim_latent_2=dim_hidden_2, num_heads=num_heads)

    def call(self, inputs, training=None):
        embeddings = self.embedding(inputs, training=training)  # (bs, num_emb, dim_emb)

        # (bs, num_emb * dim_emb)
        embeddings = tf.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))

        # weight features of the two streams using a gating mechanism
        emb_1, emb_2 = self.feature_selection(
            embeddings, training=training
        )  # (bs, num_emb * dim_emb), (bs, num_emb * dim_emb)

        # get interactions from the two branches
        # (bs, dim_hidden_1), (bs, dim_hidden_1)
        latent_1 = self.interaction_1(emb_1, training=training)
        latent_2 = self.interaction_2(emb_2, training=training)

        # merge the representations using an aggregation scheme
        logits = self.aggregation(latent_1, latent_2)  # (bs, 1)
        outputs = tf.nn.sigmoid(logits)  # (bs, 1)

        return outputs


def final_mlp(input_config, spare_features_config, dense_features_config,
        dim_hidden_fs=512,
        num_hidden_1=8,
        dim_hidden_1=[512,512,512,512,256,256,128,128],
        num_hidden_2=6,
        dim_hidden_2=[128,128,64,64,32,32],
        num_heads=2,
        dropout=0.5,):
    feature_input, feature_map, input_map = build_input(input_config)
    embed_features = build_embed_features(8, spare_features_config, feature_map)
    dense_features = build_dense_features(dense_features_config, feature_map)
    inputs=embed_features+dense_features
    embeddings=layers.concatenate(inputs)

    feature_selection = FeatureSelection(
        dim_feature=embeddings.shape[-1],
        dim_gate=len(inputs),
        dim_hidden=dim_hidden_fs,
        dropout=dropout,
    )


    # branch 1
    interaction_1 = MLP(
        num_hidden=num_hidden_1,
        dim_hidden=dim_hidden_1,
        dropout=dropout,
        name="MLP_1",
    )
    # branch 2
    interaction_2 = MLP(
        num_hidden=num_hidden_2,
        dim_hidden=dim_hidden_2,
        dropout=dropout,
        name="MLP_2",
    )

    # final aggregation layer
    aggregation = Aggregation(dim_latent_1=dim_hidden_1[-1], dim_latent_2=dim_hidden_2[-1], num_heads=num_heads)
    training=True
    emb_1, emb_2 = feature_selection(
        embeddings, training=training
    )  # (bs, num_emb * dim_emb), (bs, num_emb * dim_emb)

    # get interactions from the two branches
    # (bs, dim_hidden_1), (bs, dim_hidden_1)
    latent_1 = interaction_1(emb_1, training=training)
    latent_2 = interaction_2(emb_2, training=training)

    # merge the representations using an aggregation scheme
    logits = aggregation(latent_1, latent_2)  # (bs, 1)
    outputs = tf.nn.sigmoid(logits)  # (bs, 1)
    model = tf.keras.Model(input_map, outputs)
    return model









