import tensorflow as tf
from keras import backend as K
from keras.regularizers import l2

try:
    from tensorflow.python.ops.init_ops import Zeros, Ones, Constant, TruncatedNormal, \
        glorot_normal_initializer as glorot_normal, \
        glorot_uniform_initializer as glorot_uniform
except ImportError:
    from tensorflow.python.ops.init_ops_v2 import Zeros, Ones, Constant, TruncatedNormal, glorot_normal, glorot_uniform
from keras.layers import Dense


class CrossNetMix(tf.keras.layers.Layer):
    """The Cross Network part of DCN-Mix model, which improves DCN-M by:
      1 add MOE to learn feature interactions in different subspaces
      2 add nonlinear transformations in low-dimensional space

      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.

      Arguments
        - **low_rank** : Positive integer, dimensionality of low-rank sapce.

        - **num_experts** : Positive integer, number of experts.

        - **layer_num**: Positive integer, the cross layer number

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix

        - **seed**: A Python integer to use as random seed.

      References
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, low_rank=32, num_experts=4, layer_num=2, k_reg=None, seed=1024, **kwargs):
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.layer_num = layer_num
        self.k_reg = k_reg
        self.seed = seed
        super(CrossNetMix, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = int(input_shape[-1])

        # U: (dim, low_rank)
        self.U_list = [self.add_weight(name='U_list' + str(i),
                                       shape=(self.num_experts, dim, self.low_rank),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.k_reg) if self.k_reg is not None else None,
                                       trainable=True) for i in range(self.layer_num)]
        # V: (dim, low_rank)
        self.V_list = [self.add_weight(name='V_list' + str(i),
                                       shape=(self.num_experts, dim, self.low_rank),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.k_reg),
                                       trainable=True) for i in range(self.layer_num)]
        # C: (low_rank, low_rank)
        self.C_list = [self.add_weight(name='C_list' + str(i),
                                       shape=(self.num_experts, self.low_rank, self.low_rank),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.k_reg),
                                       trainable=True) for i in range(self.layer_num)]

        self.gating = [Dense(1, use_bias=False) for i in range(self.num_experts)]

        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        # Be sure to call this somewhere!
        super(CrossNetMix, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](tf.squeeze(x_l, axis=2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = tf.einsum('ij,bjk->bik', tf.transpose(self.V_list[i][expert_id]), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = tf.nn.tanh(v_x)
                v_x = tf.einsum('ij,bjk->bik', self.C_list[i][expert_id], v_x)  # (bs, low_rank, 1)
                v_x = tf.nn.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = tf.einsum('ij,bjk->bik', self.U_list[i][expert_id], v_x)  # (bs, dim, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(tf.squeeze(dot_, axis=2))

            # (3) mixture of low-rank experts
            output_of_experts = tf.stack(output_of_experts, 2)  # (bs, dim, num_experts)
            gating_score_of_experts = tf.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = tf.matmul(output_of_experts, tf.nn.softmax(gating_score_of_experts, 1))
            x_l = moe_out + x_l  # (bs, dim, 1)
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

    def get_config(self, ):

        config = {'low_rank': self.low_rank, 'num_experts': self.num_experts, 'layer_num': self.layer_num,
                  'l2_reg': self.k_reg, 'seed': self.seed}
        base_config = super(CrossNetMix, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape


class CrossNet(tf.keras.layers.Layer):
    def __init__(self, layer_nums=2, k_reg=0.05):
        super(CrossNet, self).__init__()
        self.layer_nums = layer_nums
        self.k_reg = k_reg

    def build(self, input_shape):
        # 计算w的维度，w的维度与输入数据的最后一个维度相同
        self.dim = int(input_shape[-1])

        # 注意，在DCN中W不是一个矩阵而是一个向量，这里根据残差的层数定义一个权重列表
        self.W = [
            self.add_weight(name='W_' + str(i), shape=(self.dim,), regularizer=tf.keras.regularizers.l2(self.k_reg)) for
            i in range(self.layer_nums)]
        self.b = [self.add_weight(name='b_' + str(i), shape=(self.dim,), initializer='zeros') for i in
                  range(self.layer_nums)]

    def call(self, inputs):
        # 进行特征交叉时的x_0一直没有变，变的是x_l和每一层的权重
        x_0 = inputs  # B x dims
        x_l = x_0
        for i in range(self.layer_nums):
            # 将x_l的第一个维度与w[i]的第0个维度计算点积
            xl_w = tf.tensordot(x_l, self.W[i], axes=(1, 0))  # B,
            xl_w = tf.expand_dims(xl_w, axis=-1)  # 在最后一个维度上添加一个维度 # B x 1
            cross = tf.multiply(x_0, xl_w)  # B x dims
            x_l = cross + self.b[i] + x_l

        return x_l


class CIN(tf.keras.layers.Layer):
    def __init__(self, cin_size=[64, 64], l2_reg=1e-4):
        """
        :param: cin_size: A list. [H_1, H_2, ....H_T], a list of number of layers
        """
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # input_shape  [None, field_nums, embedding_dim]
        self.field_nums = input_shape[1]

        # CIN 的每一层大小，这里加入第0层，也就是输入层H_0
        self.field_nums = [self.field_nums] + self.cin_size

        # 过滤器
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),  # 这个大小要理解
                initializer='random_uniform',
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True
            )
            for i in range(len(self.field_nums) - 1)
        }

        super(CIN, self).build(input_shape)

    def call(self, inputs):
        # inputs [None, field_num, embed_dim]
        embed_dim = inputs.shape[-1]
        hidden_layers_results = [inputs]

        # 从embedding的维度把张量一个个的切开,这个为了后面逐通道进行卷积，算起来好算
        # 这个结果是个list， list长度是embed_dim, 每个元素维度是[None, field_nums[0], 1]  field_nums[0]即输入的特征个数
        # 即把输入的[None, field_num, embed_dim]，切成了embed_dim个[None, field_nums[0], 1]的张量
        split_X_0 = tf.split(hidden_layers_results[0], embed_dim, 2)

        for idx, size in enumerate(self.cin_size):
            # 这个操作和上面是同理的，也是为了逐通道卷积的时候更加方便，分割的是当一层的输入Xk-1
            split_X_K = tf.split(hidden_layers_results[-1], embed_dim,
                                 2)  # embed_dim个[None, field_nums[i], 1] feild_nums[i] 当前隐藏层单元数量

            # 外积的运算
            out_product_res_m = tf.matmul(split_X_0, split_X_K,
                                          transpose_b=True)  # [embed_dim, None, field_nums[0], field_nums[i]]
            out_product_res_o = tf.reshape(out_product_res_m,
                                           shape=[embed_dim, -1, self.field_nums[0] * self.field_nums[idx]])  # 后两维合并起来
            out_product_res = tf.transpose(out_product_res_o,
                                           perm=[1, 0, 2])  # [None, dim, field_nums[0]*field_nums[i]]

            # 卷积运算
            # 这个理解的时候每个样本相当于1张通道为1的照片 dim为宽度， field_nums[0]*field_nums[i]为长度
            # 这时候的卷积核大小是field_nums[0]*field_nums[i]的, 这样一个卷积核的卷积操作相当于在dim上进行滑动，每一次滑动会得到一个数
            # 这样一个卷积核之后，会得到dim个数，即得到了[None, dim, 1]的张量， 这个即当前层某个神经元的输出
            # 当前层一共有field_nums[i+1]个神经元， 也就是field_nums[i+1]个卷积核，最终的这个输出维度[None, dim, field_nums[i+1]]
            cur_layer_out = tf.nn.conv1d(input=out_product_res, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                         padding='VALID')

            cur_layer_out = tf.transpose(cur_layer_out, perm=[0, 2, 1])  # [None, field_num[i+1], dim]

            hidden_layers_results.append(cur_layer_out)

        # 最后CIN的结果，要取每个中间层的输出，这里不要第0层的了
        final_result = hidden_layers_results[1:]  # 这个的维度T个[None, field_num[i], dim]  T 是CIN的网络层数

        # 接下来在第一维度上拼起来
        result = tf.concat(final_result, axis=1)  # [None, H1+H2+...HT, dim]
        # 接下来， dim维度上加和，并把第三个维度1干掉
        result = tf.reduce_sum(result, axis=-1, keepdims=False)  # [None, H1+H2+..HT]

        return result
