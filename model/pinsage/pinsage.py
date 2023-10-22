from typing import Any, Callable
from typing import Optional

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.graph import broadcast_ops
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import pool_ops
from tensorflow_gnn.graph import tag_utils
from tensorflow_gnn.models import graph_sage
from tensorflow_gnn.models.graph_sage import GraphSAGEPoolingConv

Field = const.Field
FieldName = const.FieldName
NodeSetName = const.NodeSetName
EdgeSetName = const.EdgeSetName
IncidentNodeTag = const.IncidentNodeTag
IncidentNodeOrContextTag = const.IncidentNodeOrContextTag
GraphTensor = gt.GraphTensor


def neighbour_weight(graph_tensor: GraphTensor,
                     edge_set_name: EdgeSetName,
                     node_tag: IncidentNodeTag,
                     *,
                     feature_value: Optional[Field] = None,
                     feature_name: Optional[FieldName] = None) -> Field:
    return tf.expand_dims(graph_tensor.edge_sets[edge_set_name].features["w"], axis=1)


def neighbour_weight_test(graph, receiver_tag, edge_set_name, feature_value):
    return tf.expand_dims(graph.edge_sets[edge_set_name].features["w"], axis=1)


def _get_init_or_call_arg(class_name, arg_name, init_value, call_value):
    """Returns unified value for arg that can be set at init or call time."""
    if call_value is None:
        if init_value is None:
            raise ValueError(
                f"{class_name} requires {arg_name} to be set at init or call time")
        return init_value
    else:
        if init_value not in [None, call_value]:
            raise ValueError(
                f"{class_name}(..., {arg_name}={init_value})"
                f"was called with contradictory value {arg_name}={call_value}")
        return call_value


class PinSAGEPoolingConv(GraphSAGEPoolingConv):

    def __init__(
            self, **kwargs):
        kwargs.setdefault("name", "pin_sage_pooling_conv")
        kwargs.setdefault("reduce_type", "mean")
        super().__init__(
            **kwargs)

    def call(self, graph: gt.GraphTensor, *,
             edge_set_name: Optional[gt.EdgeSetName] = None,
             node_set_name: Optional[gt.NodeSetName] = None,
             receiver_tag: Optional[const.IncidentNodeOrContextTag] = None,
             training: Optional[bool] = False) -> tf.Tensor:
        # pylint: disable=g-long-lambda

        # Normalize inputs.
        class_name = self.__class__.__name__
        gt.check_scalar_graph_tensor(graph, class_name)
        receiver_tag = _get_init_or_call_arg(class_name, "receiver_tag",
                                             self._receiver_tag, receiver_tag)

        # Find the receiver graph piece (NodeSet or Context), the EdgeSet (if any)
        # and the sender NodeSet (if any) with its broadcasting function.
        if receiver_tag == const.CONTEXT:
            if (edge_set_name is None) + (node_set_name is None) != 1:
                raise ValueError(
                    "Must pass exactly one of edge_set_name, node_set_name "
                    "for receiver_tag CONTEXT.")
            if edge_set_name is not None:
                # Pooling from EdgeSet to Context; no node set involved.
                name_kwarg = dict(edge_set_name=edge_set_name)
                edge_set = graph.edge_sets[edge_set_name]
                sender_node_set = None
                broadcast_from_sender_node = None
            else:
                # Pooling from NodeSet to Context, no EdgeSet involved.
                name_kwarg = dict(node_set_name=node_set_name)
                edge_set = None
                sender_node_set = graph.node_sets[node_set_name]
                # Values are computed per sender node, no need to broadcast
                broadcast_from_sender_node = lambda feature_value: feature_value
            receiver_piece = graph.context
        else:
            # Convolving from nodes to nodes.
            if edge_set_name is None or node_set_name is not None:
                raise ValueError("Must pass edge_set_name, not node_set_name")
            name_kwarg = dict(edge_set_name=edge_set_name)
            edge_set = graph.edge_sets[edge_set_name]
            sender_node_tag = tag_utils.reverse_tag(receiver_tag)
            sender_node_set = graph.node_sets[
                edge_set.adjacency.node_set_name(sender_node_tag)]
            broadcast_from_sender_node = (
                lambda feature_value: broadcast_ops.broadcast_node_to_edges(
                    graph, edge_set_name, sender_node_tag,
                    feature_value=feature_value))
            neighbour_weight_from_edge = (
                lambda feature_value: neighbour_weight(
                    graph, edge_set_name, sender_node_tag,
                    feature_value=feature_value))
            receiver_piece = graph.node_sets[
                edge_set.adjacency.node_set_name(receiver_tag)]

        # Set up the broadcast/pool ops for the receiver, plus any ops requested
        # by the subclass. The tag and name arguments conveniently encode the
        # distinction between operating over edge/node, node/context or
        # edge/context.
        def bind_receiver_args(fn):
            return lambda feature_value, **kwargs: fn(
                graph, receiver_tag, **name_kwarg,
                feature_value=feature_value, **kwargs)

        broadcast_from_receiver = bind_receiver_args(broadcast_ops.broadcast_v2)
        pool_to_receiver = bind_receiver_args(pool_ops.pool_v2)
        if self._extra_receiver_ops is None:
            extra_receiver_ops_kwarg = {}  # Pass no argument for this.
        else:
            extra_receiver_ops_kwarg = dict(
                extra_receiver_ops={name: bind_receiver_args(fn)
                                    for name, fn in self._extra_receiver_ops.items()})

        # Set up the inputs.
        receiver_input = sender_node_input = sender_edge_input = None
        if self._receiver_feature is not None:
            receiver_input = receiver_piece[self._receiver_feature]
        if None not in [sender_node_set, self._sender_node_feature]:
            sender_node_input = sender_node_set[self._sender_node_feature]
        if None not in [edge_set, self._sender_edge_feature]:
            sender_edge_input = edge_set[self._sender_edge_feature]

        return self.convolve(
            sender_node_input=sender_node_input,
            sender_edge_input=sender_edge_input,
            receiver_input=receiver_input,
            broadcast_from_sender_node=broadcast_from_sender_node,
            broadcast_from_receiver=broadcast_from_receiver,
            pool_to_receiver=pool_to_receiver,
            neighbour_weight_from_edge=neighbour_weight_from_edge,
            **extra_receiver_ops_kwarg,
            training=training)

    def convolve(self, *, sender_node_input: Optional[tf.Tensor],
                 sender_edge_input: Optional[tf.Tensor],
                 receiver_input: Optional[tf.Tensor],
                 broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
                 broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
                 pool_to_receiver: Callable[..., tf.Tensor],
                 neighbour_weight_from_edge: Callable[..., tf.Tensor],
                 extra_receiver_ops: Any = None,
                 training: bool) -> tf.Tensor:
        """Overridden internal method of the base class."""
        assert extra_receiver_ops is None, "Internal error: bad super().__init__()"
        assert sender_node_input is not None, "sender_node_input can't be None."
        result = broadcast_from_sender_node(sender_node_input)
        result = self._dropout(result, training=training)
        # The "Pooling aggregator" from Eq. (3) of the paper, plus dropout.
        result = self._pooling_transform_fn(result)
        weight = neighbour_weight_from_edge(sender_node_input)
        # 应用边权重
        result = result * weight
        result = pool_to_receiver(result, reduce_type=self._reduce_type)
        result = self._transform_neighbor_fn(result)
        return result


# Encodes the following imaginary papers:
#   [0] K. Kernel, L. Limit: "Anisotropic approximation", 2018.
#   [1] K. Kernel, L. Limit, M. Minor: "Better bipartite bijection bounds", 2019.
#   [2] M. Minor, N. Normal: "Convolutional convergence criteria", 2020.
# where paper [1] cites [0] and paper [2] cites [0] and [1].
#
graph = tfgnn.GraphTensor.from_pieces(
    node_sets={
        "illust": tfgnn.NodeSet.from_fields(
            sizes=tf.constant([8]),
            features={
                "illust_id": tf.constant([111, 222, 333, 444, 55, 66, 77, 88]),
                "embedding": tf.random.uniform(shape=[8, 1024]),
            }),
        "user": tfgnn.NodeSet.from_fields(
            sizes=tf.constant([4]),
            features={

                "user_id": tf.constant([123, 124, 125, 126]),
            }),

    },
    edge_sets={
        # "like": tfgnn.EdgeSet.from_fields(
        #     sizes=tf.constant([3]),
        #     adjacency=tfgnn.Adjacency.from_indices(
        #         source=("user", tf.constant([1, 2, 2])),
        #         target=("illust", tf.constant([0, 0, 1]))),
        #     features={'w': [0, 0, 0]},
        #
        # ),
        "like": tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([6]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=("user", tf.constant([0, 1, 2, 2, 2, 2])),
                target=("illust", tf.constant([0, 0, 0, 1, 2, 3, ]))),
            features={'w': tf.constant([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])},

        )

    })

input_graph = tf.keras.layers.Input(type_spec=graph.spec)


def set_initial_node_state(node_set, node_set_name):
    if node_set_name == "illust":
        id_embedding = tf.keras.layers.Embedding(10240000, 128)
        return tf.keras.layers.Concatenate()(
            [node_set["embedding"], id_embedding(node_set["illust_id"])])
    if node_set_name == "user":
        id_embedding = tf.keras.layers.Embedding(10240000, 128)
        return id_embedding(node_set["user_id"])


graph = tfgnn.keras.layers.MapFeatures(
    node_sets_fn=set_initial_node_state)(input_graph)


def gnn(graph):
    # k-1层邻居节点的表征经过一层DNN，然后聚合（可以考虑边的权重），是聚合函数符号，聚合函数可以是max/mean-pooling、加权求和、求平均；
    #
    # graph = tfgnn.keras.layers.GraphUpdate(
    #     #节点集以及其更新方法
    # node_sets={
    #     "illust":
    #         graph_sage.GCNGraphSAGENodeSetUpdate(
    #             reduce_type="mean",
    #             #层邻居节点的表征经过一层DNN
    #             use_pooling=True,
    #             edge_set_names=["like", "dislike"],
    #             l2_normalize=True,
    #             receiver_tag=tfgnn.TARGET,
    #             units=32)
    # })(graph)
    # graph = tfgnn.keras.layers.GraphUpdate(
    # node_sets={
    #     "illust":
    #         graph_sage.GCNGraphSAGENodeSetUpdate(
    #             reduce_type="mean",
    #             use_pooling=True,
    #             edge_set_names=["like", "dislike"],
    #             receiver_tag=tfgnn.TARGET,
    #             units=32)
    # })(graph)
    # GraphSAGEPoolingConv是卷积层 结果交给更新层 权重的聚合需要修改卷积层的pool_to_receiver之前的结果
    # 拼接 第k-1层目标节点的embedding，然后再经过另一层DNN，形成目标节点新的embedding； 定义GraphSAGENextState的参数combine_type即可
    # https://github.com/tensorflow/gnn/blob/09dcf290044833d6c9403366273868e452991338/tensorflow_gnn/docs/api_docs/python/tfgnn/keras/layers/AnyToAnyConvolutionBase.md
    graph = tfgnn.keras.layers.GraphUpdate(node_sets={
        "illust": tfgnn.keras.layers.NodeSetUpdate(
            {
                "like": PinSAGEPoolingConv(
                    units=32, hidden_units=16, receiver_tag=tfgnn.TARGET,
                )},
            graph_sage.GraphSAGENextState(units=32, combine_type="concat")),

    })(graph)

    graph = tfgnn.keras.layers.GraphUpdate(node_sets={
        "illust": tfgnn.keras.layers.NodeSetUpdate(
            {
                "like": PinSAGEPoolingConv(
                    units=32, hidden_units=16, receiver_tag=tfgnn.TARGET,
                )},
            graph_sage.GraphSAGENextState(units=32, combine_type="concat")),

    })(graph)
    return graph


graph = gnn(graph)

# pooled_features = tfgnn.keras.layers.Pool(
#      tfgnn.CONTEXT, "mean", node_set_name="illust")(graph)

model = tf.keras.Model(input_graph, graph)
tfgnn.keras.layers.Pool()
test = tfgnn.GraphTensor.from_pieces(
    node_sets={

        "illust": tfgnn.NodeSet.from_fields(
            sizes=tf.constant([8]),
            features={
                "illust_id": tf.constant([111, 222, 333, 444, 55, 66, 77, 88]),
                "embedding": tf.ones(shape=[8, 1024]),
            }),
        "user": tfgnn.NodeSet.from_fields(
            sizes=tf.constant([4]),
            features={

                "user_id": tf.constant([123, 124, 125, 126]),
            }),

    },
    edge_sets={
        "like": tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([6]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=("user", tf.constant([0, 1, 2, 2, 2, 2])),
                target=("illust", tf.constant([0, 0, 0, 1, 2, 3, ]))),
            features={'w': tf.constant([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])},

        )

    })

a = model(test)
print(a.node_sets["illust"]['hidden_state'])
