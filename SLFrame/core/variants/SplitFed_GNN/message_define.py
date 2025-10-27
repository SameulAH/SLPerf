class MyMessage:
    """
    Message type and payload key definitions for the SplitFed GNN variant.
    """

    # -------------------- Message Types -------------------- #
    # Client -> Server
    MSG_TYPE_C2S_ACTIVATION = 1
    MSG_TYPE_C2S_SEND_MODEL = 2
    MSG_TYPE_C2S_FINISHED = 3

    # Server -> Client
    MSG_TYPE_S2C_GRADIENT = 101
    MSG_TYPE_S2C_BROADCAST_MODEL = 102

    # -------------------- Common Keys -------------------- #
    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    # Activation exchange
    MSG_ARG_KEY_ACTIVATION = "activation"
    MSG_ARG_KEY_LABELS = "labels"
    MSG_ARG_KEY_NODE_IDS = "node_ids"
    MSG_ARG_KEY_IS_FULL_GRAPH = "is_full_graph"
    MSG_ARG_KEY_PHASE = "phase"
    MSG_ARG_KEY_ACT_SHAPE = "activation_shape"

    # Gradient / metrics payload
    MSG_ARG_KEY_GRAD = "grad"
    MSG_ARG_KEY_LOSS = "loss"
    MSG_ARG_KEY_ACCURACY = "accuracy"

    # Model exchange
    MSG_ARG_KEY_MODEL = "model"
    MSG_ARG_KEY_SAMPLE_NUM = "sample_num"
    MSG_ARG_KEY_ROUND = "round"
    MSG_ARG_KEY_COMM_STATS = "comm_stats"
