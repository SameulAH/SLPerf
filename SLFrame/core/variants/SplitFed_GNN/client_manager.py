import queue
import threading
from typing import Dict

from core.communication.message import Message
from core.communication.msg_manager import MessageManager
from core.log.Log import Log

from .message_define import MyMessage


class ClientManager(MessageManager):
    """
    Message manager for SplitFed-GNN clients. Handles the main training loop,
    activation exchange, and model uploads.
    """

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "client", args["comm"], args["rank"], args["max_rank"] + 1, backend)
        self.trainer = trainer
        self.args = args
        self.config = args.config_dict if hasattr(args, "config_dict") else None
        self.log = Log(self.__class__.__name__, args)

        self.rounds = int(self._get_config_value("epochs", self._get_config_value("rounds", 1)))
        self.local_epochs = int(self._get_config_value("local_epochs", 1))
        self.server_rank = int(self._get_config_value("server_rank", 0))
        self.comm_log_level = int(self._get_config_value("comm_log_level", 1) or 1)

        self.grad_queue: "queue.Queue" = queue.Queue()
        self.model_queue: "queue.Queue" = queue.Queue()
        self.comm_stats: Dict[str, float] = {"up": 0.0, "down": 0.0}
        self.training_stats = []

    # ------------------------------------------------------------------ #
    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_GRADIENT, self.handle_receive_gradients
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_BROADCAST_MODEL, self.handle_receive_model
        )

    def run(self):
        training_thread = threading.Thread(target=self._train_loop, daemon=True)
        training_thread.start()
        super().run()

    # ------------------------------------------------------------------ #
    def _train_loop(self):
        self.trainer.train_mode()
        for round_idx in range(self.rounds):
            self.comm_stats = {"up": 0.0, "down": 0.0}
            self.training_stats = []
            self.log.info(
                f"Client {self._get_config_value('rank', 0)} starting epoch {round_idx}"
            )

            for _ in range(self.local_epochs):
                for _ in range(len(self.trainer.trainloader)):
                    acts, labels, node_ids, meta = self.trainer.forward_pass(phase="train")
                    self.send_activations_to_server(acts, labels, node_ids, meta)
                    grad_payload = self.grad_queue.get()
                    if grad_payload.get("phase") != "train":
                        continue
                    grads = grad_payload["grad"]
                    is_full = grad_payload["is_full_graph"]
                    self.trainer.backward_pass(grads, is_full)

            train_summary = self._summarize_stats(self.training_stats)
            self.log.info(
                "TrainingSummary[client={client}][epoch={epoch}] loss={loss:.4f} acc={acc:.4f} "
                "comm_up_bytes={up:.0f} comm_down_bytes={down:.0f}".format(
                    client=self._get_config_value("rank", 0),
                    epoch=round_idx,
                    loss=train_summary["loss"],
                    acc=train_summary["accuracy"],
                    up=self.comm_stats["up"],
                    down=self.comm_stats["down"],
                )
            )

            self.run_validation(round_idx, tag="client_preagg_val_acc")

            self.send_model_to_server(round_idx, train_summary)
            global_state = self.model_queue.get()
            self.trainer.load_model_state(global_state)

            self.run_validation(round_idx, tag="client_postagg_val_acc")

        self.send_finish_to_server()
        self.log.info(
            f"Client {self._get_config_value('rank', 0)} completed {self.rounds} epochs"
        )

    # ------------------------------------------------------------------ #
    def handle_receive_gradients(self, msg_params: Message):
        grads = self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_GRAD)
        is_full = self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_IS_FULL_GRAPH)
        phase = self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_PHASE, "train")
        loss = self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_LOSS)
        acc = self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_ACCURACY)
        batch_size = self._get_msg_param(msg_params, "batch_size")
        if grads is not None:
            self.comm_stats["down"] += grads.numel() * grads.element_size()
        payload = {
            "grad": grads,
            "is_full_graph": is_full,
            "phase": phase,
            "loss": loss,
            "accuracy": acc,
            "batch_size": batch_size,
        }
        self.grad_queue.put(payload)
        if (
            phase == "train"
            and loss is not None
            and acc is not None
            and batch_size is not None
        ):
            self.training_stats.append((float(loss), float(acc), int(batch_size)))

    def handle_receive_model(self, msg_params: Message):
        model_state = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        round_idx = self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_ROUND, 0)
        loss = float(self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_LOSS, 0.0) or 0.0)
        acc = float(self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_ACCURACY, 0.0) or 0.0)
        self.log.info(
            f"Client {self._get_config_value('rank', 0)} received global model for epoch {round_idx} "
            f"(train_loss={loss:.4f}, train_acc={acc:.4f})"
        )
        self.model_queue.put(model_state)

    # ------------------------------------------------------------------ #
    def send_activations_to_server(self, acts, labels, node_ids, meta):
        message = Message(MyMessage.MSG_TYPE_C2S_ACTIVATION, self.args["rank"], self.server_rank)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTIVATION, acts)
        message.add_params(MyMessage.MSG_ARG_KEY_LABELS, labels)
        message.add_params(MyMessage.MSG_ARG_KEY_NODE_IDS, node_ids)
        message.add_params(MyMessage.MSG_ARG_KEY_IS_FULL_GRAPH, meta.get("is_full_graph", False))
        message.add_params(MyMessage.MSG_ARG_KEY_PHASE, meta.get("phase", "train"))
        act_shape = tuple(acts.shape)
        label_shape = tuple(labels.shape)
        message.add_params(MyMessage.MSG_ARG_KEY_ACT_SHAPE, act_shape)
        self.comm_stats["up"] += acts.numel() * acts.element_size()
        self._log_comm(
            "activation_send | from=%s | to=%s | phase=%s | full_graph=%s | acts=%s | labels=%s",
            self.args["rank"],
            self.server_rank,
            meta.get("phase", "train"),
            meta.get("is_full_graph", False),
            act_shape,
            label_shape,
        )
        self.send_message(message)

    def send_model_to_server(self, round_idx: int, train_summary: Dict[str, float]):
        state_dict = {k: v.cpu() for k, v in self.trainer.get_model_state().items()}
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL, self.args["rank"], self.server_rank)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL, state_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_SAMPLE_NUM, self.trainer.get_local_sample_number())
        message.add_params(MyMessage.MSG_ARG_KEY_COMM_STATS, dict(self.comm_stats))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_LOSS, train_summary["loss"])
        message.add_params(MyMessage.MSG_ARG_KEY_ACCURACY, train_summary["accuracy"])
        self.send_message(message)

    def send_finish_to_server(self):
        message = Message(MyMessage.MSG_TYPE_C2S_FINISHED, self.args["rank"], self.server_rank)
        self.send_message(message)

    def run_validation(self, epoch_idx: int, tag: str):
        self.trainer.eval_mode()
        val_stats = []
        for _ in range(len(self.trainer.testloader)):
            acts, labels, node_ids, meta = self.trainer.forward_pass(phase="val")
            self.send_activations_to_server(acts, labels, node_ids, meta)
            payload = self.grad_queue.get()
            if payload.get("phase") == "val":
                loss = payload.get("loss")
                acc = payload.get("accuracy")
                batch_size = payload.get("batch_size")
                if loss is not None and acc is not None and batch_size is not None:
                    val_stats.append((float(loss), float(acc), int(batch_size)))
        summary = self._summarize_stats(val_stats)
        self.log.info(
            f"{tag}[client={self._get_config_value('rank', 0)}][epoch={epoch_idx}] "
            f"loss={summary['loss']:.4f} acc={summary['accuracy']:.4f}"
        )
        self.trainer.train_mode()
        return summary

    # ------------------------------------------------------------------ #
    def _summarize_stats(self, stats):
        total = sum(batch for _, _, batch in stats)
        if total == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        loss = sum(loss * batch for loss, _, batch in stats) / total
        acc = sum(acc * batch for _, acc, batch in stats) / total
        return {"loss": loss, "accuracy": acc}

    def _get_config_value(self, key, default=None):
        if self.config is not None and key in self.config and self.config[key] is not None:
            return self.config[key]
        try:
            value = self.args[key]
            if value is not None:
                return value
        except Exception:
            pass
        if hasattr(self.args, key):
            value = getattr(self.args, key)
            if value is not None:
                return value
        return default

    def _get_msg_param(self, message, key, default=None):
        try:
            return message.get(key)
        except KeyError:
            return default
        except Exception:
            return default

    def _log_comm(self, message: str, *args):
        if self.comm_log_level <= 0:
            return
        formatted = message % args if args else message
        if self.comm_log_level > 1:
            self.log.debug(formatted)
        else:
            self.log.info(formatted)
