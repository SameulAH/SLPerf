from collections import defaultdict
import copy
from typing import Dict, Tuple

import torch

from core.communication.message import Message
from core.communication.msg_manager import MessageManager
from core.log.Log import Log

from .message_define import MyMessage


class ServerManager(MessageManager):
    """
    Server-side message manager for the SplitFed GNN variant. Handles activation
    processing, FedAvg aggregation, and broadcast of global models.
    """

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"], args["max_rank"] + 1, backend)
        self.trainer = trainer
        self.args = args
        self.config = args.config_dict if hasattr(args, "config_dict") else None
        self.log = Log(self.__class__.__name__, args)

        self.client_count = int(self._get_config_value("max_rank", 0))
        self.total_rounds = int(
            self._get_config_value("epochs", self._get_config_value("rounds", 1))
        )

        self.received_models: Dict[int, Tuple[int, Dict[str, torch.Tensor]]] = {}
        self.comm_stats: Dict[int, Dict[str, float]] = defaultdict(lambda: {"up": 0.0, "down": 0.0})
        self.round_index = 0
        self.finished_clients = set()

        self.train_metrics = {"loss": 0.0, "correct": 0, "total": 0}
        self.val_metrics = {"loss": 0.0, "correct": 0, "total": 0}

        self.eval_client_model = copy.deepcopy(self._get_config_value("client_model"))
        if self.eval_client_model is not None:
            self.eval_client_model = self.eval_client_model.to(self.trainer.device)
        self.features = self._get_config_value("node_features")
        self.edge_index = self._get_config_value("edge_index")
        self.val_idx = self._get_config_value("val_idx", self._get_config_value("test_idx"))
        if self.features is not None:
            self.features = self.features.to(self.trainer.device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(self.trainer.device)
        self.comm_log_level = int(self._get_config_value("comm_log_level", 1) or 1)

    # ------------------------------------------------------------------ #
    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_ACTIVATION, self.handle_receive_activation
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL, self.handle_receive_model
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_FINISHED, self.handle_receive_finish
        )

    def _log_comm(self, message: str, *args):
        if self.comm_log_level <= 0:
            return
        formatted = message % args if args else message
        if self.comm_log_level > 1:
            self.log.debug(formatted)
        else:
            self.log.info(formatted)

    # ------------------------------------------------------------------ #
    def handle_receive_activation(self, msg_params: Message):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        activations = msg_params.get(MyMessage.MSG_ARG_KEY_ACTIVATION)
        labels = msg_params.get(MyMessage.MSG_ARG_KEY_LABELS)
        node_ids = msg_params.get(MyMessage.MSG_ARG_KEY_NODE_IDS)
        is_full = msg_params.get(MyMessage.MSG_ARG_KEY_IS_FULL_GRAPH)
        phase = self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_PHASE, "train")
        act_shape = self._get_msg_param(
            msg_params, MyMessage.MSG_ARG_KEY_ACT_SHAPE, tuple(activations.shape)
        )

        self._log_comm(
            "activation_receive | sender=%s | phase=%s | full_graph=%s | nodes=%d | shape=%s",
            sender,
            phase,
            is_full,
            int(node_ids.numel()) if node_ids is not None else -1,
            act_shape,
        )

        grad, loss, correct, total = self.trainer.process_batch(
            activations, labels, node_ids, is_full, phase
        )

        metrics_store = self.train_metrics if phase == "train" else self.val_metrics
        metrics_store["loss"] += loss * total
        metrics_store["correct"] += correct
        metrics_store["total"] += total

        response = Message(MyMessage.MSG_TYPE_S2C_GRADIENT, self.args["rank"], sender)
        response.add_params(MyMessage.MSG_ARG_KEY_GRAD, grad)
        response.add_params(MyMessage.MSG_ARG_KEY_IS_FULL_GRAPH, is_full)
        response.add_params(MyMessage.MSG_ARG_KEY_PHASE, phase)
        response.add_params(MyMessage.MSG_ARG_KEY_LOSS, loss)
        response.add_params(
            MyMessage.MSG_ARG_KEY_ACCURACY, correct / total if total else 0.0
        )
        response.add_params("batch_size", total)
        self.send_message(response)

    def handle_receive_model(self, msg_params: Message):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        raw_state = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        state_dict = {k: v.cpu() for k, v in raw_state.items()}
        sample_num = int(self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_SAMPLE_NUM, 0) or 0)
        comm_stats = self._get_msg_param(
            msg_params, MyMessage.MSG_ARG_KEY_COMM_STATS, {"up": 0.0, "down": 0.0}
        )
        train_loss = self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_LOSS)
        train_acc = self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_ACCURACY)
        round_idx = self._get_msg_param(msg_params, MyMessage.MSG_ARG_KEY_ROUND, self.round_index)

        self.comm_stats[sender] = {
            "up": float(comm_stats.get("up", 0.0) or 0.0),
            "down": float(comm_stats.get("down", 0.0) or 0.0),
        }
        self.received_models[sender] = (sample_num, state_dict)

        self.log.info(
            f"ModelUpload[client={sender}][epoch={round_idx}] samples={sample_num} "
            f"train_loss={train_loss} train_acc={train_acc}"
        )

        if len(self.received_models) == self.client_count:
            aggregated_state = self._aggregate_models()
            train_summary = self.trainer.get_round_metrics()
            val_summary = None
            if self.val_metrics["total"] > 0:
                val_summary = self._summarize_metrics(self.val_metrics)

            total_up = sum(stats["up"] for stats in self.comm_stats.values())
            total_down = sum(stats["down"] for stats in self.comm_stats.values())

            self.log.info(
                "EpochSummary[epoch={epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                "global_postagg_val_loss={val_loss:.4f} global_postagg_val_acc[{epoch}]={val_acc:.4f} "
                "comm_up_bytes={up:.0f} comm_down_bytes={down:.0f}".format(
                    epoch=self.round_index,
                    train_loss=train_summary["loss"],
                    train_acc=train_summary["accuracy"],
                    val_loss=val_summary["loss"] if val_summary else 0.0,
                    val_acc=val_summary["accuracy"] if val_summary else 0.0,
                    up=total_up,
                    down=total_down,
                )
            )

            self._broadcast_model(aggregated_state, train_summary)

            self.trainer.reset_round_metrics()
            self.received_models.clear()
            self.comm_stats.clear()
            self.train_metrics = {"loss": 0.0, "correct": 0, "total": 0}
            self.val_metrics = {"loss": 0.0, "correct": 0, "total": 0}
            self.round_index += 1

            if self.round_index >= self.total_rounds:
                self.log.info(f"Server reached configured epochs={self.total_rounds}")

    def handle_receive_finish(self, msg_params: Message):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        self.finished_clients.add(sender)
        if len(self.finished_clients) == self.client_count:
            self.log.info("All clients finished; shutting down server")
            self.finish()

    # ------------------------------------------------------------------ #
    def _aggregate_models(self):
        sample_sum = sum(sample for sample, _ in self.received_models.values())
        use_uniform = False
        if sample_sum == 0:
            sample_sum = len(self.received_models)
            use_uniform = True

        first_state = next(iter(self.received_models.values()))[1]
        aggregated = {key: torch.zeros_like(param) for key, param in first_state.items()}

        for sample_num, state in self.received_models.values():
            if use_uniform or sample_sum == 0:
                weight = 1.0 / max(len(self.received_models), 1)
            else:
                weight = sample_num / sample_sum
            for key in aggregated:
                aggregated[key] += state[key] * weight

        return aggregated

    def _broadcast_model(self, state_dict, metrics):
        for client_id in range(1, self.client_count + 1):
            message = Message(MyMessage.MSG_TYPE_S2C_BROADCAST_MODEL, self.args["rank"], client_id)
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL, state_dict)
            message.add_params(MyMessage.MSG_ARG_KEY_ROUND, self.round_index)
            message.add_params(MyMessage.MSG_ARG_KEY_LOSS, metrics["loss"])
            message.add_params(MyMessage.MSG_ARG_KEY_ACCURACY, metrics["accuracy"])
            self.send_message(message)

    def _summarize_metrics(self, metrics_store):
        total = metrics_store.get("total", 0)
        if total == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        loss = metrics_store.get("loss", 0.0) / total
        accuracy = metrics_store.get("correct", 0) / total
        return {"loss": loss, "accuracy": accuracy}

    def _evaluate_global_model(self, client_state):
        if self.eval_client_model is None or self.features is None or self.edge_index is None:
            return None
        self.eval_client_model.load_state_dict(client_state)
        self.eval_client_model.eval()
        self.trainer.model.eval()
        with torch.no_grad():
            hidden = self.eval_client_model(self.features, self.edge_index)
            logits = self.trainer.model(hidden, self.edge_index)
            if self.val_idx is not None:
                idx = self.val_idx.to(self.trainer.device)
                logits = logits[idx]
                labels = self.trainer.labels[idx]
            else:
                labels = self.trainer.labels
            loss = self.trainer.criterion(logits, labels).item()
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()
        return {"loss": loss, "accuracy": accuracy}

    # ------------------------------------------------------------------ #
    def _get_msg_param(self, message, key, default=None):
        try:
            return message.get(key)
        except KeyError:
            return default

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
