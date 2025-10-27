from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from core.log.Log import Log


class SplitNNServer:
    """
    Server-side portion of the SplitFed GNN. Receives activations from clients,
    continues the forward pass, computes loss, and returns gradients.
    """

    def __init__(self, args):
        self.args = args
        self.config = args.config_dict if hasattr(args, "config_dict") else None
        self.log = Log(self.__class__.__name__, args)

        device_pref = self._get_config_value("device", "cpu")
        self.device = (
            device_pref
            if isinstance(device_pref, torch.device)
            else torch.device(device_pref or "cpu")
        )

        self.model = self._get_config_value("server_model")
        if self.model is None:
            raise ValueError("Server model must be provided for SplitFed GNN")
        self.model = self.model.to(self.device)

        optimizer_cfg: Dict = self._get_config_value("optimizer", {}) or {}
        base_lr = self._get_config_value("lr", 0.01)
        lr = optimizer_cfg.get("lr", base_lr if base_lr is not None else 0.01)
        weight_decay = optimizer_cfg.get("weight_decay", 0.0)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        self.edge_index = self._get_config_value("edge_index")
        self.labels = self._get_config_value("node_labels")
        if self.edge_index is None or self.labels is None:
            raise ValueError("Server requires edge_index and node_labels tensors")
        self.edge_index = self.edge_index.to(self.device)
        self.labels = self.labels.to(self.device)

        self.model_args = self._get_config_value("model_args", {}) or {}
        self.cut = str(self.model_args.get("cut", "mlp")).lower()
        self.comm_log_level = int(self._get_config_value("comm_log_level", 1) or 1)

        self.round_index = 0
        self.reset_round_metrics()

    # ------------------------------------------------------------------ #
    def reset_round_metrics(self):
        self.round_loss = 0.0
        self.round_correct = 0
        self.round_total = 0

    def _log_comm(self, message: str, *args):
        if self.comm_log_level <= 0:
            return
        formatted = message % args if args else message
        if self.comm_log_level > 1:
            self.log.debug(formatted)
        else:
            self.log.info(formatted)

    def process_batch(self, activation, labels, node_ids, is_full_graph: bool, phase: str):
        training = phase == "train"
        if training:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        orig_shape = tuple(activation.shape)
        orig_device = getattr(activation, "device", None)

        acts = activation.to(self.device)
        if training:
            acts.requires_grad_(True)
            acts.retain_grad()

        labels = labels.to(self.device)
        node_ids = node_ids.to(self.device)

        logits = self.model(acts, self.edge_index)
        if is_full_graph:
            logits = logits[node_ids]

        loss = self.criterion(logits, labels)

        self._log_comm(
            "activation_process | phase=%s | full_graph=%s | nodes=%d "
            "| recv_shape=%s | recv_device=%s | proc_shape=%s | proc_device=%s",
            phase,
            is_full_graph,
            int(node_ids.numel()) if node_ids is not None else -1,
            orig_shape,
            orig_device,
            tuple(acts.shape),
            acts.device,
        )

        grad = None
        if training:
            loss.backward()
            self.optimizer.step()
            grad = acts.grad.detach().cpu()

        _, predictions = torch.max(logits, 1)
        correct = predictions.eq(labels).sum().item()
        batch_total = labels.size(0)

        if training:
            self.round_loss += loss.item() * batch_total
            self.round_correct += correct
            self.round_total += batch_total

        return grad, loss.item(), correct, batch_total

    def get_round_metrics(self):
        if self.round_total == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        avg_loss = self.round_loss / self.round_total
        accuracy = self.round_correct / self.round_total
        return {"loss": avg_loss, "accuracy": accuracy}

    def load_model_state(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_model_state(self):
        return self.model.state_dict()

    # ------------------------------------------------------------------ #
    def _get_config_value(self, key, default=None):
        if self.config is not None and key in self.config and self.config[key] is not None:
            return self.config[key]

        try:
            value = self.args[key]  # type: ignore[index]
            if value is not None:
                return value
        except Exception:
            pass

        if hasattr(self.args, key):
            value = getattr(self.args, key)
            if value is not None:
                return value

        return default
