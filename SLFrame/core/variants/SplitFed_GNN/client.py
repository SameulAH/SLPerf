import threading
from typing import Dict, Optional

import torch
import torch.optim as optim

from core.log.Log import Log


class SplitNNClient:
    """
    Client role for SplitFed GNN. Holds the front portion of the GCN, performs
    forward/backward on local layers, and communicates activations/gradients with
    the central server.
    """

    def __init__(self, args):
        self.args = args
        self.config = args.config_dict if hasattr(args, "config_dict") else None
        self.log = Log(self.__class__.__name__, args)

        self.rank = int(self._get_config_value("rank", 0))
        device_pref = self._get_config_value("device", "cpu")
        self.device = (
            device_pref
            if isinstance(device_pref, torch.device)
            else torch.device(device_pref or "cpu")
        )

        self.model = self._get_config_value("client_model")
        if self.model is None:
            raise ValueError("Client model must be provided for SplitFed GNN")
        self.model = self.model.to(self.device)

        self.trainloader = self._get_config_value("trainloader")
        if self.trainloader is None:
            raise ValueError("Cora SplitFed client requires a training dataloader")
        self.testloader = self._get_config_value("testloader")
        if self.testloader is None:
            raise ValueError("Cora SplitFed client requires a test dataloader")

        self.edge_index = self._get_config_value("edge_index")
        self.features = self._get_config_value("node_features")
        self.labels = self._get_config_value("node_labels")
        if any(x is None for x in (self.edge_index, self.features, self.labels)):
            raise ValueError(
                "Missing graph tensors (edge_index/node_features/node_labels) for client"
            )
        self.edge_index = self.edge_index.to(self.device)
        self.features = self.features.to(self.device)
        self.labels = self.labels.to(self.device)

        self.partition_map = self._get_config_value("partition_map", {})
        self.server_rank = int(self._get_config_value("server_rank", 0))

        if self.rank <= 0:
            raise ValueError("Client rank must be positive for SplitFed GNN")
        if self.partition_map is None or self.rank - 1 not in self.partition_map:
            raise ValueError(
                f"Partition map missing entries for client rank {self.rank}"
            )
        self.local_sample_number = len(self.partition_map[self.rank - 1])

        optimizer_cfg: Dict = self._get_config_value("optimizer", {}) or {}
        base_lr = self._get_config_value("lr", 0.01)
        lr = optimizer_cfg.get("lr", base_lr if base_lr is not None else 0.01)
        weight_decay = optimizer_cfg.get("weight_decay", 0.0)
        model_params = list(self.model.parameters())
        self.optimizer = (
            optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
            if model_params
            else None
        )

        model_args = self._get_config_value("model_args", {}) or {}
        self.cut = str(model_args.get("cut", "c1")).lower()

        self.current_acts: Optional[torch.Tensor] = None
        self.full_hidden: Optional[torch.Tensor] = None
        self.current_nodes: Optional[torch.Tensor] = None
        self.comm_log_level = int(self._get_config_value("comm_log_level", 1) or 1)

        self.iter_lock = threading.Lock()
        self.train_iter = iter(self.trainloader)
        self.eval_iter = iter(self.testloader)

    # ------------------------------------------------------------------ #
    def _log_comm(self, message: str, *args):
        if self.comm_log_level <= 0:
            return
        formatted = message % args if args else message
        if self.comm_log_level > 1:
            self.log.debug(formatted)
        else:
            self.log.info(formatted)

    def _next_batch(self, phase: str) -> torch.Tensor:
        if phase == "train":
            loader = self.trainloader
            iterator_attr = "train_iter"
        else:
            loader = self.testloader
            iterator_attr = "eval_iter"

        with self.iter_lock:
            iterator = getattr(self, iterator_attr)
            try:
                batch_nodes = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                setattr(self, iterator_attr, iterator)
                batch_nodes = next(iterator)

        if isinstance(batch_nodes, (list, tuple)):
            batch_nodes = batch_nodes[0]
        return batch_nodes.long()

    def forward_pass(self, phase: str = "train"):
        batch_nodes = self._next_batch(phase).to(self.device)
        if phase == "train" and self.optimizer is not None:
            self.optimizer.zero_grad()

        hidden = self.model(self.features, self.edge_index)
        labels = self.labels[batch_nodes]

        meta = {"phase": phase}
        cut_mode = self.cut
        act_tensor: Optional[torch.Tensor]

        if cut_mode in {"c0", "c1"}:
            self.full_hidden = hidden
            if phase == "train" and self.optimizer is not None:
                self.full_hidden.retain_grad()
            acts = hidden.detach().cpu()
            meta["is_full_graph"] = True
            act_tensor = self.full_hidden
        elif cut_mode == "c2":
            self.full_hidden = None
            self.current_acts = hidden[batch_nodes]
            if phase == "train":
                self.current_acts.retain_grad()
            acts = self.current_acts.detach().cpu()
            meta["is_full_graph"] = False
            act_tensor = self.current_acts
        else:
            self.full_hidden = None
            self.current_acts = hidden[batch_nodes]
            if phase == "train":
                self.current_acts.retain_grad()
            acts = self.current_acts.detach().cpu()
            meta["is_full_graph"] = False
            act_tensor = self.current_acts

        self.current_nodes = batch_nodes

        if act_tensor is not None:
            act_shape = tuple(act_tensor.shape)
            node_count = int(batch_nodes.numel())
            self._log_comm(
                "activation_summary | cut=%s | rank=%s | phase=%s | full_graph=%s "
                "| nodes=%d | shape=%s | device=%s",
                self.cut,
                self.rank,
                phase,
                meta["is_full_graph"],
                node_count,
                act_shape,
                act_tensor.device,
            )

        return acts, labels.detach().cpu(), batch_nodes.detach().cpu(), meta

    def backward_pass(self, grad_tensor: torch.Tensor, is_full_graph: bool):
        if grad_tensor is None or self.optimizer is None:
            return
        grad = grad_tensor.to(self.device)
        if is_full_graph:
            if self.full_hidden is None:
                raise RuntimeError("Missing full_hidden for backward pass")
            self.full_hidden.backward(grad)
        else:
            if self.current_acts is None:
                raise RuntimeError("Missing current_acts for backward pass")
            self.current_acts.backward(grad)
        self.optimizer.step()

    # ------------------------------------------------------------------ #
    def get_local_sample_number(self) -> int:
        return self.local_sample_number

    def get_model_state(self):
        return self.model.state_dict()

    def load_model_state(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train_mode(self):
        self.model.train()
        self.train_iter = iter(self.trainloader)

    def eval_mode(self):
        self.model.eval()
        self.eval_iter = iter(self.testloader)

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
