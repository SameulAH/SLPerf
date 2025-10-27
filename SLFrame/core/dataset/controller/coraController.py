import os
import shutil
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from core.log.Log import Log
from ..partition.partitionFactory import partitionFactory


class _NodeIndexDataset(Dataset):
    """DataLoader helper that yields node indices for batching."""

    def __init__(self, indices: torch.Tensor):
        if indices.dim() != 1:
            indices = indices.view(-1)
        self.indices = indices.long()

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.indices[idx]


class coraController:
    """
    Controller responsible for loading the Cora Planetoid dataset, performing graph
    partitioning, and exposing DataLoaders for SplitFed GNN clients.
    """

    def __init__(self, parse, transform=None):
        self.parse = parse
        self.transform = transform
        self.log = Log(self.__class__.__name__, parse)

        self.dataset: Optional[Planetoid] = None
        self.data: Optional[Data] = None
        self.partition_map: Dict[int, List[int]] = {}

        self._train_idx: Optional[torch.Tensor] = None
        self._val_idx: Optional[torch.Tensor] = None
        self._test_idx: Optional[torch.Tensor] = None

        self.batch_size = parse["batch_size"] or 128
        self.num_clients = parse["num_clients"] or parse["client_number"]
        self.data_root = parse["dataDir"] or "./data/"

    # ------------------------------------------------------------------ #
    # Data accessors
    def in_dim(self) -> int:
        self.ensure_data_loaded()
        return int(self.data.num_node_features)  # type: ignore[union-attr]

    def out_dim(self) -> int:
        self.ensure_data_loaded()
        return int(self.dataset.num_classes)  # type: ignore[union-attr]

    def num_nodes(self) -> int:
        self.ensure_data_loaded()
        return int(self.data.num_nodes)  # type: ignore[union-attr]

    def get_train_indices(self) -> torch.Tensor:
        self.ensure_data_loaded()
        return self._train_idx.clone()  # type: ignore[union-attr]

    def get_val_indices(self) -> torch.Tensor:
        self.ensure_data_loaded()
        return self._val_idx.clone()  # type: ignore[union-attr]

    def get_test_indices(self) -> torch.Tensor:
        self.ensure_data_loaded()
        return self._test_idx.clone()  # type: ignore[union-attr]

    def get_full_data(self):
        self.ensure_data_loaded()
        return self.data

    # ------------------------------------------------------------------ #
    def ensure_data_loaded(self):
        if self.data is not None:
            return
        root = os.path.join(self.data_root, "Planetoid")
        os.makedirs(root, exist_ok=True)

        dataset_dir = os.path.join(root, "Cora")
        processed_dir = os.path.join(dataset_dir, "processed")

        for attempt in range(5):
            try:
                self.dataset = Planetoid(root=root, name="Cora", transform=NormalizeFeatures())
                break
            except (EOFError, RuntimeError) as exc:
                if os.path.isdir(processed_dir):
                    shutil.rmtree(processed_dir, ignore_errors=True)
                wait_time = 1 + attempt
                self.log.warning(f"Planetoid load failed ({exc}); retrying in {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
        else:
            raise RuntimeError("Failed to load Planetoid Cora dataset after multiple attempts")

        self.data = self.dataset[0]
        self.data.y = self.data.y.long()

        self._train_idx = torch.nonzero(self.data.train_mask, as_tuple=False).view(-1)
        self._val_idx = torch.nonzero(self.data.val_mask, as_tuple=False).view(-1)
        self._test_idx = torch.nonzero(self.data.test_mask, as_tuple=False).view(-1)
        self.log.info(
            f"Loaded Cora dataset with {self.data.num_nodes} nodes, "
            f"{self.data.num_node_features} features, {self.dataset.num_classes} classes"
        )

    def loadData(self):
        self.ensure_data_loaded()
        return self.data

    def partition_data(self):
        partition = partitionFactory(parse=self.parse).factory()
        return partition(self.loadData)

    def _build_dataloader(self, indices: torch.Tensor, shuffle: bool) -> DataLoader:
        dataset = _NodeIndexDataset(indices)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def load_partition_data(self, process_id: int):
        self.ensure_data_loaded()
        (
            _,
            y_train,
            _,
            _y_test,
            net_dataidx_map,
            traindata_cls_counts,
        ) = self.partition_data()

        self.partition_map = net_dataidx_map
        class_num = len(torch.unique(y_train))
        train_data_num = int(self._train_idx.numel())

        # Shared graph info for every process
        shared_payload = {
            "graph_data": self.data,
            "edge_index": self.data.edge_index,
            "node_features": self.data.x,
            "node_labels": self.data.y,
            "train_idx": self.get_train_indices(),
            "val_idx": self.get_val_indices(),
            "test_idx": self.get_test_indices(),
            "in_dim": self.in_dim(),
            "out_dim": self.out_dim(),
            "num_nodes": self.num_nodes(),
            "partition_map": net_dataidx_map,
        }
        for key, value in shared_payload.items():
            self.parse[key] = value
        self.parse["in_dim"] = shared_payload["in_dim"]
        self.parse["out_dim"] = shared_payload["out_dim"]

        if process_id == 0:
            self.parse["train_loader_global"] = self._build_dataloader(self._train_idx, shuffle=True)
            self.parse["val_loader_global"] = self._build_dataloader(self._val_idx, shuffle=False)
            self.parse["test_loader_global"] = self._build_dataloader(self._test_idx, shuffle=False)
            local_data_num = 0
            train_data_local = None
            test_data_local = None
        else:
            client_id = process_id - 1
            client_indices = torch.tensor(net_dataidx_map[client_id], dtype=torch.long)
            if client_indices.numel() == 0:
                client_indices = self._train_idx[:0]
            train_data_local = self._build_dataloader(client_indices, shuffle=True)
            test_data_local = self._build_dataloader(self._test_idx, shuffle=False)
            local_data_num = int(client_indices.numel())

        self.parse["trainloader"] = train_data_local
        self.parse["testloader"] = test_data_local
        self.parse["train_data_num"] = train_data_num
        self.parse["train_data_global"] = None
        self.parse["test_data_global"] = None
        self.parse["local_data_num"] = local_data_num
        self.parse["class_num"] = class_num

        # Compatibility with existing logging
        self.parse["traindata_cls_counts"] = traindata_cls_counts

        return (
            train_data_num,
            None,
            None,
            local_data_num,
            train_data_local,
            test_data_local,
            class_num,
        )
