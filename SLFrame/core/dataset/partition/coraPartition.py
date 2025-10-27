from typing import Callable, Dict, List

import networkx as nx
from networkx.algorithms.community import louvain_communities
import numpy as np
import torch

from core.log.Log import Log
from .basePartition import abstractPartition


def _record_class_counts(data, labels: torch.Tensor, mapping: Dict[int, List[int]]):
    """
    Record label distribution and structural statistics for each client partition.
    """
    graph = nx.Graph()
    graph.add_edges_from(data.edge_index.cpu().numpy().T)

    stats = {}
    for cid, indices in mapping.items():
        client_stats: Dict[str, Dict[int, int] | int | float] = {}

        if len(indices) == 0:
            client_stats["labels"] = {}
        else:
            label_subset = labels[indices]
            unique, counts = torch.unique(label_subset, return_counts=True)
            client_stats["labels"] = {
                int(k.item()): int(v.item()) for k, v in zip(unique, counts)
            }

        sub_nodes = set(indices)
        sub_graph = graph.subgraph(indices)
        boundary_edges = sum(1 for u, v in graph.edges(indices) if v not in sub_nodes)
        avg_degree = (
            float(np.mean([d for _, d in sub_graph.degree()])) if len(sub_graph) > 0 else 0.0
        )

        client_stats["num_nodes"] = len(indices)
        client_stats["avg_degree"] = avg_degree
        client_stats["boundary_edges"] = boundary_edges
        stats[cid] = client_stats

    return stats


class coraPartition(abstractPartition):
    """
    Partition trainer indices of the Cora Planetoid dataset across clients according
    to the selected strategy.
    """

    def __init__(self, parse):
        self.parse = parse
        self.log = Log(self.__class__.__name__, parse)

    def partition_data(self):
        partition_type = (
            self.parse["partition_type"]
            or self.parse["partition_method"]
            or "iid"
        )
        partition_type = str(partition_type).lower()
        num_clients = self._resolve_client_number()

        self.log.info(
            f"Cora partition resolution | partition_type={partition_type} | "
            f"raw_type={self.parse['partition_type']} | raw_method={self.parse['partition_method']}"
        )

        if num_clients <= 0:
            raise ValueError("Number of clients must be positive for Cora partitioning")

        def _apply(load_data: Callable[[], torch.Tensor]):
            data = load_data()
            labels = data.y
            train_idx = torch.nonzero(data.train_mask, as_tuple=False).view(-1)

            if partition_type in {"iid", "homo"}:
                mapping = self._iid_partition(train_idx, num_clients)
            elif partition_type in {"dirichlet", "dir", "noniid", "hetero"}:
                alpha = float(
                    self.parse["dirichlet_alpha"]
                    or self.parse["partition_alpha"]
                    or 0.3
                )
                mapping = self._dirichlet_partition(train_idx, labels, num_clients, alpha)
            elif partition_type == "community":
                mapping = self._community_partition(data, train_idx, num_clients)
            elif partition_type == "degree":
                mapping = self._degree_partition(data, train_idx, num_clients)
            else:
                raise ValueError(
                    f"Unsupported partition_type '{partition_type}' for Cora dataset"
                )

            stats = _record_class_counts(data, labels, mapping)
            client_counts = {cid: info.get("labels", {}) for cid, info in stats.items()}
            structural = {
                cid: {k: v for k, v in info.items() if k != "labels"} for cid, info in stats.items()
            }

            self.log.info(
                f"Partitioned Cora with strategy={partition_type}, client_counts={client_counts}"
            )
            self.log.debug(f"Cora partition structural stats: {structural}")
            self.parse["cora_partition_stats"] = stats

            return data, labels, data, labels, mapping, client_counts

        return _apply

    # ------------------------------------------------------------------ #
    def _resolve_client_number(self) -> int:
        runtime_clients = self.parse["client_number"]
        if runtime_clients:
            return int(runtime_clients)
        config_clients = self.parse["num_clients"]
        if config_clients:
            return int(config_clients)
        return 0

    def _rng(self, seed_fallback: int = 42) -> np.random.Generator:
        seed = self.parse["seed"]
        if seed is None:
            seed = seed_fallback
        return np.random.default_rng(int(seed))

    def _iid_partition(self, train_idx: torch.Tensor, num_clients: int) -> Dict[int, List[int]]:
        rng = self._rng()
        indices = train_idx.cpu().numpy().copy()
        rng.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        return {i: split.astype(int).tolist() for i, split in enumerate(splits)}

    def _dirichlet_partition(
        self,
        train_idx: torch.Tensor,
        labels: torch.Tensor,
        num_clients: int,
        alpha: float,
    ) -> Dict[int, List[int]]:
        rng = self._rng()
        label_array = labels.cpu().numpy()
        train_indices = train_idx.cpu().numpy()
        client_indices = {i: [] for i in range(num_clients)}
        classes = np.unique(label_array[train_indices])

        attempts = 0
        while attempts < 10:
            temp_map = {i: [] for i in range(num_clients)}
            for cls in classes:
                cls_idx = train_indices[label_array[train_indices] == cls]
                rng.shuffle(cls_idx)
                proportions = rng.dirichlet([alpha] * num_clients)
                proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
                split = np.split(cls_idx, proportions)
                for client_id, subset in enumerate(split):
                    temp_map[client_id].extend(subset.astype(int).tolist())
            sizes = [len(temp_map[i]) for i in range(num_clients)]
            if min(sizes) > 0:
                client_indices = temp_map
                break
            attempts += 1

        if not all(len(client_indices[i]) > 0 for i in range(num_clients)):
            raise RuntimeError("Failed to create Dirichlet partition with non-empty clients")

        for client_id in client_indices:
            rng.shuffle(client_indices[client_id])
        return client_indices

    def _community_partition(
        self,
        data: torch.Tensor,
        train_idx: torch.Tensor,
        num_clients: int,
    ) -> Dict[int, List[int]]:
        graph = nx.Graph()
        graph.add_edges_from(data.edge_index.cpu().numpy().T)

        communities = louvain_communities(graph, seed=int(self.parse["seed"] or 42))
        train_set = set(int(i) for i in train_idx.cpu().numpy())
        clusters = [sorted(train_set.intersection(comm)) for comm in communities if comm]
        clusters = [cluster for cluster in clusters if cluster]
        if not clusters:
            return self._iid_partition(train_idx, num_clients)

        mapping = {i: [] for i in range(num_clients)}
        for idx, cluster in enumerate(clusters):
            target_client = idx % num_clients
            mapping[target_client].extend(cluster)

        for client_id in mapping:
            mapping[client_id].sort()
        return mapping

    def _degree_partition(
        self,
        data: torch.Tensor,
        train_idx: torch.Tensor,
        num_clients: int,
    ) -> Dict[int, List[int]]:
        edge_index = data.edge_index
        degrees = torch.bincount(edge_index[0], minlength=data.num_nodes)
        sorted_nodes = torch.argsort(degrees, descending=True)
        train_mask = torch.zeros_like(sorted_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        sorted_train_nodes = sorted_nodes[train_mask[sorted_nodes]].cpu().numpy()
        splits = np.array_split(sorted_train_nodes, num_clients)
        return {i: split.astype(int).tolist() for i, split in enumerate(splits)}
