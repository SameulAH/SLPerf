from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

VALID_CUTS = {"c0", "c1", "c2"}


class GCNClient(nn.Module):
    """
    Front half of a two-layer GCN that can be cut at different depths.

    - ``c0``  : client forwards raw node features to the server (no local GCN).
    - ``c1``  : client runs the first GCN layer and shares the hidden embeddings.
    - ``c2``  : client runs both GCN layers and only a classifier lives on the server.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        *,
        dropout: float = 0.5,
        cut: str = "c1",
    ) -> None:
        super().__init__()
        cut = cut.lower()
        if cut not in VALID_CUTS:
            raise ValueError(f"Unsupported cut '{cut}'. Expected one of {sorted(VALID_CUTS)}.")

        self.cut = cut
        self.dropout = dropout

        self.conv1 = GCNConv(in_dim, hidden_dim) if cut in {"c1", "c2"} else None
        self.conv2 = GCNConv(hidden_dim, hidden_dim) if cut == "c2" else None

    def forward(self, x, edge_index):
        if self.cut == "c0":
            return x

        if self.conv1 is None:
            raise RuntimeError("GCNClient.conv1 not initialised for cut '%s'" % self.cut)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.cut == "c2":
            if self.conv2 is None:
                raise RuntimeError("GCNClient.conv2 not initialised for cut 'c2'")
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GCNTail(nn.Module):
    """
    Back half of the split GCN. The layers present depend on the chosen cut:

    - ``c0``  : two GCN layers + linear classifier (entire GCN on server).
    - ``c1``  : one GCN layer + linear classifier.
    - ``c2``  : only the linear classifier.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        dropout: float = 0.5,
        cut: str = "c1",
    ) -> None:
        super().__init__()
        cut = cut.lower()
        if cut not in VALID_CUTS:
            raise ValueError(f"Unsupported cut '{cut}'. Expected one of {sorted(VALID_CUTS)}.")

        self.cut = cut
        self.dropout = dropout

        self.conv1 = None
        self.conv2 = None

        if cut == "c0":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif cut == "c1":
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        if self.cut == "c0":
            if self.conv1 is None or self.conv2 is None:
                raise RuntimeError("Server expected to own both GCN layers for cut 'c0'.")
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        elif self.cut == "c1":
            if self.conv2 is None:
                raise RuntimeError("Second GCN layer missing for cut 'c1'.")
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # cut == "c2" skips additional convolutions altogether.

        return self.classifier(x)


def create_split_gcn(
    *,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float = 0.5,
    cut: str = "c1",
) -> Tuple[GCNClient, GCNTail]:
    """
    Factory that instantiates the client and server halves of the split GCN.

    Returns
    -------
    (GCNClient, GCNTail)
        Pair of modules ready to be plugged into the SplitFed GNN pipeline.
    """

    cut = cut.lower()
    if cut not in VALID_CUTS:
        raise ValueError(f"Unsupported cut '{cut}'. Expected one of {sorted(VALID_CUTS)}.")

    client = GCNClient(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        cut=cut,
    )

    tail_input = in_dim if cut == "c0" else hidden_dim
    server = GCNTail(
        input_dim=tail_input,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
        cut=cut,
    )

    return client, server
