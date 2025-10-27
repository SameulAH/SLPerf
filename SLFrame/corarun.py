import sys
from pathlib import Path

import torch

from Parse.parseFactory import parseFactory, YAML
from core.dataset.datasetFactory import datasetFactory
from core.model.gnn.gcn_split import create_split_gcn
from core.splitApi import SplitNN_init, SplitNN_distributed


def _resolve_device(preferred):
    if preferred:
        if preferred == "cpu":
            return torch.device("cpu")
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred.startswith("cuda") and torch.cuda.is_available():
            return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_models(parse):
    model_args = parse["model_args"] or {}
    hidden_dim = model_args.get("hidden_dim") or model_args.get("hidden") or 128
    dropout = model_args.get("dropout", 0.5)
    cut = str(model_args.get("cut", "c1")).lower()

    in_dim = int(parse["in_dim"])
    out_dim = int(parse["out_dim"])

    client_model, server_model = create_split_gcn(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
        cut=cut,
    )

    # Store normalized args back on the parse object for later use.
    parse["model_args"] = {
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "cut": cut,
    }

    return client_model, server_model


def main(config_path: str):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    args = parseFactory(fileType=YAML).factory()
    args.load(str(config_path))

    comm, process_id, worker_number = SplitNN_init(args)

    dataset = datasetFactory(args).factory()
    dataset.load_partition_data(process_id)

    client_model, server_model = _build_models(args)
    args["client_model"] = client_model
    args["server_model"] = server_model

    preferred_device = args["device"]
    args["device"] = _resolve_device(preferred_device)

    SplitNN_distributed(process_id, args)


if __name__ == "__main__":
    default_config = Path("./config.yaml")
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    main(cfg_path)
