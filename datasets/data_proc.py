import logging

import torch

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

from sklearn.preprocessing import StandardScaler
import os


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset,
    "custom": None,
}


def load_small_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."

    # --- BLOCK: Custom Graph Loading Logic ---
    if dataset_name == "custom":
        # Hardcoding path to ./data/public_graph_for_mae.pt
        file_path = os.path.join("data", "public_graph_for_mae.pt")
        print(f"Loading custom graph from: {file_path}")

        data_dict = torch.load(file_path)
        x = data_dict["x"]
        edge_index = data_dict["edge_index"]
        num_nodes = x.shape[0]

        # Convert to DGL
        # DGL expects (u, v) tuples or arrays
        src = edge_index[0]
        dst = edge_index[1]
        graph = dgl.graph((src, dst), num_nodes=num_nodes)

        # Preprocessing (Undirected + Self-loops)
        graph = dgl.to_bidirected(graph)
        graph = graph.remove_self_loop().add_self_loop()

        # Set Features
        graph.ndata["feat"] = x.float()

        # Set Labels (Create dummy if missing)
        if "y" in data_dict and data_dict["y"] is not None:
            graph.ndata["label"] = data_dict["y"]
            num_classes = int(data_dict["y"].max().item()) + 1
        else:
            print("No labels found, using dummy labels.")
            graph.ndata["label"] = torch.zeros(num_nodes, dtype=torch.long)
            num_classes = 1

        # Set Masks (Required by training loop)
        # We set all nodes to train since this is unsupervised pre-training
        graph.ndata["train_mask"] = torch.ones(num_nodes, dtype=torch.bool)
        graph.ndata["val_mask"] = torch.ones(num_nodes, dtype=torch.bool)
        graph.ndata["test_mask"] = torch.ones(num_nodes, dtype=torch.bool)

        num_features = graph.ndata["feat"].shape[1]
        return graph, (num_features, num_classes)

    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = (
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
        )
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = (
            train_mask,
            val_mask,
            test_mask,
        )
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


def preprocess(graph):
    # make bidirected
    if "feat" in graph.ndata:
        feat = graph.ndata["feat"]
    else:
        feat = None
    src, dst = graph.all_edges()
    # graph.add_edges(dst, src)
    graph = dgl.to_bidirected(graph)
    if feat is not None:
        graph.ndata["feat"] = feat

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()
    # graph.create_formats_()
    return graph


def scale_feats(x):
    logging.info("### scaling features ###")
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats
