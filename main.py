import argparse

from pathlib import Path

from torch import nn
import torch
from torch.utils.data.dataloader import DataLoader

from training.dataset.lidar_dataset import LiDARDataset, split_train_test
from training.dataset.preprocessing import canonicalize_tree
from training.network import NeuralNetwork
from training.trainer import train


def canonicalize(args):
    if not args.input_path:
        raise ValueError("Canonicalization requires input path and output path")

    if not args.output_path:
        raise ValueError("Canonicalization requires input path and output path")

    canonicalize_tree(args.input_path, args.output_path)


def run_train(args):
    if not args.dataset_path:
        raise ValueError("Dataset path is required to train a model")

    dataset = LiDARDataset(args.dataset_path)

    train_dataset, test_dataset = split_train_test(
        dataset, args.train_split, args.test_split
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch * 4)

    model = NeuralNetwork(2048, len(dataset.available_labels))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    runs_path = Path("./runs/")

    runs_path.mkdir(exist_ok=True)

    if args.run_name is None:
        args.run_name = f"run{len(list(runs_path.iterdir()))}"

    save_path = runs_path / args.run_name
    if save_path.exists() and save_path.is_dir() and any(save_path.iterdir()):
        raise ValueError(
            "The RUN folder already exists and is not empty, make sure to provide a unique name to this run"
        )

    save_path.mkdir(parents=True, exist_ok=True)

    train(
        train_dataloader,
        test_dataloader,
        model,
        loss_fn,
        optimizer,
        epochs=5,
        device="mps",
        save_path=save_path,
        class_names=dataset.available_labels,
    )


def main(args):
    if args.canonicalize:
        canonicalize(args)
    elif args.train:
        run_train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--canonicalize", action="store_true")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch", type=int, default=64)

    main(parser.parse_args())
