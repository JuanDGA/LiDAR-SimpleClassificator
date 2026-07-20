<!--
Copyright 2026 Juan David Guevara Arévalo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LiDAR-SimpleClassificator

A minimal, end-to-end training pipeline for classifying LiDAR point-cloud encodings.
The project reads CSV-encoded feature vectors, trains a small fully-connected neural
network, and exports a single checkpoint with class labels so it can be reused for
inference.

## Overview

This repository is the companion classifier for
[JuanDGA/LIVOX-MID360-SimpleDriver](https://github.com/JuanDGA/LIVOX-MID360-SimpleDriver).
The encodings used as input must be generated with that driver, which turns raw
Livox MID-360 point clouds into compact numerical representations.

Once the encodings are available, this project lets you:

- Reorganize raw encoding folders into a class-ready dataset.
- Train a neural network classifier on the CSV encodings.
- Save the trained weights and class names in a single `.pt` file.

## How it works

The dataset is expected to look like this:

```text
dataset/
  class_a/
    encodings_a.csv
    encodings_b.csv
  class_b/
    encodings_a.csv
```

Each CSV file contains one sample per line, and each line is a comma-separated list
of floating point values (a fixed-size encoding vector). Folder names are used as
class labels.

The model is a simple MLP with batch normalization and ReLU activations, trained
with cross-entropy loss and SGD. Training reports top-1 and top-3 accuracy on the
held-out test split.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/) (recommended) or any PEP 517-compatible builder

## Installation

Using `uv`:

```bash
uv sync
```

Or with a plain virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### 1. Generate the encodings

First, use the LiDAR driver to generate the encodings:

<https://github.com/JuanDGA/LIVOX-MID360-SimpleDriver>

### 2. Canonicalize the dataset

If the driver produced folders with an `encodings/` subfolder per class, run:

```bash
uv run main.py --canonicalize \
  --input-path ./raw_encodings \
  --output-path ./dataset
```

This copies every CSV from `raw_encodings/<class>/encodings/` into a flat
`dataset/<class>/` structure.

### 3. Train the classifier

```bash
uv run main.py --train \
  --dataset-path ./dataset \
  --train-split 0.8 \
  --test-split 0.2 \
  --batch 64 \
  --run-name my_first_run
```

The checkpoint will be written to `runs/my_first_run/weights.pt`.

### Available arguments

| Flag | Description | Default |
| --- | --- | --- |
| `--canonicalize` | Reorganize raw encoding folders into a dataset | `False` |
| `--input-path` | Source folder for canonicalization | `None` |
| `--output-path` | Destination folder for canonicalization | `None` |
| `--train` | Train a model | `False` |
| `--dataset-path` | Path to the canonicalized dataset | `None` |
| `--train-split` | Fraction of samples used for training | `0.8` |
| `--test-split` | Fraction of samples used for testing | `0.2` |
| `--batch` | Training batch size | `64` |
| `--run-name` | Name of the run folder under `runs/` | auto-generated |

## Project structure

```text
lidar_ml/
├── main.py                    # CLI entry point
├── training/
│   ├── dataset/
│   │   ├── lidar_dataset.py   # Lazy-loading CSV dataset
│   │   └── preprocessing.py   # Dataset canonicalization
│   ├── network.py             # Neural network definition
│   └── trainer.py             # Training and evaluation loop
├── utils/
│   └── logger.py              # Rich-based logging helpers
├── pyproject.toml
├── LICENSE
└── README.md
```

## Model

The classifier is a feed-forward network with the following architecture:

```text
Input (2048)
  -> Linear(1024) -> BatchNorm1d -> ReLU
  -> Linear(1024) -> BatchNorm1d -> ReLU
  -> Linear(512)  -> BatchNorm1d -> ReLU
  -> Linear(num_classes)
```

The input size is fixed at 2048 dimensions, matching the encoding produced by the
driver. The number of output classes is inferred from the dataset folders.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE)
file for details.

## Author

Juan David Guevara Arévalo
