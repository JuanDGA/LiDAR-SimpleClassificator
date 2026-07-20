#
# Copyright 2026 Juan David Guevara Arévalo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from collections import OrderedDict
from pathlib import Path
from typing import BinaryIO

import torch
from torch.utils.data.dataset import Dataset

from utils.logger import logger

# Cap on simultaneously open CSV handles per process, so datasets with many
# files stay well below the OS file descriptor limit.
_MAX_OPEN_FILES = 64


class LiDARDataset(Dataset):
    """Lazy-loading dataset for CSV-encoded LiDAR samples.

    Expects a folder containing one subfolder per class, each holding CSV
    files with one sample per line:

        dataset_folder/
        - class_a/
          - encodings_a.csv
          - encodings_b.csv
        - class_b/
          - encodings_a.csv

    Samples are kept on disk. __init__ only records the byte offset of every
    line, and __getitem__ seeks to that offset and parses the line on demand.
    """

    def __init__(self, dataset_folder: Path) -> None:
        if not dataset_folder.exists():
            raise ValueError("The provided path does not exists")

        if not dataset_folder.is_dir():
            raise ValueError("The provided path must be a folder")

        self.available_labels: dict[str, int] = {}
        # One (csv_path, byte_offset, label) entry per sample.
        self._index: list[tuple[Path, int, int]] = []
        self._pid: int | None = None
        self._handles: OrderedDict[Path, BinaryIO] = OrderedDict()

        with logger.status("Indexing..."):
            for child in sorted(dataset_folder.iterdir()):
                if not child.is_dir():
                    logger.warning(f"{child.name} is not a folder, skipping...")
                    continue

                label = self.available_labels.setdefault(
                    child.name, len(self.available_labels)
                )

                for encoding_file in sorted(child.iterdir()):
                    if encoding_file.suffix != ".csv":
                        logger.warning(
                            f"{child.name}/{encoding_file.name} is not a CSV file, skipping..."
                        )
                        continue
                    self._index_file(encoding_file, label)

        logger.info(
            f"Indexed {len(self._index)} samples "
            f"across {len(self.available_labels)} classes"
        )

    def _index_file(self, path: Path, label: int) -> None:
        with open(path, "rb") as file:
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                if line.strip():
                    self._index.append((path, offset, label))

    def _handle_for(self, path: Path) -> BinaryIO:
        pid = os.getpid()
        if self._pid != pid:
            # After a fork (DataLoader workers) each process must own its
            # handles, otherwise they would share file offsets.
            self._handles = OrderedDict()
            self._pid = pid

        handle = self._handles.get(path)
        if handle is not None:
            self._handles.move_to_end(path)
            return handle

        handle = open(path, "rb")
        self._handles[path] = handle
        if len(self._handles) > _MAX_OPEN_FILES:
            _, evicted = self._handles.popitem(last=False)
            evicted.close()
        return handle

    def __getstate__(self) -> dict:
        # Open handles cannot be pickled (DataLoader with spawn context).
        state = self.__dict__.copy()
        state["_pid"] = None
        state["_handles"] = OrderedDict()
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            handle.close()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, offset, label = self._index[index]

        handle = self._handle_for(path)
        handle.seek(offset)
        # float() tolerates the trailing newline, so no strip is needed.
        data = torch.tensor(
            list(map(float, handle.readline().split(b","))), dtype=torch.float32
        )

        target = torch.tensor(label, dtype=torch.long)

        return data, target

    @classmethod
    def _view(cls, dataset: "LiDARDataset", indices: list[int]) -> "LiDARDataset":
        view = cls.__new__(cls)
        view.available_labels = dict(dataset.available_labels)
        view._index = [dataset._index[i] for i in indices]
        view._pid = None
        view._handles = OrderedDict()
        return view


def split_train_test(
    dataset: LiDARDataset, train: float = 0.8, test: float = 0.2
) -> tuple[LiDARDataset, LiDARDataset]:
    if train < 0 or test < 0:
        raise ValueError("train and test must be non-negative")

    if train + test > 1.0 + 1e-4:
        raise ValueError("train and test must sum to at most 1.0")

    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot split an empty dataset")

    indices = torch.randperm(n).tolist()

    train_size = int(round(train * n))
    test_size = int(round(test * n))

    # Ensure we never exceed the dataset size
    train_size = min(train_size, n)
    test_size = min(test_size, n - train_size)

    train_indices = indices[:train_size]
    test_indices = indices[train_size : train_size + test_size]

    return (
        LiDARDataset._view(dataset, train_indices),
        LiDARDataset._view(dataset, test_indices),
    )
