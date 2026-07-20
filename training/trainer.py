from pathlib import Path

import torch
from tqdm import tqdm

from utils.logger import logger


def _train(dataloader, model, loss_fn, optimizer, device="mps"):
    size = len(dataloader)
    model.train()

    pbar = tqdm(dataloader, total=size, desc="Running epoch")

    for X, y in pbar:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        pbar.set_postfix(loss=f"{loss:>7f}")


def _test(dataloader, model, loss_fn, device="mps"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct_top1, correct_top3 = 0, 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader, total=num_batches, desc="Running test"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * X.size(0)

            # Top-1
            correct_top1 += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Top-3
            _, top3 = pred.topk(3, dim=1)
            correct_top3 += (
                top3.eq(y.view(-1, 1)).any(dim=1).type(torch.float).sum().item()
            )

    test_loss /= num_batches
    correct_top1 /= size
    correct_top3 /= size
    logger.info(
        f"Test Error: \n"
        f"Top-1 Accuracy: {(100 * correct_top1):>0.1f}%, "
        f"Top-3 Accuracy: {(100 * correct_top3):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )


def save_model(model, class_names, path):
    """Save the model weights and class names into a single .pt file."""
    if isinstance(class_names, dict):
        # available_labels maps name -> index; store an index-ordered list.
        class_names = [
            name for name, _ in sorted(class_names.items(), key=lambda item: item[1])
        ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Move weights to CPU so the checkpoint loads on any device.
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": state_dict,
            "class_names": list(class_names),
        },
        path,
    )
    logger.info("Model successfully saved")


def train(
    train_dataloader,
    test_dataloader,
    model,
    loss_fn,
    optimizer,
    save_path: Path,
    epochs=5,
    device="mps",
    class_names=None,
):
    save_path.mkdir(parents=True, exist_ok=True)

    for t in range(epochs):
        logger.info(f"Epoch {t + 1}\n-------------------------------")
        model = model.to(device)
        _train(train_dataloader, model, loss_fn, optimizer, device)
        _test(test_dataloader, model, loss_fn, device)

    logger.info("Training finished...")

    model_save_path = save_path / "weights.pt"

    logger.info(f"Saving model at {model_save_path.absolute()}")

    if class_names is None:
        class_names = getattr(train_dataloader.dataset, "available_labels", {})
    save_model(model, class_names, model_save_path)
