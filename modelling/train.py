from torch.utils.data import DataLoader
from datetime import datetime
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm
from torch import nn
import torch
import os

CHKPT_DIR = Path("data/models/checkpoints")
if not os.path.isdir(CHKPT_DIR):
     os.mkdir("data/models")
     os.mkdir(CHKPT_DIR)


def train(
        model : nn.Module,
        loss_fn : nn.Module,
        train_dl : DataLoader,
        val_dl : DataLoader,
        learning_rate: float,
        max_epochs: int,
        l2_lambda: float = 0.0,) -> None:
    now = datetime.now()
    formatted_now = now.strftime('%Y-%m-%d_%H:%M')
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    early_stopping_threshold_count = 0
    best_val_loss = float("inf")

    for epoch in range(max_epochs):
        total_loss_train = 0
        model.train()

        for train_in, train_label in tqdm(train_dl):
            output = model(train_in)
            loss = loss_fn(output, train_label)
            total_loss_train += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        with torch.no_grad():
            total_loss_val = 0

            for val_input, val_label in tqdm(val_dl):
                output = model(val_input)
                loss = loss_fn(output, val_label)
                total_loss_val += loss.item()
        print(f'Epochs: {epoch + 1} '
            f'| Train Loss: {total_loss_train / len(train_dl): .3f} '
            f'| Val Loss: {total_loss_val / len(val_dl): .3f} ')
        
        early_stopping_threshold_count += 1
        if best_val_loss > total_loss_val:
            best_val_loss = total_loss_val
            torch.save({
                 'epoch': epoch,
                 "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "val_loss": total_loss_val / len(val_dl),
                 "train_loss": total_loss_train / len(train_dl)
            }, CHKPT_DIR / f"best_model_{formatted_now}_epoch_{epoch}.pth")
            print(f"[+] Saved checkpoint 'best_model_{formatted_now}_epoch_{epoch}.pth'")
            early_stopping_threshold_count = 0

        if early_stopping_threshold_count >= 1:
                print("Early stopping")
                break

