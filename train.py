import torch
from model import model
from metrics import iou
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm.notebook import tqdm
from dataset import train_dataloader,valid_dataloader

device="cuda" if torch.cuda.is_available() else "cpu"
lr=0.001
n_epochs=10
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=lr)


def main():
    model.to(device)
    train_loss_history = []
    val_loss_history = []
    train_iou_history = []
    val_iou_history = []

    for epoch in range(n_epochs):
        train_loss = 0
        train_iou = 0
        model.train()
        for i, batch in tqdm(enumerate(train_dataloader)):
            inputs = batch["image"].to(device)
            labels = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            _iou = iou(outputs.detach().cpu().numpy() >= 0, labels.detach().cpu().numpy())
            train_iou += _iou

        train_loss /= len(train_dataloader)
        train_iou /= len(train_dataloader)
        train_loss_history.append(train_loss)
        train_iou_history.append(train_iou)

        val_loss = 0
        val_iou = 0
        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(valid_dataloader)):
                inputs = batch["image"].to(device)
                labels = batch["mask"].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _iou = iou(outputs.detach().cpu().numpy() >= 0, labels.detach().cpu().numpy())
                val_iou += _iou

        val_loss /= len(valid_dataloader)
        val_iou /= len(valid_dataloader)
        val_loss_history.append(val_loss)
        val_iou_history.append(val_iou)

        print(
            "Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Train IOU: {:.4f}, Val IOU: {:.4f}".format(
                epoch + 1, n_epochs, train_loss, val_loss, train_iou, val_iou
            )
        )