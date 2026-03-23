import torch
from tqdm import tqdm

def train_and_val(model, 
                  train_loader,
                  val_loader,
                  optimizer,
                  loss_fn,
                  epochs=30,
                  device="cpu"):
    for epoch in range(epochs):
        train_loss = 0
        total_train_loss = 0
        model.train()
        for img, mask in tqdm(train_loader):
            img = img.to(device)
            mask = mask.to(device).float() / 255.0

            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        total_train_loss += train_loss / len(train_loader)

        val_loss = 0
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for img, mask in tqdm(val_loader):
                img = img.to(device)
                mask = mask.to(device).float() / 255.0
                output = model(img)
                loss = loss_fn(output, mask)
                val_loss += loss.item()

        total_val_loss += val_loss / len(val_loader)

        print(f"epoch: {epoch+1} | train_loss: {total_train_loss} | val_loss: {total_val_loss}")

    return model