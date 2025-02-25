import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, dataloader, criterion, optimizer, device, logger, checkpoint_path, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
            images, labels = images.to(device), labels.to(device)
            labels = labels - 1  # from 1-based to 0-based

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:  # Print loss every 10 batches
                logger.log(f"Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}")

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.log(f"Epoch {epoch+1} loss: {avg_epoch_loss:.4f}")

        # Save model weights after each epoch
        print(f"Saving model weights to {checkpoint_path}_epoch{epoch+1}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
        }, f"{checkpoint_path}_epoch{epoch+1}.pth")

