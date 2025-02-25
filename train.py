import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, dataloader, criterion, optimizer, device, logger, checkpoint_path):
    model.train()
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training", unit="batch")):
        images, labels = images.to(device), labels.to(device)
        labels = labels - 1  # from 1-based to 0-based

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:  # Print loss every 10 batches
            logger.log(f"Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}")

    logger.log(f"Training loss: {loss.item():.4f}")

    # Save weights after training
    print(f"Saving model weights to {checkpoint_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)