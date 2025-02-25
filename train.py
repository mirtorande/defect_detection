import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, dataloader, classification_criterion, segmentation_criterion, optimizer, device, logger, checkpoint_path, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_classification = 0
        epoch_loss_segmentation = 0
        for batch_idx, (images, labels, masks) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            labels = labels - 1  # Da 1-based a 0-based per la classificazione

            optimizer.zero_grad()
            
            # Passaggio in avanti: il modello restituisce classificazione e segmentazione
            class_preds, seg_preds = model(images)
            
            # Calcola la perdita per la classificazione
            class_loss = classification_criterion(class_preds, labels)
            
            # Calcola la perdita per la segmentazione
            masks = masks.unsqueeze(1)
            masks = masks.float()  # Make sure masks are float32
            seg_loss = segmentation_criterion(seg_preds, masks)
            
            # Somma delle perdite (puoi ponderare le perdite se necessario)
            total_loss = class_loss + seg_loss
            
            total_loss.backward()  # Retropropagazione
            optimizer.step()  # Aggiornamento dei pesi

            epoch_loss_classification += class_loss.item()
            epoch_loss_segmentation += seg_loss.item()

            if (batch_idx + 1) % 10 == 0:  # Stampa la perdita ogni 10 batch
                logger.log(f"Batch {batch_idx+1}/{len(dataloader)}: Class Loss = {class_loss.item():.4f}, Seg Loss = {seg_loss.item():.4f}")

        # Calcola la perdita media per ogni tipo di perdita nell'epoca
        avg_epoch_loss_classification = epoch_loss_classification / len(dataloader)
        avg_epoch_loss_segmentation = epoch_loss_segmentation / len(dataloader)
        logger.log(f"Epoch {epoch+1} Classification Loss: {avg_epoch_loss_classification:.4f}")
        logger.log(f"Epoch {epoch+1} Segmentation Loss: {avg_epoch_loss_segmentation:.4f}")

        # Salva i pesi del modello dopo ogni epoca
        print(f"Saving model weights to {checkpoint_path}_epoch{epoch+1}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
        }, f"{checkpoint_path}_epoch{epoch+1}.pth")

    print(f"Renaming checkpoint to {checkpoint_path}.pth")
    os.rename(f"{checkpoint_path}_epoch{num_epochs}.pth", f"{checkpoint_path}")
