import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def evaluate_model(model, dataloader, classification_criterion, segmentation_criterion, device, logger):
    model.eval()
    correct = 0
    total = 0
    total_loss_classification = 0.0
    total_loss_segmentation = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, masks in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            labels = labels - 1  # from 1-based to 0-based

            # Forward pass: classification and segmentation
            class_preds, seg_preds = model(images)

            # Calculate classification loss
            class_loss = classification_criterion(class_preds, labels)

            # Calculate segmentation loss
            masks = masks.unsqueeze(1)
            masks = masks.float()  # Make sure masks are float32
            seg_loss = segmentation_criterion(seg_preds, masks)

            # Total loss
            total_loss_classification += class_loss.item()
            total_loss_segmentation += seg_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(class_preds, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total * 100
    print(f'Accuracy: {accuracy:.2f}%')
    
    avg_loss_classification = total_loss_classification / len(dataloader)
    avg_loss_segmentation = total_loss_segmentation / len(dataloader)
    logger.log(f"Evaluation Classification Loss: {avg_loss_classification:.4f}")
    logger.log(f"Evaluation Segmentation Loss: {avg_loss_segmentation:.4f}")

    # Visualize predictions
    visualize_predictions(model, dataloader, device)

    return accuracy

def visualize_predictions(model, dataloader, device, num_images=5):
    model.eval()
    images, labels, masks = next(iter(dataloader))
    images, labels, masks = images.to(device), labels.to(device), masks.to(device)
    labels = labels - 1  # from 1-based to 0-based

    # Resize masks to match image size (if necessary)
    resize_transform = transforms.Resize((224, 224))  # Match image size

    with torch.no_grad():
        # Get predictions
        class_preds, seg_preds = model(images)
        _, predicted_class = torch.max(class_preds, 1)

        # Get segmentation masks (after applying sigmoid to logits)
        seg_preds = torch.sigmoid(seg_preds)
        seg_preds = (seg_preds > 0.4).float()  # Binarize predictions
        seg_preds = resize_transform(seg_preds)

    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
    for i in range(num_images):
        image = images[i].cpu().numpy().transpose((1, 2, 0))
        image = np.clip(image, 0, 1)

        # Plot the image
        axes[i].imshow(image)
        axes[i].set_title(f'Class Pred: {predicted_class[i].item()} \nLabel: {labels[i].item()}')

        # Plot the segmentation mask
        seg_mask = seg_preds[i].squeeze(0).cpu().numpy()  # Remove the channel dimension
        axes[i].imshow(seg_mask, alpha=0.5, cmap='viridis')  # Overlay segmentation mask
        axes[i].axis('off')

    plt.savefig('output/predictions_with_masks.png')  # Saves the plot as an image file
    plt.show()
