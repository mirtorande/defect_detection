import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 

def evaluate_model(model, dataloader, criterion, device, logger):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            labels = labels - 1  # from 1-based to 0-based

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total * 100
    print(f'Accuracy: {accuracy:.2f}%')
    avg_loss = total_loss / len(dataloader)
    logger.log(f"Evaluation loss: {avg_loss:.4f}")
    
    visualize_predictions(model, dataloader, device)
    
    return accuracy

def visualize_predictions(model, dataloader, device, num_images=5):
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    labels = labels - 1  # from 1-based to 0-based
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
    for i in range(num_images):
        image = images[i].cpu().numpy().transpose((1, 2, 0))
        image = np.clip(image, 0, 1)
        
        axes[i].imshow(image)
        axes[i].set_title(f'Pred: {predicted[i].item()} \nLabel: {labels[i].item()}')
        axes[i].axis('off')
    plt.savefig('output/predictions.png')  # Saves the plot as an image file

    plt.show()
