import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from preprocessing import PreprocessingStrategy
from data_loader import DataLoader_
from model_factory import ModelFactory
from train import train_model
from utils import load_config, load_weights
from evaluate import evaluate_model
from logger import Logger


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train or evaluate the model.")
    parser.add_argument('mode', nargs='?', default='evaluate', choices=['train', 'evaluate'],
                        help="Set to 'train' to train the model, otherwise it will evaluate.")
    args = parser.parse_args()

    config = load_config()
    logger = Logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Dataset
    dataloader = DataLoader_.get_dataloader(config, PreprocessingStrategy.AUGMENTED, args.mode == 'train')

    # Load Model
    model = ModelFactory.get_model(config["model"], config["num_classes"]).to(device)
    checkpoint_path = "weights\model_checkpoint.pth"
    classification_criterion = nn.CrossEntropyLoss()
    segmentation_criterion = nn.BCEWithLogitsLoss()  # Per la segmentazione binaria

    # Load weights if available
    if args.mode == 'evaluate':
        model = load_weights(model, checkpoint_path)
        evaluate_model(model, dataloader, classification_criterion, segmentation_criterion, device, logger)

    elif args.mode == 'train':
        
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        
        # Train Model
        train_model(model, dataloader, classification_criterion, segmentation_criterion, optimizer, device, logger, checkpoint_path, config["num_epochs"])
