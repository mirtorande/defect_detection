import os
import yaml
import torch

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Function to load the model weights if available
def load_weights(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Weights loaded successfully.")
    else:
        print("No checkpoint found, starting from scratch.")
    return model