import os
import yaml
import torch
import numpy as np

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

def rle_to_mask(rle, shape=(256, 256)):
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if rle != "-1":  # Se c'è un difetto
        rle = list(map(int, rle.split()))
        for start, length in zip(rle[0::2], rle[1::2]):
            mask[start : start + length] = 1
    
    return mask.reshape(shape).T  # Converte in matrice