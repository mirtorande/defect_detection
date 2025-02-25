import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from preprocessing import Preprocessing, PreprocessingStrategy  # Import the PreprocessingStrategy class

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, dataset_path, transform=None):
        self.csv_file = csv_file
        self.dataset_path = dataset_path
        self.transform = transform
        
        # Load CSV data
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_path, self.data.iloc[idx, 0])  # ImageId column
        image = Image.open(img_name).convert("RGB")
        label = int(self.data.iloc[idx, 1])  # ClassId column
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DataLoader_:
    @staticmethod
    def get_dataloader(config, strategy=PreprocessingStrategy.DEFAULT):
        transform = Preprocessing.get_transforms(strategy)
        
        # Create the custom dataset using the CSV file
        dataset = CustomImageDataset(csv_file=config["csv_path"], dataset_path=config["dataset_path"], transform=transform)
        print(dataset.data['ClassId'].unique())

        
        # Create the dataloader
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        
        return dataloader
