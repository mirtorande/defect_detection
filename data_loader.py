import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from preprocessing import Preprocessing, PreprocessingStrategy  # Import the PreprocessingStrategy class

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, dataset_path, transform=None):
        self.csv_file = csv_file
        self.dataset_path = dataset_path
        self.transform = transform
        
        # Load CSV data
        #self.data = pd.read_csv(csv_file)
        self.data = pd.read_csv(csv_file).drop(columns=['EncodedPixels'], errors='ignore')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_path, self.data.iloc[idx, 0])  # ImageId column
        image = Image.open(img_name).convert("RGB")
        label = int(self.data.loc[idx, 'ClassId'])  # ClassId column
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DataLoader_:
    @staticmethod
    def get_dataloader(config, strategy=PreprocessingStrategy.DEFAULT, training=False):
        transform = Preprocessing.get_transforms(strategy)

        # Load full dataset
        dataset = CustomImageDataset(csv_file=config["csv_path"], dataset_path=config["dataset_path"], transform=transform)

        # Load dataset metadata
        df = dataset.data  # Assuming `CustomImageDataset` reads a CSV into `self.data`
        
        # Check if train/test split already exists
        try:
            train_df = pd.read_csv("data/train.csv")
            test_df = pd.read_csv("data/test.csv")
            print("Loaded existing train/test split.")
        except FileNotFoundError:
            # Perform train/test split (first time)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ClassId'])
            
            # Save to CSV for future runs
            train_df.to_csv("data/train.csv", index=False)
            test_df.to_csv("data/test.csv", index=False)
            print("Saved new train/test split.")

        # Get dataset indices corresponding to train/test splits
        train_indices = train_df.index.tolist()
        test_indices = test_df.index.tolist()

        # Create subsets for train and test
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        # Select the appropriate dataset
        selected_dataset = train_dataset if training else test_dataset

        # Create the dataloader
        dataloader = DataLoader(selected_dataset, batch_size=config["batch_size"], shuffle=training)

        return dataloader