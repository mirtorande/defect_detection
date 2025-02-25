from enum import Enum
import torchvision.transforms as transforms

class PreprocessingStrategy(Enum):
    DEFAULT = "default"
    AUGMENTED = "augmented"

class Preprocessing:
    @staticmethod
    def get_transforms(strategy=PreprocessingStrategy.DEFAULT):
        if strategy == PreprocessingStrategy.DEFAULT:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif strategy == PreprocessingStrategy.AUGMENTED:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Preprocessing strategy {strategy} not supported.")
