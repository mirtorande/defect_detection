import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import MobileNet_V2_Weights

class ModelFactory:
    @staticmethod
    def get_model(model_name, num_classes):
        if model_name == "resnet18":
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported.")
        
        return model
