import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, MobileNet_V2_Weights

class ClassificationWithSegmentationHead(nn.Module):
    def __init__(self, backbone, num_classes, input_channels=3):
        super(ClassificationWithSegmentationHead, self).__init__()

        # Handle the backbone type
        if isinstance(backbone, models.MobileNetV2):
            self.backbone = backbone.features  # Only take the feature extractor part
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(backbone.last_channel, num_classes)  # Use last_channel as input
            )
        else:
            # For other backbones like ResNet
            self.backbone = backbone
            self.classification_head = nn.Linear(backbone.fc.in_features, num_classes)
            backbone.fc = nn.Identity()  # Remove the original fully connected layer

        # Head per la segmentazione (convoluzioni)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),  # Canale in uscita per la maschera binaria
        )

    def forward(self, x):
        # Passaggio attraverso il backbone
        features = self.backbone(x)

        # Ensure features have the correct shape before applying global pooling
        if features.dim() == 4:  # (batch_size, channels, height, width)
            # Global average pooling (apply mean on spatial dimensions)
            features = features.mean([2, 3])  # Global average pooling over height and width
        elif features.dim() == 2:  # If it's already a flattened tensor (MobileNetV2)
            pass  # Don't need to apply pooling
        else:
            raise ValueError(f"Unexpected feature dimensions: {features.dim()}")

        # Classification head
        class_output = self.classification_head(features)

        # Segmentazione: passaggio attraverso la testa di segmentazione
        seg_output = self.segmentation_head(x)

        # Resize segmentation output to match target dimensions
        seg_output_resized = F.interpolate(seg_output, size=(256, 256), mode='bilinear', align_corners=False)

        return class_output, seg_output_resized

class ModelFactory:
    @staticmethod
    def get_model(model_name, num_classes, input_channels=3):
        if model_name == "resnet18":
            backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == "resnet50":
            backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == "mobilenet_v2":
            backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        else:
            raise ValueError(f"Model {model_name} not supported.")
        
        # Use the custom model with classification and segmentation heads
        model = ClassificationWithSegmentationHead(backbone, num_classes, input_channels)
        
        return model
