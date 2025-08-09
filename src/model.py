import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class UnifiedNet(nn.Module):
    """
    A unified network for feature extraction based on ResNet-50.
    """
    def __init__(self, embedding_dim=128):
        super(UnifiedNet, self).__init__()
        # Load a pre-trained ResNet-50
        weights = ResNet50_Weights.DEFAULT
        base_model = resnet50(weights=weights)
        
        # The feature extractor is the model without the final classification layer
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # Get the number of features from the base model
        in_features = base_model.fc.in_features
        
        # The embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        print(f"Model initialized. Embedding dimension: {embedding_dim}")

    def forward(self, x):
        # Extract features from the base model
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        
        # Generate embeddings
        embedding = self.embedding_head(features)
        
        return embedding
