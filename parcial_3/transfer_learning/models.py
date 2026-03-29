import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models

class TransferLearningPetModel(pl.LightningModule):
    def __init__(self, model_name='efficientnet_v2_s', num_classes=37, lr=1e-3, freeze_backbone=True):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        
        # Cargar modelo con pesos recomendados: IMAGENET1K_V2
        if model_name == 'efficientnet_v2_s':
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
            self.backbone = models.efficientnet_v2_s(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            
            # Congelar opcionalmente
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            # Reemplazar cabezal para nuestro dataset (siempre descongelado)
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
            
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                    
            # Reemplazar cabezal
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'mobilenet_v3_large':
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            self.backbone = models.mobilenet_v3_large(weights=weights)
            num_features = self.backbone.classifier[3].in_features
            
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                    
            self.backbone.classifier[3] = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Modelo {model_name} no implementado")

    def forward(self, x):
        return self.backbone(x)
        
    def extract_features(self, x):
        """ Extrae características espaciales antes de la última capa lineal """
        if isinstance(self.backbone, models.ResNet):
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            return torch.flatten(x, 1)
            
        elif isinstance(self.backbone, models.EfficientNet):
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            return torch.flatten(x, 1)

        elif isinstance(self.backbone, models.MobileNetV3):
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            return torch.flatten(x, 1)
            
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        # Optimizar solo los pesos que requieren gradiente
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer
