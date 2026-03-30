import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_pet_data(data_dir='./data', batch_size=32, num_workers=0):
    """
    Descarga y prepara el dataset Oxford-IIIT Pet.
    Aplica transformaciones estándar de ImageNet para Transfer Learning.
    Retorna dataloaders y los nombres de las clases.
    """
    # Transformaciones base (ImageNet standard)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Descargando/Verificando dataset OxfordIIITPet...")
    full_dataset = datasets.OxfordIIITPet(root=data_dir, split='trainval', download=True, transform=transform)

    # Dividimos manual con semilla para reproducibilidad (Regla #4)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Dataloaders - Usando num_workers=0 por default para evitar cuellos de botella en clases con Windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    class_names = full_dataset.classes
    return train_loader, val_loader, class_names

def inverse_normalize(tensor):
    """ Función auxiliar para denormalizar una imagen transformada en ImageNet """
    inv_mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    inv_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    inv_normalize = transforms.Normalize(mean=inv_mean, std=inv_std)
    tensor = inv_normalize(tensor)
    return torch.clamp(tensor, 0, 1)

