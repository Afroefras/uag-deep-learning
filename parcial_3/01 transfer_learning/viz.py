import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from data import inverse_normalize
import math
import numpy as np
from typing import List
import torch

def show_images_batch(images: torch.Tensor, labels: torch.Tensor, class_names: List[str], max_images: int = 6):
    """
    Despliega una cuadrícula (grid) interactiva de las imágenes enviadas, limitando a max_images.
    Usa matplotlib en lugar de plotly para imágenes puras para no sobrecargar el navegador,
    ya que Plotly es mejor para gráficas de datos y scatter plots.
    """
    import matplotlib.pyplot as plt
    
    num_imgs = min(len(images), max_images)
    cols = min(3, num_imgs)
    rows = math.ceil(num_imgs / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten() if num_imgs > 1 else [axes]
    
    for i in range(num_imgs):
        img_tensor = inverse_normalize(images[i])
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        ax = axes[i]
        ax.imshow(img_np)
        ax.axis('off')
        ax.set_title(class_names[labels[i]])
        
    for j in range(num_imgs, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_pca_features(features: np.ndarray, labels: np.ndarray, class_names: List[str], n_components: int = 3):
    """
    Proyecta las características (embeddings) extraídas de la CNN a 2D o 3D usando PCA de scikit-learn
    y genera una gráfica interactiva asombrosa con Plotly.
    """
    from sklearn.decomposition import PCA
    
    # Aplicar PCA para reducir dimensionalidad a n_components
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(features)
    
    # Crear un DataFrame para Plotly Express
    labels_text = [class_names[int(label)] for label in labels]
    
    if n_components == 3:
        df = pd.DataFrame({
            'PC1': components[:, 0],
            'PC2': components[:, 1],
            'PC3': components[:, 2],
            'Raza': labels_text
        })
        fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Raza',
                            title="Representación Interna (Características) extraída de la CNN",
                            opacity=0.8, size_max=5)
    else:
        df = pd.DataFrame({
            'PC1': components[:, 0],
            'PC2': components[:, 1],
            'Raza': labels_text
        })
        fig = px.scatter(df, x='PC1', y='PC2', color='Raza',
                         title="Representación Interna (Características) extraída de la CNN",
                         opacity=0.8)
                         
    fig.update_layout(template='plotly_dark')  # Diseño premium para impacto visual
    fig.show()

def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, class_names: List[str], title: str = "Matriz de Confusión"):
    """
    Despliega una matriz de confusión altamente estética para impresionar a los alumnos,
    usando Plotly. Ideal para comprobar si el modelo está clasificando de forma aleatoria vs aprendida.
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(labels, preds)
    
    # Asegurarnos de que entren todas las etiquetas (a veces son muchas)
    fig = px.imshow(cm,
                    labels=dict(x="Predicción", y="Real", color="Frecuencia"),
                    x=class_names,
                    y=class_names,
                    title=title,
                    color_continuous_scale="Viridis", # Colores premium
                    aspect="auto")
                    
    fig.update_layout(template='plotly_dark',
                      xaxis_tickangle=-45,
                      width=900,
                      height=900,
                      margin=dict(l=50, r=50, t=80, b=150))
    fig.show()
