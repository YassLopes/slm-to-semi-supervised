import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Configura sementes aleatórias para reprodutibilidade.
    
    Args:
        seed: Valor da semente
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    """
    Determina o dispositivo disponível (GPU ou CPU).
    
    Returns:
        device: Dispositivo PyTorch
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_dataset_examples(dataset, num_examples=3):
    """
    Imprime exemplos do dataset para inspeção.
    
    Args:
        dataset: Dataset a ser inspecionado
        num_examples: Número de exemplos a serem exibidos
    """
    print("\nExemplos do dataset:")
    for i in range(min(num_examples, len(dataset['train']))):
        print(f"Exemplo {i+1}:")
        print(f"Sintomas: {dataset['train'][i]['input_text']}")
        print(f"Diagnóstico: {dataset['train'][i]['output_text']}")
        print()

def print_training_summary(train_size, val_size, test_size, num_labels):
    """
    Imprime um resumo dos dados de treinamento.
    
    Args:
        train_size: Tamanho do conjunto de treino
        val_size: Tamanho do conjunto de validação  
        test_size: Tamanho do conjunto de teste
        num_labels: Número de classes
    """
    total_size = train_size + val_size + test_size
    print(f"\nResumo dos dados:")
    print(f"Total de amostras: {total_size}")
    print(f"Treino: {train_size} ({train_size/total_size*100:.1f}%)")
    print(f"Validação: {val_size} ({val_size/total_size*100:.1f}%)")
    print(f"Teste: {test_size} ({test_size/total_size*100:.1f}%)")
    print(f"Número de classes: {num_labels}")

def format_metrics(metrics, prefix=""):
    """
    Formata métricas para exibição.
    
    Args:
        metrics: Dicionário com métricas
        prefix: Prefixo para as métricas
        
    Returns:
        formatted_str: String formatada com as métricas
    """
    if prefix:
        prefix = f"{prefix}_"
    
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{prefix}{key}: {value:.4f}")
        else:
            formatted.append(f"{prefix}{key}: {value}")
    
    return " | ".join(formatted) 