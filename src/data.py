import torch
from torch.utils.data import TensorDataset, random_split
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import random
import numpy as np

def load_symptom_diagnosis_dataset(max_samples=None):
    """
    Carrega o dataset de sintomas e diagnósticos.
    
    Args:
        max_samples: Se especificado, limita o número de amostras para teste rápido
        
    Returns:
        dataset: Dataset carregado
    """
    dataset = load_dataset("gretelai/symptom_to_diagnosis")
    
    if max_samples and len(dataset['train']) > max_samples:
        dataset['train'] = dataset['train'].select(range(max_samples))
    
    return dataset

def prepare_labels(dataset):
    """
    Prepara os rótulos e retorna um encoder para os diagnósticos.
    
    Args:
        dataset: Dataset contendo rótulos em 'output_text'
        
    Returns:
        label_encoder: O encoder treinado nos rótulos
        num_labels: Número de classes únicas
    """
    label_encoder = LabelEncoder()
    all_labels = dataset['train']['output_text']
    label_encoder.fit(all_labels)
    num_labels = len(label_encoder.classes_)
    
    return label_encoder, num_labels

def tokenize_dataset(dataset, tokenizer):
    """
    Tokeniza o dataset usando o tokenizer fornecido.
    
    Args:
        dataset: Dataset a ser tokenizado
        tokenizer: Tokenizer a ser usado
        
    Returns:
        tokenized_dataset: Dataset tokenizado
    """
    def tokenize_function(examples):
        return tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=128)
    
    return dataset.map(tokenize_function, batched=True)

def create_supervised_dataset(dataset, label_encoder, tokenizer):
    """
    Cria um dataset supervisionado completo para treinamento.
    
    Args:
        dataset: Dataset original
        label_encoder: Encoder para os rótulos
        tokenizer: Tokenizer para processar os textos
        
    Returns:
        supervised_dataset: TensorDataset com todos os dados rotulados
    """
    # Obter todos os textos e rótulos
    texts = dataset["train"]["input_text"]
    labels = dataset["train"]["output_text"]
    
    # Tokenizar textos
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    # Codificar rótulos
    encoded_labels = torch.tensor(label_encoder.transform(labels), dtype=torch.long)
    
    # Criar TensorDataset
    supervised_dataset = TensorDataset(
        tokenized["input_ids"],
        tokenized["attention_mask"],
        encoded_labels
    )
    
    return supervised_dataset

def split_train_val_test(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Divide um dataset em conjuntos de treino, validação e teste.
    
    Args:
        dataset: Dataset a ser dividido
        train_ratio: Fração de dados para treino
        val_ratio: Fração de dados para validação
        test_ratio: Fração de dados para teste
        
    Returns:
        train_dataset, val_dataset, test_dataset: Datasets divididos
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    return random_split(dataset, [train_size, val_size, test_size]) 