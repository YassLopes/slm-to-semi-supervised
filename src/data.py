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

def create_semi_supervised_split(tokenized_dataset, dataset, label_encoder, tokenizer, labeled_ratio=0.2):
    """
    Cria divisões de dados para aprendizado semi-supervisionado.
    
    Args:
        tokenized_dataset: Dataset tokenizado
        dataset: Dataset original com rótulos
        label_encoder: Encoder para os rótulos
        tokenizer: Tokenizer para processar os textos
        labeled_ratio: Fração de dados rotulados a serem usados
        
    Returns:
        labeled_dataset: TensorDataset com dados rotulados
        unlabeled_dataset: TensorDataset com dados não rotulados
        unlabeled_indices: Índices dos dados não rotulados
        labeled_indices: Índices dos dados rotulados
        unlabeled_tokenized: Tokens dos dados não rotulados
    """
    # Embaralhar os índices
    train_indices = list(range(len(tokenized_dataset["train"])))
    random.shuffle(train_indices)
    
    # Dividir em conjunto rotulado e não-rotulado
    num_labeled = int(labeled_ratio * len(train_indices))
    labeled_indices = train_indices[:num_labeled]
    unlabeled_indices = train_indices[num_labeled:]
    
    # Função para codificar os rótulos
    def encode_labels(labels):
        return label_encoder.transform(labels)
    
    # Criar conjuntos de dados rotulados
    labeled_texts = [tokenized_dataset["train"][i]["input_text"] for i in labeled_indices]
    labeled_tokenized = tokenizer(labeled_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    labeled_labels = [dataset["train"][i]["output_text"] for i in labeled_indices]
    
    # Converter rótulos para long (corrigido)
    labeled_encoded_labels = torch.tensor(encode_labels(labeled_labels), dtype=torch.long)
    
    labeled_dataset = TensorDataset(
        labeled_tokenized["input_ids"],
        labeled_tokenized["attention_mask"],
        labeled_encoded_labels
    )
    
    # Criar conjuntos de dados não-rotulados
    unlabeled_texts = [tokenized_dataset["train"][i]["input_text"] for i in unlabeled_indices]
    unlabeled_tokenized = tokenizer(unlabeled_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    unlabeled_dataset = TensorDataset(
        unlabeled_tokenized["input_ids"],
        unlabeled_tokenized["attention_mask"]
    )
    
    return labeled_dataset, unlabeled_dataset, unlabeled_indices, labeled_indices, unlabeled_tokenized

def split_train_val(dataset, val_ratio=0.1):
    """
    Divide um dataset em conjuntos de treino e validação.
    
    Args:
        dataset: Dataset a ser dividido
        val_ratio: Fração de dados para validação
        
    Returns:
        train_dataset, val_dataset: Datasets de treino e validação
    """
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size]) 