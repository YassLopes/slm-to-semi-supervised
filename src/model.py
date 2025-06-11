import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Dictionary mapping friendly names to HuggingFace model identifiers
MODEL_MAPPINGS = {
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "BlueBERT": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
}

def get_tokenizer(model_name):
    """
    Obtém o tokenizer para o modelo especificado.
    
    Args:
        model_name: Nome do modelo (deve estar em MODEL_MAPPINGS)
        
    Returns:
        tokenizer: Tokenizer configurado
    """
    if model_name in MODEL_MAPPINGS:
        model_id = MODEL_MAPPINGS[model_name]
    else:
        model_id = model_name
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Adicionar padding token se não existir
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer

def get_model(model_name, num_labels=22):
    """
    Cria e retorna o modelo de classificação.
    
    Args:
        model_name: Nome do modelo (deve estar em MODEL_MAPPINGS)
        num_labels: Número de classes para classificação
        
    Returns:
        model: Modelo de classificação configurado
    """
    if model_name in MODEL_MAPPINGS:
        model_id = MODEL_MAPPINGS[model_name]
    else:
        model_id = model_name
    
    # Usar apenas modelos BERT-like para classificação
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
        
    return model

def compute_metrics(eval_pred):
    """
    Calcula as métricas de avaliação para o modelo.
    
    Args:
        eval_pred: Objeto EvalPrediction contendo predictions e label_ids
        
    Returns:
        metrics: Dicionário com métricas calculadas
    """
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    # Calcular métricas
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def evaluate_model(model, eval_dataloader, device):
    """
    Avalia o desempenho do modelo em um dataset.
    
    Args:
        model: Modelo treinado
        eval_dataloader: DataLoader com dados de avaliação
        device: Dispositivo (CPU ou GPU)
        
    Returns:
        metrics: Dicionário com métricas de avaliação
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    } 