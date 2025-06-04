import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, BartForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Dictionary mapping friendly names to HuggingFace model identifiers
MODEL_MAPPINGS = {
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "BioGPTBART-base": "microsoft/biogpt", # Using BioGPT as BioGPTBART-base wasn't found
    "T5": "t5-base"
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
        
    return AutoTokenizer.from_pretrained(model_id)

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
    
    # Handling for T5 and BART models which require different model classes
    if "t5" in model_id.lower():
        model = T5ForConditionalGeneration.from_pretrained(model_id)
    elif "bart" in model_id.lower() or "biogpt" in model_id.lower():
        model = BartForConditionalGeneration.from_pretrained(model_id)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
        
    return model

def compute_metrics(pred):
    """
    Calcula as métricas de avaliação para o modelo.
    
    Args:
        pred: Saída de predição do modelo
        
    Returns:
        metrics: Dicionário com métricas calculadas
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def generate_pseudo_labels(model, dataloader, device, is_generative=False):
    """
    Gera pseudo-rótulos para dados não rotulados.
    
    Args:
        model: Modelo treinado
        dataloader: DataLoader com dados não rotulados
        device: Dispositivo (CPU ou GPU)
        is_generative: Se o modelo é generativo (T5, BART)
        
    Returns:
        pseudo_labels: Lista de pseudo-rótulos gerados
    """
    model.eval()
    pseudo_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            if is_generative:
                # For T5 and BART models
                outputs = model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    max_length=10  # Assuming short diagnosis labels
                )
                # Convert generated IDs to labels (implement mapping to label indices)
                # This is simplified and would need proper implementation
                predictions = torch.zeros(len(outputs), dtype=torch.long)
            else:
                # For BERT-like models
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            pseudo_labels.extend(predictions.cpu().numpy())
    
    return pseudo_labels

def evaluate_model(model, eval_dataloader, device, is_generative=False):
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: Trained model
        eval_dataloader: DataLoader with evaluation data
        device: Device (CPU or GPU)
        is_generative: If the model is generative (T5, BART)
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            if is_generative:
                # For T5 and BART models
                outputs = model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    max_length=10
                )
                # Convert generated IDs to labels (implementation needed)
                predictions = torch.zeros(len(outputs), dtype=torch.long)
            else:
                # For BERT-like models
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    } 