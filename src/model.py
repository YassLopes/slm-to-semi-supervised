import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

def get_tokenizer(model_name="distilbert/distilbert-base-uncased"):
    """
    Obtém o tokenizer para o modelo especificado.
    
    Args:
        model_name: Nome do modelo pré-treinado
        
    Returns:
        tokenizer: Tokenizer configurado
    """
    return AutoTokenizer.from_pretrained(model_name)

def get_model(model_name="distilbert/distilbert-base-uncased", num_labels=22):
    """
    Cria e retorna o modelo de classificação.
    
    Args:
        model_name: Nome do modelo pré-treinado
        num_labels: Número de classes para classificação
        
    Returns:
        model: Modelo de classificação configurado
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
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

def generate_pseudo_labels(model, dataloader, device):
    """
    Gera pseudo-rótulos para dados não rotulados.
    
    Args:
        model: Modelo treinado
        dataloader: DataLoader com dados não rotulados
        device: Dispositivo (CPU ou GPU)
        
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
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            pseudo_labels.extend(predictions.cpu().numpy())
    
    return pseudo_labels 