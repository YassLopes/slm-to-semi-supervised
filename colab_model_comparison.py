#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para Google Colab que demonstra a comparação de modelos de linguagem médica/científica 
para classificação de sintomas usando aprendizado semi-supervisionado.

Para executar no Google Colab, faça o seguinte:
1. Faça upload deste arquivo para o Colab
2. Execute as células abaixo
"""

import os
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    TrainingArguments, 
    Trainer
)
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dataclasses import dataclass
import pandas as pd
from prettytable import PrettyTable

# Mapeamento de modelos
MODEL_MAPPINGS = {
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "BioGPTBART-base": "microsoft/biogpt",  # Usando BioGPT como proxy
    "T5": "t5-base"
}

# Configurações
OUTPUT_DIR = "./results"
MAX_SAMPLES = 500  # Limite de amostras para execução rápida no Colab
LABELED_RATIO = 0.2
SEED = 42
MODELS_TO_COMPARE = ["BioBERT", "ClinicalBERT"]  # Para execução rápida, use apenas dois modelos

# Funções utilitárias
def set_seed(seed):
    """Define sementes para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Determina o dispositivo de execução (GPU ou CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Processamento de dados
def load_symptom_diagnosis_dataset(max_samples=None):
    """Carrega o dataset de sintomas e diagnósticos."""
    dataset = load_dataset("gretelai/symptom_to_diagnosis")
    
    if max_samples and len(dataset['train']) > max_samples:
        dataset['train'] = dataset['train'].select(range(max_samples))
    
    return dataset

def prepare_labels(dataset):
    """Prepara os rótulos e retorna um encoder para os diagnósticos."""
    label_encoder = LabelEncoder()
    all_labels = dataset['train']['output_text']
    label_encoder.fit(all_labels)
    num_labels = len(label_encoder.classes_)
    
    return label_encoder, num_labels

def tokenize_dataset(dataset, tokenizer):
    """Tokeniza o dataset usando o tokenizer fornecido."""
    def tokenize_function(examples):
        return tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=128)
    
    return dataset.map(tokenize_function, batched=True)

def create_semi_supervised_split(tokenized_dataset, dataset, label_encoder, tokenizer, labeled_ratio=0.2):
    """Cria divisões de dados para aprendizado semi-supervisionado."""
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
    
    # Converter rótulos para long
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
    """Divide um dataset em conjuntos de treino e validação."""
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])

# Definição do modelo e métricas
def get_tokenizer(model_name):
    """Obtém o tokenizer para o modelo especificado."""
    if model_name in MODEL_MAPPINGS:
        model_id = MODEL_MAPPINGS[model_name]
    else:
        model_id = model_name
        
    return AutoTokenizer.from_pretrained(model_id)

def get_model(model_name, num_labels=22):
    """Cria e retorna o modelo de classificação."""
    if model_name in MODEL_MAPPINGS:
        model_id = MODEL_MAPPINGS[model_name]
    else:
        model_id = model_name
    
    # Handling for T5 and BART models which require different model classes
    if "t5" in model_id.lower():
        # Use versão para classificação de sequência em vez da generativa
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    elif "bart" in model_id.lower() or "biogpt" in model_id.lower():
        # Use versão para classificação de sequência em vez da generativa
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
        
    return model

def compute_metrics(pred):
    """Calcula as métricas de avaliação para o modelo."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def generate_pseudo_labels(model, dataloader, device, is_generative=False):
    """Gera pseudo-rótulos para dados não rotulados."""
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
                # Convert generated IDs to labels (implementation needed)
                predictions = torch.zeros(len(outputs), dtype=torch.long)
            else:
                # For BERT-like models
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            pseudo_labels.extend(predictions.cpu().numpy())
    
    return pseudo_labels

def evaluate_model(model, eval_dataloader, device, is_generative=False):
    """Avalia o desempenho do modelo em um dataset."""
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

# Funções de treinamento
@dataclass
class DataCollatorForTextClassification:
    """Collator personalizado para classificação de textos."""
    def __call__(self, features):
        # Cada feature é (input_ids, attention_mask, label) de um TensorDataset
        if isinstance(features[0], tuple):
            input_ids = torch.stack([f[0] for f in features])
            attention_mask = torch.stack([f[1] for f in features])
            
            if len(features[0]) > 2:  # Se tiver rótulos
                # Garantir que os rótulos sejam do tipo long
                labels = torch.stack([f[2] for f in features]).long()
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        
        # Fallback para o comportamento padrão
        batch = {k: torch.stack([f[k] for f in features]) for k in features[0].keys()}
        
        # Garantir que os rótulos sejam long se existirem
        if "labels" in batch:
            batch["labels"] = batch["labels"].long()
            
        return batch

def get_training_args(output_dir, num_train_epochs=3, batch_size=16, eval_batch_size=64):
    """Configura os argumentos de treinamento."""
    # Usar apenas argumentos essenciais básicos
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
    )
    
    return training_args

def create_pseudo_labeled_dataset(unlabeled_tokenized, pseudo_labels):
    """Cria um dataset com pseudo-rótulos."""
    return TensorDataset(
        unlabeled_tokenized["input_ids"],
        unlabeled_tokenized["attention_mask"],
        torch.tensor(pseudo_labels, dtype=torch.long)  # Garantir que os rótulos sejam do tipo long
    )

def train_semi_supervised_model(model, pseudo_labeled_dataset, labeled_train_dataset, labeled_val_dataset, compute_metrics, output_dir="./results_semi_supervised", combined_training=True):
    """Treina o modelo usando dados rotulados e pseudo-rotulados."""
    # Verificar se devemos combinar ou usar apenas dados rotulados
    if combined_training and pseudo_labeled_dataset is not None:
        # Combinar dados rotulados com pseudo-rotulados
        combined_dataset = ConcatDataset([labeled_train_dataset, pseudo_labeled_dataset])
        train_dataset = combined_dataset
        print(f"Treinando com dataset combinado: {len(combined_dataset)} amostras")
    else:
        # Usar apenas dados rotulados
        train_dataset = labeled_train_dataset
        print(f"Treinando apenas com dados rotulados: {len(labeled_train_dataset)} amostras")
    
    # Configurar treinamento
    training_args = get_training_args(
        output_dir=output_dir,
        num_train_epochs=3  # Ajuste conforme necessário
    )
    
    data_collator = DataCollatorForTextClassification()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=labeled_val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,  # Usar o collator personalizado
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    
    return trainer, eval_results

def train_and_evaluate_model(model_name, dataset, label_encoder, num_labels, device, labeled_ratio, output_dir, seed):
    """Função para treinar e avaliar um modelo específico."""
    print(f"\n{'='*50}")
    print(f"Treinando e avaliando modelo: {model_name}")
    print(f"{'='*50}")
    
    # Inicializar tokenizer e modelo
    tokenizer = get_tokenizer(model_name)
    
    # Tokenizar o dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Criar divisão semi-supervisionada
    labeled_dataset, unlabeled_dataset, unlabeled_indices, labeled_indices, unlabeled_tokenized = create_semi_supervised_split(
        tokenized_dataset, dataset, label_encoder, tokenizer, labeled_ratio=labeled_ratio
    )
    
    # Dividir dados rotulados em treino e validação
    labeled_train_dataset, labeled_val_dataset = split_train_val(labeled_dataset)
    
    # Verificar se o modelo é generativo
    is_generative = model_name in ["T5", "BioGPTBART-base"]
    
    # Inicializar o modelo
    model = get_model(model_name, num_labels=num_labels)
    model.to(device)
    
    # FASE 1: Treinamento inicial com dados rotulados
    print(f"Treinando modelo inicial com {len(labeled_train_dataset)} amostras rotuladas...")
    model_output_dir = f"{output_dir}/{model_name}"
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Para modelos generativos, é necessário implementar treinamento específico
    if is_generative:
        print("Modelo generativo detectado - usando abordagem simplificada")
    
    # Treinamos o modelo inicial
    supervised_trainer, _ = train_semi_supervised_model(
        model, None, labeled_train_dataset, labeled_val_dataset, 
        compute_metrics, output_dir=f"{model_output_dir}/initial",
        combined_training=False
    )
    
    # FASE 2: Pseudo-rotulagem
    print("Gerando pseudo-rótulos para dados não rotulados...")
    
    # Criar DataLoader para dados não rotulados
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Gerar pseudo-rótulos com o modelo treinado
    pseudo_labels = generate_pseudo_labels(model, unlabeled_dataloader, device, is_generative=is_generative)
    
    # Criar dataset com pseudo-rótulos
    pseudo_labeled_dataset = create_pseudo_labeled_dataset(unlabeled_tokenized, pseudo_labels)
    
    # FASE 3: Treinamento semi-supervisionado
    print(f"Treinando modelo com dataset combinado (rotulado + pseudo-rotulado)...")
    semi_supervised_trainer, final_eval_results = train_semi_supervised_model(
        model, pseudo_labeled_dataset, labeled_train_dataset, labeled_val_dataset, 
        compute_metrics, output_dir=f"{model_output_dir}/semi_supervised",
        combined_training=True
    )
    
    # Avaliar modelo final
    eval_dataloader = DataLoader(
        labeled_val_dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Avaliação detalhada com métricas adicionais
    detailed_metrics = evaluate_model(model, eval_dataloader, device, is_generative=is_generative)
    
    # Registrar resultados
    results = {
        "model": model_name,
        "accuracy": detailed_metrics["accuracy"],
        "f1": detailed_metrics["f1"],
        "precision": detailed_metrics["precision"],
        "recall": detailed_metrics["recall"]
    }
    
    print(f"\nResultados para {model_name}:")
    print(f"Acurácia: {results['accuracy']:.4f}")
    print(f"F1-score: {results['f1']:.4f}")
    print(f"Precisão: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    
    return results

def main():
    """Função principal que executa a comparação de modelos."""
    # Configurar sementes para reprodutibilidade
    set_seed(SEED)
    
    # Determinar dispositivo (GPU ou CPU)
    device = get_device()
    print(f"Usando dispositivo: {device}")
    
    # Carregar o dataset
    print("Carregando dataset...")
    dataset = load_symptom_diagnosis_dataset(max_samples=MAX_SAMPLES)
    print(f"Dataset carregado com {len(dataset['train'])} amostras de treino")
    
    # Preparar os rótulos (diagnósticos)
    label_encoder, num_labels = prepare_labels(dataset)
    print(f"Número de diagnósticos possíveis: {num_labels}")
    
    # Verificar quais modelos existem no mapeamento
    valid_models = [m for m in MODELS_TO_COMPARE if m in MODEL_MAPPINGS]
    if not valid_models:
        print(f"Nenhum modelo válido encontrado. Modelos disponíveis: {list(MODEL_MAPPINGS.keys())}")
        return
    
    print(f"Comparando os seguintes modelos: {valid_models}")
    
    # Treinar e avaliar cada modelo
    all_results = []
    for model_name in valid_models:
        model_result = train_and_evaluate_model(
            model_name, dataset, label_encoder, num_labels, 
            device, LABELED_RATIO, OUTPUT_DIR, SEED
        )
        all_results.append(model_result)
    
    # Compilar e exibir resultados comparativos
    results_df = pd.DataFrame(all_results)
    
    # Salvar resultados em CSV
    results_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nResultados salvos em: {results_path}")
    
    # Exibir tabela de resultados
    table = PrettyTable()
    table.field_names = ["Modelo", "Acurácia", "F1-Score", "Precisão", "Recall"]
    
    for _, row in results_df.iterrows():
        table.add_row([
            row["model"],
            f"{row['accuracy']:.4f}",
            f"{row['f1']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}"
        ])
    
    print("\nResultados comparativos:")
    print(table)
    
    # Identificar o melhor modelo
    best_model_idx = results_df["f1"].idxmax()
    best_model = results_df.iloc[best_model_idx]["model"]
    print(f"\nO melhor modelo baseado em F1-Score é: {best_model}")
    
    print("\nComparação de modelos usando aprendizado semi-supervisionado concluída!")

if __name__ == "__main__":
    main() 