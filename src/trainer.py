import torch
from torch.utils.data import TensorDataset, ConcatDataset
from transformers import TrainingArguments, Trainer
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

@dataclass
class DataCollatorForTextClassification:
    """
    Collator personalizado para classificação de textos que lida com nossos TensorDatasets.
    """
    def __call__(self, features):
        """
        Agrupa elementos em um batch.
        
        Args:
            features: Lista de elementos do dataset
            
        Returns:
            Batch de dados formatado corretamente
        """
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
    """
    Configura os argumentos de treinamento.
    
    Args:
        output_dir: Diretório para salvar resultados
        num_train_epochs: Número de épocas de treinamento
        batch_size: Tamanho do batch para treinamento
        eval_batch_size: Tamanho do batch para avaliação
        
    Returns:
        training_args: Argumentos de treinamento configurados
    """
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

def train_supervised_model(model, tokenizer, labeled_train_dataset, labeled_val_dataset, compute_metrics, output_dir="./results_supervised"):
    """
    Treina o modelo usando apenas dados rotulados (aprendizado supervisionado).
    
    Args:
        model: Modelo a ser treinado
        tokenizer: Tokenizer usado
        labeled_train_dataset: Dataset de treino rotulado
        labeled_val_dataset: Dataset de validação
        compute_metrics: Função para calcular métricas
        output_dir: Diretório para salvar resultados
        
    Returns:
        trainer: Objeto Trainer treinado
        eval_results: Resultados da avaliação
    """
    training_args = get_training_args(output_dir)
    data_collator = DataCollatorForTextClassification()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=labeled_train_dataset,
        eval_dataset=labeled_val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,  # Usar o collator personalizado
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    
    return trainer, eval_results

def train_semi_supervised_model(model, pseudo_labeled_dataset, labeled_train_dataset, labeled_val_dataset, compute_metrics, output_dir="./results_semi_supervised", combined_training=True):
    """
    Treina o modelo usando dados rotulados e pseudo-rotulados (aprendizado semi-supervisionado).
    
    Args:
        model: Modelo pré-treinado com dados rotulados
        pseudo_labeled_dataset: Dataset com pseudo-rótulos (pode ser None se combined_training=False)
        labeled_train_dataset: Dataset de treino rotulado
        labeled_val_dataset: Dataset de validação
        compute_metrics: Função para calcular métricas
        output_dir: Diretório para salvar resultados
        combined_training: Se True, treina com dados combinados; se False, só usa dados rotulados
        
    Returns:
        trainer: Objeto Trainer treinado
        eval_results: Resultados da avaliação
    """
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

def create_pseudo_labeled_dataset(unlabeled_tokenized, pseudo_labels):
    """
    Cria um dataset com pseudo-rótulos.
    
    Args:
        unlabeled_tokenized: Tensores tokenizados de dados não rotulados
        pseudo_labels: Lista de pseudo-rótulos gerados pelo modelo
        
    Returns:
        pseudo_labeled_dataset: TensorDataset com dados pseudo-rotulados
    """
    return TensorDataset(
        unlabeled_tokenized["input_ids"],
        unlabeled_tokenized["attention_mask"],
        torch.tensor(pseudo_labels, dtype=torch.long)  # Garantir que os rótulos sejam do tipo long
    ) 