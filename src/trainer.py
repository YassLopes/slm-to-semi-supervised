import torch
from torch.utils.data import TensorDataset
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

def get_training_args(output_dir, num_train_epochs=3, batch_size=16, eval_batch_size=64, 
                     learning_rate=2e-5, warmup_steps=500, logging_steps=50):
    """
    Configura os argumentos de treinamento.
    
    Args:
        output_dir: Diretório para salvar resultados
        num_train_epochs: Número de épocas de treinamento
        batch_size: Tamanho do batch para treinamento
        eval_batch_size: Tamanho do batch para avaliação
        learning_rate: Taxa de aprendizado
        warmup_steps: Passos de aquecimento
        logging_steps: Frequência de logging
        
    Returns:
        training_args: Argumentos de treinamento configurados
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_steps=500,
        eval_steps=500,
        do_eval=True,
    )
    
    return training_args

def train_supervised_model(model, train_dataset, val_dataset, test_dataset, compute_metrics, 
                          output_dir="./results_supervised", num_epochs=3):
    """
    Treina o modelo usando aprendizado supervisionado tradicional.
    
    Args:
        model: Modelo a ser treinado
        train_dataset: Dataset de treino
        val_dataset: Dataset de validação
        test_dataset: Dataset de teste
        compute_metrics: Função para calcular métricas
        output_dir: Diretório para salvar resultados
        num_epochs: Número de épocas de treinamento
        
    Returns:
        trainer: Objeto Trainer treinado
        eval_results: Resultados da avaliação no conjunto de validação
        test_results: Resultados da avaliação no conjunto de teste
    """
    training_args = get_training_args(output_dir, num_train_epochs=num_epochs)
    data_collator = DataCollatorForTextClassification()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    # Treinar o modelo
    print(f"Iniciando treinamento com {len(train_dataset)} amostras de treino...")
    trainer.train()
    
    # Avaliar no conjunto de validação
    print("Avaliando no conjunto de validação...")
    eval_results = trainer.evaluate()
    
    # Avaliar no conjunto de teste
    print("Avaliando no conjunto de teste...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    # Adicionar prefixo 'test_' aos resultados de teste para distinguir
    test_results = {f"test_{k}": v for k, v in test_results.items()}
    
    return trainer, eval_results, test_results 