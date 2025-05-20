#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para demonstração de aprendizado semi-supervisionado para classificação de sintomas em diagnósticos.
"""

import torch
from torch.utils.data import DataLoader
import argparse

from src.data import (
    load_symptom_diagnosis_dataset,
    prepare_labels,
    tokenize_dataset,
    create_semi_supervised_split,
    split_train_val
)
from src.model import (
    get_tokenizer,
    get_model,
    compute_metrics,
    generate_pseudo_labels
)
from src.trainer import (
    train_supervised_model,
    train_semi_supervised_model,
    create_pseudo_labeled_dataset
)
from src.utils import (
    set_seed,
    get_device,
    print_dataset_examples,
    print_comparison_results
)

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Demonstração de aprendizado semi-supervisionado")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Número máximo de amostras (para testes rápidos)")
    parser.add_argument("--labeled_ratio", type=float, default=0.2, 
                        help="Fração de dados rotulados (padrão: 0.2)")
    parser.add_argument("--model_name", type=str, default="distilbert/distilbert-base-uncased", 
                        help="Nome do modelo pré-treinado")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Diretório de saída para resultados")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Semente para reprodutibilidade")
    
    return parser.parse_args()

def main():
    """
    Função principal que executa o fluxo completo de treinamento semi-supervisionado.
    """
    args = parse_args()
    
    # Configurar sementes para reprodutibilidade
    set_seed(args.seed)
    
    # Determinar dispositivo (GPU ou CPU)
    device = get_device()
    print(f"Usando dispositivo: {device}")
    
    # Carregar o dataset
    print("Carregando dataset...")
    dataset = load_symptom_diagnosis_dataset(max_samples=args.max_samples)
    print(f"Dataset carregado com {len(dataset['train'])} amostras de treino")
    
    # Mostrar alguns exemplos
    print_dataset_examples(dataset)
    
    # Preparar os rótulos (diagnósticos)
    label_encoder, num_labels = prepare_labels(dataset)
    print(f"Número de diagnósticos possíveis: {num_labels}")
    print(f"Diagnósticos: {label_encoder.classes_}")
    
    # Inicializar tokenizer e modelo
    tokenizer = get_tokenizer(args.model_name)
    
    # Tokenizar o dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Criar divisão semi-supervisionada
    labeled_dataset, unlabeled_dataset, unlabeled_indices, labeled_indices, unlabeled_tokenized = create_semi_supervised_split(
        tokenized_dataset, dataset, label_encoder, tokenizer, labeled_ratio=args.labeled_ratio
    )
    
    print(f"Usando {len(labeled_indices)} amostras rotuladas e {len(unlabeled_indices)} não-rotuladas")
    
    # Dividir dados rotulados em treino e validação
    labeled_train_dataset, labeled_val_dataset = split_train_val(labeled_dataset)
    
    print(f"Conjunto de treino rotulado: {len(labeled_train_dataset)} amostras")
    print(f"Conjunto de validação: {len(labeled_val_dataset)} amostras")
    
    # Inicializar o modelo
    model = get_model(args.model_name, num_labels=num_labels)
    model.to(device)
    
    # FASE 1: Treinamento supervisionado com dados rotulados
    print("\nTreinando modelo inicial com dados rotulados...")
    supervised_trainer, eval_results = train_supervised_model(
        model, tokenizer, labeled_train_dataset, labeled_val_dataset, 
        compute_metrics, output_dir=f"{args.output_dir}/supervised"
    )
    
    print(f"\nResultados após treinamento supervisionado:")
    print(f"Acurácia: {eval_results['eval_accuracy']:.4f}")
    print(f"F1-score: {eval_results['eval_f1']:.4f}")
    
    # FASE 2: Pseudo-rotulagem
    print("\nIniciando fase de pseudo-rotulagem...")
    
    # Criar DataLoader para dados não rotulados
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Gerar pseudo-rótulos com o modelo treinado
    pseudo_labels = generate_pseudo_labels(model, unlabeled_dataloader, device)
    
    # Criar dataset com pseudo-rótulos
    pseudo_labeled_dataset = create_pseudo_labeled_dataset(unlabeled_tokenized, pseudo_labels)
    
    # FASE 3: Treinamento semi-supervisionado
    print("\nTreinando modelo com dataset combinado (rotulado + pseudo-rotulado)...")
    semi_supervised_trainer, final_eval_results = train_semi_supervised_model(
        model, pseudo_labeled_dataset, labeled_train_dataset, labeled_val_dataset, 
        compute_metrics, output_dir=f"{args.output_dir}/semi_supervised"
    )
    
    print(f"\nResultados após treinamento semi-supervisionado:")
    print(f"Acurácia: {final_eval_results['eval_accuracy']:.4f}")
    print(f"F1-score: {final_eval_results['eval_f1']:.4f}")
    
    # Comparar resultados
    print_comparison_results(eval_results, final_eval_results)
    
    print("\nDemonstração de aprendizado semi-supervisionado concluída!")

if __name__ == "__main__":
    main() 