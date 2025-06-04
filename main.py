#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para comparação de modelos de linguagem médica/científica 
usando aprendizado semi-supervisionado para classificação de sintomas em diagnósticos.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os
import pandas as pd
from prettytable import PrettyTable

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
    generate_pseudo_labels,
    evaluate_model,
    MODEL_MAPPINGS
)
from src.trainer import (
    train_semi_supervised_model,
    create_pseudo_labeled_dataset
)
from src.utils import (
    set_seed,
    get_device,
    print_dataset_examples
)

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Comparação de modelos usando aprendizado semi-supervisionado")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Número máximo de amostras (para testes rápidos)")
    parser.add_argument("--labeled_ratio", type=float, default=0.2, 
                        help="Fração de dados rotulados (padrão: 0.2)")
    parser.add_argument("--models", nargs="+", default=["BioBERT", "ClinicalBERT", "SciBERT", "BioGPTBART-base", "T5"],
                        help="Lista de modelos para comparar")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Diretório de saída para resultados")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Semente para reprodutibilidade")
    
    return parser.parse_args()

def train_and_evaluate_model(model_name, dataset, label_encoder, num_labels, labeled_indices, unlabeled_indices, 
                            device, labeled_ratio, output_dir, seed):
    """
    Função para treinar e avaliar um modelo específico.
    
    Args:
        model_name: Nome do modelo a ser avaliado
        dataset: Dataset completo
        label_encoder: Encoder para os rótulos
        num_labels: Número total de rótulos
        labeled_indices: Índices dos dados rotulados
        unlabeled_indices: Índices dos dados não rotulados
        device: Dispositivo (CPU ou GPU)
        labeled_ratio: Proporção de dados rotulados
        output_dir: Diretório para salvar resultados
        seed: Semente para reprodutibilidade
        
    Returns:
        results: Dicionário com métricas de avaliação
    """
    print(f"\n{'='*50}")
    print(f"Treinando e avaliando modelo: {model_name}")
    print(f"{'='*50}")
    
    # Inicializar tokenizer e modelo
    tokenizer = get_tokenizer(model_name)
    
    # Tokenizar o dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Criar divisão semi-supervisionada
    labeled_dataset, unlabeled_dataset, _, _, unlabeled_tokenized = create_semi_supervised_split(
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
    # Esta é uma simplificação - modelos como T5 precisam de processamento especial
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
    """
    Função principal que executa a comparação de modelos usando aprendizado semi-supervisionado.
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
    
    # Criar índices para divisão consistente entre modelos
    train_indices = list(range(len(dataset['train'])))
    num_labeled = int(args.labeled_ratio * len(train_indices))
    labeled_indices = train_indices[:num_labeled]
    unlabeled_indices = train_indices[num_labeled:]
    print(f"Usando {len(labeled_indices)} amostras rotuladas e {len(unlabeled_indices)} não-rotuladas")
    
    # Verificar quais modelos existem no mapeamento
    valid_models = [m for m in args.models if m in MODEL_MAPPINGS]
    if not valid_models:
        print(f"Nenhum modelo válido encontrado. Modelos disponíveis: {list(MODEL_MAPPINGS.keys())}")
        return
    
    print(f"Comparando os seguintes modelos: {valid_models}")
    
    # Treinar e avaliar cada modelo
    all_results = []
    for model_name in valid_models:
        model_result = train_and_evaluate_model(
            model_name, dataset, label_encoder, num_labels, 
            labeled_indices, unlabeled_indices, device, 
            args.labeled_ratio, args.output_dir, args.seed
        )
        all_results.append(model_result)
    
    # Compilar e exibir resultados comparativos
    results_df = pd.DataFrame(all_results)
    
    # Salvar resultados em CSV
    results_path = os.path.join(args.output_dir, "model_comparison_results.csv")
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