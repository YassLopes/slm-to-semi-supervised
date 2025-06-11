#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para comparação de modelos de linguagem médica/científica 
usando aprendizado supervisionado para classificação de sintomas em diagnósticos.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os
import pandas as pd
from prettytable import PrettyTable
import json

from src.data import (
    load_symptom_diagnosis_dataset,
    prepare_labels,
    create_supervised_dataset,
    split_train_val_test
)
from src.model import (
    get_tokenizer,
    get_model,
    compute_metrics,
    evaluate_model,
    MODEL_MAPPINGS
)
from src.trainer import (
    train_supervised_model
)
from src.utils import (
    set_seed,
    get_device,
    print_dataset_examples,
    print_training_summary,
    format_metrics
)

def save_model_results(model_name, results, output_dir):
    """
    Salva os resultados de um modelo específico em arquivo JSON.
    
    Args:
        model_name: Nome do modelo
        results: Dicionário com as métricas do modelo
        output_dir: Diretório de saída
    """
    model_output_dir = f"{output_dir}/{model_name}"
    os.makedirs(model_output_dir, exist_ok=True)
    
    results_file = os.path.join(model_output_dir, "results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Resultados salvos em: {results_file}")

def load_model_results(model_name, output_dir):
    """
    Carrega os resultados salvos de um modelo específico.
    
    Args:
        model_name: Nome do modelo
        output_dir: Diretório de saída
        
    Returns:
        results: Dicionário com as métricas do modelo ou None se não existir
    """
    results_file = os.path.join(output_dir, model_name, "results.json")
    
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"Resultados carregados para {model_name}: {results_file}")
            return results
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Erro ao carregar resultados para {model_name}: {e}")
            return None
    
    return None

def model_already_trained(model_name, output_dir, seed):
    """
    Verifica se um modelo já foi treinado com os mesmos parâmetros.
    
    Args:
        model_name: Nome do modelo
        output_dir: Diretório de saída
        seed: Semente usada
        
    Returns:
        bool: True se o modelo já foi treinado, False caso contrário
    """
    results = load_model_results(model_name, output_dir)
    
    if results is None:
        return False
    
    # Verificar se os parâmetros são os mesmos
    if 'training_params' in results:
        params = results['training_params']
        return params.get('seed') == seed
    
    # Se não há informação dos parâmetros, assumir que é um resultado antigo
    return True

def train_and_evaluate_model(model_name, train_dataset, val_dataset, test_dataset, 
                            num_labels, device, output_dir, seed, num_epochs=3, force_retrain=False):
    """
    Função para treinar e avaliar um modelo específico usando aprendizado supervisionado.
    
    Args:
        model_name: Nome do modelo a ser avaliado
        train_dataset: Dataset de treino
        val_dataset: Dataset de validação
        test_dataset: Dataset de teste
        num_labels: Número total de rótulos
        device: Dispositivo (CPU ou GPU)
        output_dir: Diretório para salvar resultados
        seed: Semente para reprodutibilidade
        num_epochs: Número de épocas de treinamento
        force_retrain: Se True, força o retreinamento mesmo se já existe
        
    Returns:
        results: Dicionário com métricas de avaliação
    """
    print(f"\n{'='*50}")
    print(f"Processando modelo: {model_name}")
    print(f"{'='*50}")
    
    # Verificar se o modelo já foi treinado
    if not force_retrain and model_already_trained(model_name, output_dir, seed):
        print(f"Modelo {model_name} já foi treinado. Carregando resultados salvos...")
        results = load_model_results(model_name, output_dir)
        if results:
            print(f"Resultados carregados para {model_name}:")
            print(f"Validação - {format_metrics(results['validation'])}")
            print(f"Teste - {format_metrics(results['test'])}")
            return results
        else:
            print(f"Erro ao carregar resultados. Retreinando {model_name}...")
    
    print(f"Treinando modelo: {model_name}")
    
    # Inicializar tokenizer e modelo
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, num_labels=num_labels)
    model.to(device)
    
    # Configurar diretório de saída
    model_output_dir = f"{output_dir}/{model_name}"
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Treinar o modelo
    trainer, eval_results, test_results = train_supervised_model(
        model, train_dataset, val_dataset, test_dataset, 
        compute_metrics, output_dir=model_output_dir, num_epochs=num_epochs
    )
    
    # Extrair métricas principais
    validation_metrics = {
        "accuracy": eval_results["eval_accuracy"],
        "f1": eval_results["eval_f1"],
        "precision": eval_results["eval_precision"],
        "recall": eval_results["eval_recall"]
    }
    
    test_metrics = {
        "accuracy": test_results["test_eval_accuracy"],
        "f1": test_results["test_eval_f1"],
        "precision": test_results["test_eval_precision"],
        "recall": test_results["test_eval_recall"]
    }
    
    # Registrar resultados com parâmetros de treinamento
    results = {
        "model": model_name,
        "validation": validation_metrics,
        "test": test_metrics,
        "training_params": {
            "seed": seed,
            "num_labels": num_labels,
            "num_epochs": num_epochs,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset)
        }
    }
    
    # Salvar resultados do modelo
    save_model_results(model_name, results, output_dir)
    
    print(f"\nResultados para {model_name}:")
    print(f"Validação - {format_metrics(validation_metrics)}")
    print(f"Teste - {format_metrics(test_metrics)}")
    
    return results

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Comparação de modelos usando aprendizado supervisionado")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Número máximo de amostras (para testes rápidos)")
    parser.add_argument("--models", nargs="+", default=["BioBERT", "ClinicalBERT", "SciBERT", "BlueBERT", "PubMedBERT"],
                        help="Lista de modelos para comparar")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Diretório de saída para resultados")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Semente para reprodutibilidade")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Número de épocas de treinamento")
    parser.add_argument("--force_retrain", action="store_true", 
                        help="Força o retreinamento de todos os modelos, mesmo se já foram treinados")
    
    return parser.parse_args()

def main():
    """
    Função principal que executa a comparação de modelos usando aprendizado supervisionado.
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
    print(f"Dataset carregado com {len(dataset['train'])} amostras")
    
    # Mostrar alguns exemplos
    print_dataset_examples(dataset)
    
    # Preparar os rótulos (diagnósticos)
    label_encoder, num_labels = prepare_labels(dataset)
    print(f"Número de diagnósticos possíveis: {num_labels}")
    
    # Verificar quais modelos existem no mapeamento
    valid_models = [m for m in args.models if m in MODEL_MAPPINGS]
    if not valid_models:
        print(f"Nenhum modelo válido encontrado. Modelos disponíveis: {list(MODEL_MAPPINGS.keys())}")
        return
    
    print(f"Comparando os seguintes modelos: {valid_models}")
    
    # Verificar quais modelos já foram treinados
    trained_models = []
    untrained_models = []
    
    for model_name in valid_models:
        if not args.force_retrain and model_already_trained(model_name, args.output_dir, args.seed):
            trained_models.append(model_name)
        else:
            untrained_models.append(model_name)
    
    if trained_models:
        print(f"\nModelos já treinados (carregando resultados): {trained_models}")
    if untrained_models:
        print(f"Modelos a serem treinados: {untrained_models}")
    if args.force_retrain:
        print("Modo força retreinamento ativado - todos os modelos serão retreinados")
    
    # Preparar dados para treinamento (apenas se houver modelos para treinar)
    if untrained_models or args.force_retrain:
        print("\nPreparando dados para treinamento...")
        
        # Criar dataset supervisionado usando um tokenizer temporário (será refeito para cada modelo)
        temp_tokenizer = get_tokenizer(valid_models[0])
        full_dataset = create_supervised_dataset(dataset, label_encoder, temp_tokenizer)
        
        # Dividir em treino, validação e teste
        train_dataset, val_dataset, test_dataset = split_train_val_test(full_dataset)
        
        print_training_summary(len(train_dataset), len(val_dataset), len(test_dataset), num_labels)
    else:
        train_dataset = val_dataset = test_dataset = None
    
    # Treinar e avaliar cada modelo
    all_results = []
    for model_name in valid_models:
        if untrained_models and model_name in untrained_models or args.force_retrain:
            # Recrear dataset com tokenizer específico do modelo
            tokenizer = get_tokenizer(model_name)
            full_dataset = create_supervised_dataset(dataset, label_encoder, tokenizer)
            train_dataset, val_dataset, test_dataset = split_train_val_test(full_dataset)
        
        model_result = train_and_evaluate_model(
            model_name, train_dataset, val_dataset, test_dataset,
            num_labels, device, args.output_dir, args.seed, 
            args.epochs, force_retrain=args.force_retrain
        )
        all_results.append(model_result)
    
    # Compilar resultados para tabela comparativa
    comparison_data = []
    for result in all_results:
        row = {
            "model": result["model"],
            "val_accuracy": result["validation"]["accuracy"],
            "val_f1": result["validation"]["f1"],
            "val_precision": result["validation"]["precision"],
            "val_recall": result["validation"]["recall"],
            "test_accuracy": result["test"]["accuracy"],
            "test_f1": result["test"]["f1"],
            "test_precision": result["test"]["precision"],
            "test_recall": result["test"]["recall"]
        }
        comparison_data.append(row)
    
    results_df = pd.DataFrame(comparison_data)
    
    # Salvar resultados em CSV
    results_path = os.path.join(args.output_dir, "model_comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResultados salvos em: {results_path}")
    
    # Exibir tabelas de resultados
    print("\n" + "="*80)
    print("RESULTADOS COMPARATIVOS")
    print("="*80)
    
    # Tabela de validação
    print("\nResultados no Conjunto de Validação:")
    val_table = PrettyTable()
    val_table.field_names = ["Modelo", "Acurácia", "F1-Score", "Precisão", "Recall"]
    
    for _, row in results_df.iterrows():
        val_table.add_row([
            row["model"],
            f"{row['val_accuracy']:.4f}",
            f"{row['val_f1']:.4f}",
            f"{row['val_precision']:.4f}",
            f"{row['val_recall']:.4f}"
        ])
    
    print(val_table)
    
    # Tabela de teste
    print("\nResultados no Conjunto de Teste:")
    test_table = PrettyTable()
    test_table.field_names = ["Modelo", "Acurácia", "F1-Score", "Precisão", "Recall"]
    
    for _, row in results_df.iterrows():
        test_table.add_row([
            row["model"],
            f"{row['test_accuracy']:.4f}",
            f"{row['test_f1']:.4f}",
            f"{row['test_precision']:.4f}",
            f"{row['test_recall']:.4f}"
        ])
    
    print(test_table)
    
    # Identificar o melhor modelo
    best_val_model_idx = results_df["val_f1"].idxmax()
    best_val_model = results_df.iloc[best_val_model_idx]["model"]
    
    best_test_model_idx = results_df["test_f1"].idxmax()
    best_test_model = results_df.iloc[best_test_model_idx]["model"]
    
    print(f"\nMelhor modelo (F1-Score Validação): {best_val_model} ({results_df.iloc[best_val_model_idx]['val_f1']:.4f})")
    print(f"Melhor modelo (F1-Score Teste): {best_test_model} ({results_df.iloc[best_test_model_idx]['test_f1']:.4f})")
    
    print("\nComparação de modelos usando aprendizado supervisionado concluída!")

if __name__ == "__main__":
    main() 