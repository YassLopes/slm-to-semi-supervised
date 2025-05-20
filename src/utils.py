import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Configura sementes aleatórias para reprodutibilidade.
    
    Args:
        seed: Valor da semente
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    """
    Determina o dispositivo disponível (GPU ou CPU).
    
    Returns:
        device: Dispositivo PyTorch
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_dataset_examples(dataset, num_examples=3):
    """
    Imprime exemplos do dataset para inspeção.
    
    Args:
        dataset: Dataset a ser inspecionado
        num_examples: Número de exemplos a serem exibidos
    """
    print("\nExemplos do dataset:")
    for i in range(min(num_examples, len(dataset['train']))):
        print(f"Exemplo {i+1}:")
        print(f"Sintomas: {dataset['train'][i]['input_text']}")
        print(f"Diagnóstico: {dataset['train'][i]['output_text']}")
        print()

def print_comparison_results(eval_results, final_eval_results):
    """
    Imprime uma comparação dos resultados antes e depois do treinamento semi-supervisionado.
    
    Args:
        eval_results: Resultados da avaliação após treinamento supervisionado
        final_eval_results: Resultados da avaliação após treinamento semi-supervisionado
    """
    print("\nComparação dos resultados:")
    print(f"Acurácia (supervisionado): {eval_results['eval_accuracy']:.4f}")
    print(f"Acurácia (semi-supervisionado): {final_eval_results['eval_accuracy']:.4f}")
    print(f"Melhoria na acurácia: {final_eval_results['eval_accuracy'] - eval_results['eval_accuracy']:.4f}")
    
    print(f"F1-score (supervisionado): {eval_results['eval_f1']:.4f}")
    print(f"F1-score (semi-supervisionado): {final_eval_results['eval_f1']:.4f}")
    print(f"Melhoria no F1-score: {final_eval_results['eval_f1'] - eval_results['eval_f1']:.4f}") 