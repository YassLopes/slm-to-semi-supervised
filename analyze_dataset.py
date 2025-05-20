from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Carregar o dataset
print("Carregando dataset...")
dataset = load_dataset("gretelai/symptom_to_diagnosis")
print(f"Dataset carregado com {len(dataset['train'])} amostras de treino")

# Converter para DataFrame para facilitar a análise
df = pd.DataFrame({
    'sintomas': dataset['train']['input_text'],
    'diagnostico': dataset['train']['output_text']
})

# Estatísticas básicas
print("\n=== Estatísticas Básicas ===")
print(f"Número total de amostras: {len(df)}")
print(f"Número de diagnósticos únicos: {df['diagnostico'].nunique()}")

# Distribuição dos diagnósticos
print("\n=== Distribuição dos Diagnósticos ===")
diagnosis_counts = df['diagnostico'].value_counts()
print(diagnosis_counts)

# Tamanho dos textos de sintomas
df['tamanho_sintomas'] = df['sintomas'].apply(len)
print("\n=== Estatísticas do Tamanho dos Sintomas ===")
print(df['tamanho_sintomas'].describe())

# Mostrar exemplos de cada diagnóstico
print("\n=== Exemplos de cada Diagnóstico ===")
for diagnosis in df['diagnostico'].unique():
    examples = df[df['diagnostico'] == diagnosis].sample(min(1, sum(df['diagnostico'] == diagnosis)))
    for _, row in examples.iterrows():
        print(f"\nDiagnóstico: {diagnosis}")
        print(f"Sintomas: {row['sintomas'][:200]}...")
        print("-" * 50)

# Visualização dos diagnósticos mais comuns
plt.figure(figsize=(10, 6))
top_diagnoses = diagnosis_counts.head(10)
top_diagnoses.plot(kind='bar')
plt.title('10 Diagnósticos Mais Comuns')
plt.xlabel('Diagnóstico')
plt.ylabel('Número de Ocorrências')
plt.tight_layout()
plt.savefig('top_diagnoses.png')
print("Gráfico salvo como 'top_diagnoses.png'")

# Análise das palavras mais comuns nos sintomas
print("\n=== Análise de Palavras nos Sintomas ===")
all_words = " ".join(df['sintomas']).lower().split()
word_counts = Counter(all_words)
common_words = word_counts.most_common(20)
print("Palavras mais comuns nos sintomas:")
for word, count in common_words:
    print(f"{word}: {count}")

print("\nAnálise concluída!") 