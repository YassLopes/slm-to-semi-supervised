# Comparação de Modelos Médicos/Científicos para Classificação de Sintomas

Uma prova de conceito (PoC) comparando diferentes modelos de linguagem biomédica e científica usando aprendizado semi-supervisionado para classificação de texto no contexto médico.

## Sobre o Projeto

Este projeto compara o desempenho de cinco modelos de linguagem especializados (BioBERT, ClinicalBERT, SciBERT, BioGPTBART-base e T5) na classificação de sintomas médicos em diagnósticos, utilizando técnicas de aprendizado semi-supervisionado. Para isso, utilizamos o dataset "gretelai/symptom_to_diagnosis" do HuggingFace, que contém descrições de sintomas e seus respectivos diagnósticos.

O aprendizado semi-supervisionado é especialmente útil em cenários médicos, onde pode haver muitos casos não diagnosticados (dados não rotulados) e um número limitado de casos confirmados (dados rotulados).

## Modelos Comparados

1. **BioBERT**: Modelo BERT pré-treinado em literatura biomédica (PubMed)
2. **ClinicalBERT**: Especializado em texto clínico e registros médicos eletrônicos
3. **SciBERT**: Pré-treinado em artigos científicos de diversas áreas
4. **BioGPTBART-base**: Modelo baseado em GPT adaptado para texto biomédico
5. **T5**: Modelo Text-to-Text Transfer Transformer de propósito geral

## Abordagem

A PoC implementa a técnica de pseudo-rotulagem (pseudo-labeling) para cada modelo:

1. Treinar um modelo inicial com uma pequena porção de dados rotulados (20%)
2. Usar este modelo para gerar "pseudo-rótulos" para os dados não rotulados (80%)
3. Treinar o modelo combinando os dados rotulados originais com os dados pseudo-rotulados
4. Avaliar e comparar o desempenho de cada modelo após o treinamento semi-supervisionado

## Estrutura do Projeto

O projeto está organizado em uma estrutura modular:

```
├── main.py                    # Script principal de execução e comparação
├── requirements.txt           # Dependências do projeto
├── src/                       # Módulo principal
│   ├── __init__.py            # Inicialização do módulo
│   ├── data.py                # Processamento e carregamento de dados
│   ├── model.py               # Definição e configuração dos modelos
│   ├── trainer.py             # Treinamento semi-supervisionado
│   └── utils.py               # Funções utilitárias
└── analyze_dataset.py         # Script para análise exploratória do dataset
```

## Tecnologias

- Python 3
- PyTorch
- Transformers (HuggingFace)
- Modelos de linguagem biomédica
- Scikit-learn
- Pandas
- PrettyTable

## Como Executar

1. Instalar dependências:

```bash
pip install -r requirements.txt
```

2. Executar o script principal:

```bash
python main.py
```

### Opções de Linha de Comando

O script principal aceita vários argumentos que permitem personalizar a execução:

```bash
python main.py --max_samples 1000 --labeled_ratio 0.2 --models BioBERT ClinicalBERT --output_dir ./results
```

Argumentos disponíveis:
- `--max_samples`: Limita o número de amostras (útil para testes rápidos)
- `--labeled_ratio`: Define a proporção de dados rotulados (padrão: 0.2)
- `--models`: Lista de modelos a serem comparados (padrão: todos os cinco modelos)
- `--output_dir`: Diretório para salvar resultados (padrão: "./results")
- `--seed`: Semente para reprodutibilidade (padrão: 42)

### Execução Rápida para Teste

Para um teste rápido com um subconjunto dos dados e apenas dois modelos:

```bash
python main.py --max_samples 500 --models BioBERT ClinicalBERT
```

## Resultados Esperados

Ao executar o script, você verá:

1. Resultados detalhados para cada modelo, incluindo acurácia, F1-score, precisão e recall
2. Uma tabela comparativa mostrando o desempenho relativo dos modelos
3. Identificação do melhor modelo baseado no F1-score
4. Os resultados também são salvos em formato CSV para análise posterior

A análise mostra quais modelos de linguagem são mais adequados para a tarefa de classificação de sintomas médicos usando aprendizado semi-supervisionado.
