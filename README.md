# Semi-Supervised Learning para Classificação de Sintomas Médicos

Uma prova de conceito (PoC) demonstrando como implementar aprendizado semi-supervisionado para classificação de texto no contexto médico.

## Sobre o Projeto

Este projeto demonstra como usar técnicas de aprendizado semi-supervisionado para melhorar a classificação de sintomas médicos em diagnósticos. Para isso, utilizamos o dataset "gretelai/symptom_to_diagnosis" do HuggingFace, que contém descrições de sintomas e seus respectivos diagnósticos.

O aprendizado semi-supervisionado é especialmente útil em cenários médicos, onde pode haver muitos casos não diagnosticados (dados não rotulados) e um número limitado de casos confirmados (dados rotulados).

## Abordagem

A PoC implementa a técnica de pseudo-rotulagem (pseudo-labeling), um método comum em aprendizado semi-supervisionado:

1. Treinar um modelo inicial com uma pequena porção de dados rotulados (20%)
2. Usar este modelo para gerar "pseudo-rótulos" para os dados não rotulados (80%)
3. Treinar um novo modelo combinando os dados rotulados originais com os dados pseudo-rotulados
4. Comparar o desempenho dos modelos antes e depois do uso de dados pseudo-rotulados

## Estrutura do Projeto

O projeto está organizado em uma estrutura modular:

```
├── main.py                    # Script principal de execução
├── requirements.txt           # Dependências do projeto
├── src/                       # Módulo principal
│   ├── __init__.py            # Inicialização do módulo
│   ├── data.py                # Processamento e carregamento de dados
│   ├── model.py               # Definição e configuração do modelo
│   ├── trainer.py             # Treinamento supervisionado e semi-supervisionado
│   └── utils.py               # Funções utilitárias
└── analyze_dataset.py         # Script para análise exploratória do dataset
```

## Tecnologias

- Python 3
- PyTorch
- Transformers (HuggingFace)
- DistilBERT (modelo de linguagem pequeno)
- Scikit-learn

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
python main.py --max_samples 1000 --labeled_ratio 0.2 --output_dir ./results
```

Argumentos disponíveis:
- `--max_samples`: Limita o número de amostras (útil para testes rápidos)
- `--labeled_ratio`: Define a proporção de dados rotulados (padrão: 0.2)
- `--model_name`: Nome do modelo pré-treinado (padrão: "distilbert/distilbert-base-uncased")
- `--output_dir`: Diretório para salvar resultados (padrão: "./results")
- `--seed`: Semente para reprodutibilidade (padrão: 42)

### Execução Rápida para Teste

Para um teste rápido com um subconjunto dos dados:

```bash
python main.py --max_samples 500
```

## Resultados Esperados

Ao executar o script, você verá:

1. Uma comparação do desempenho entre o modelo treinado apenas com dados rotulados e o modelo treinado com dados rotulados + pseudo-rotulados
2. Métricas de acurácia e F1-score para ambos os modelos
3. A melhoria quantitativa obtida através do aprendizado semi-supervisionado

O objetivo é demonstrar que, mesmo com uma quantidade limitada de dados rotulados, podemos aproveitar dados não rotulados para melhorar o desempenho do modelo.
