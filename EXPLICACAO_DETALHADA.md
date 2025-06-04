# Explicação Detalhada do Projeto de Comparação de Modelos de Linguagem para Diagnósticos Médicos

Este documento explica detalhadamente todas as partes importantes do código deste projeto, pensado para pessoas sem conhecimento prévio sobre aprendizado de máquina ou processamento de linguagem natural.

## Índice
1. [O que este projeto faz e por quê](#1-o-que-este-projeto-faz-e-por-quê)
2. [Conceitos fundamentais](#2-conceitos-fundamentais)
3. [Estrutura do projeto](#3-estrutura-do-projeto)
4. [Explicação detalhada dos componentes](#4-explicação-detalhada-dos-componentes)
5. [Fluxo de execução passo a passo](#5-fluxo-de-execução-passo-a-passo)
6. [Modelos de linguagem utilizados](#6-modelos-de-linguagem-utilizados)
7. [Perguntas frequentes](#7-perguntas-frequentes)

## 1. O que este projeto faz e por quê

### Problema que estamos resolvendo
No mundo médico, transformar descrições de sintomas em diagnósticos precisos é um desafio. Existem muitos casos onde temos descrições de sintomas, mas poucos têm diagnósticos confirmados (dados rotulados). O projeto tenta responder: "Como podemos utilizar todos os dados disponíveis, incluindo os que não têm diagnóstico confirmado, para melhorar nossa capacidade de prever diagnósticos?"

### Solução proposta
Usamos uma técnica chamada **aprendizado semi-supervisionado com pseudo-rotulagem** para aproveitar tanto os dados rotulados (com diagnóstico confirmado) quanto os não rotulados (sem diagnóstico confirmado). Também comparamos diferentes modelos de linguagem especializados em textos médicos/científicos para descobrir qual funciona melhor para esta tarefa.

### Por que isso é importante?
- **Uso eficiente de dados**: Aproveitamos todos os dados disponíveis, não apenas os rotulados
- **Economia de recursos**: Reduzimos a necessidade de ter muitos diagnósticos confirmados (que são caros e demorados de obter)
- **Identificação do melhor modelo**: Descobrimos qual modelo de linguagem é mais adequado para análise de sintomas médicos

## 2. Conceitos fundamentais

### Aprendizado de máquina
É uma forma de ensinar computadores a fazer previsões ou tomar decisões baseadas em dados, sem serem explicitamente programados para cada situação específica. O computador "aprende" padrões a partir de exemplos.

### Tipos de aprendizado de máquina
- **Supervisionado**: Aprende com exemplos rotulados (entrada → saída correta)
- **Não supervisionado**: Aprende padrões em dados sem rótulos
- **Semi-supervisionado**: Usa uma combinação de dados rotulados e não rotulados

### Aprendizado semi-supervisionado
Esta abordagem é útil quando temos:
- Poucos dados rotulados (com respostas corretas)
- Muitos dados não rotulados (sem respostas corretas)

É como aprender um idioma com um professor por algumas horas (dados rotulados) e depois praticar ouvindo muitas conversas sem tradução (dados não rotulados).

### Pseudo-rotulagem
Um método simples de aprendizado semi-supervisionado:
1. Treinar um modelo inicial com os poucos dados rotulados disponíveis
2. Usar este modelo para "adivinhar" os rótulos (diagnósticos) dos dados não rotulados
3. Tratar estas "adivinhações" como se fossem rótulos verdadeiros (pseudo-rótulos)
4. Treinar um modelo final usando tanto os dados rotulados originais quanto os dados com pseudo-rótulos

### Modelos de linguagem
São algoritmos treinados para entender e gerar texto em linguagem humana. Os modelos recentes são baseados em redes neurais profundas e podem:
- Entender o significado das palavras considerando o contexto
- Capturar nuances e relações complexas entre palavras
- Transferir conhecimento de uma tarefa para outra

## 3. Estrutura do projeto

O projeto está organizado em vários componentes, cada um com uma responsabilidade específica:

```
├── main.py                    # Script principal que coordena todo o processo
├── colab_model_comparison.py  # Versão autônoma para Google Colab
├── src/                       # Módulos principais
│   ├── data.py                # Processamento de dados
│   ├── model.py               # Definição e configuração dos modelos
│   ├── trainer.py             # Lógica de treinamento
│   └── utils.py               # Funções auxiliares
└── requirements.txt           # Dependências do projeto
```

## 4. Explicação detalhada dos componentes

### Processamento de dados (`data.py`)

#### Carregamento do dataset
```python
def load_symptom_diagnosis_dataset(max_samples=None):
    dataset = load_dataset("gretelai/symptom_to_diagnosis")
    if max_samples and len(dataset['train']) > max_samples:
        dataset['train'] = dataset['train'].select(range(max_samples))
    return dataset
```

**O que faz**: Carrega o conjunto de dados de sintomas e diagnósticos do repositório HuggingFace.

**Por quê**: Precisamos de dados reais de descrições de sintomas e seus diagnósticos correspondentes para treinar nossos modelos. O parâmetro `max_samples` permite limitar o tamanho do dataset para testes mais rápidos.

#### Preparação de rótulos
```python
def prepare_labels(dataset):
    label_encoder = LabelEncoder()
    all_labels = dataset['train']['output_text']
    label_encoder.fit(all_labels)
    num_labels = len(label_encoder.classes_)
    return label_encoder, num_labels
```

**O que faz**: Converte os diagnósticos (textos) em números.

**Por quê**: Os modelos de machine learning trabalham com números, não com texto. O `LabelEncoder` atribui um número único para cada diagnóstico possível (ex: "Pneumonia" = 0, "Diabetes" = 1, etc.).

#### Tokenização
```python
def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=128)
    return dataset.map(tokenize_function, batched=True)
```

**O que faz**: Converte o texto de sintomas em sequências numéricas que o modelo pode processar.

**Por quê**: Os modelos de linguagem não entendem palavras diretamente, mas sim representações numéricas (tokens). A tokenização quebra o texto em unidades (tokens) e os converte em números. O parâmetro `max_length=128` limita cada texto a 128 tokens para uniformidade, com `padding` completando os textos mais curtos e `truncation` cortando os mais longos.

#### Divisão semi-supervisionada
```python
def create_semi_supervised_split(tokenized_dataset, dataset, label_encoder, tokenizer, labeled_ratio=0.2):
    # Código de divisão
    return labeled_dataset, unlabeled_dataset, unlabeled_indices, labeled_indices, unlabeled_tokenized
```

**O que faz**: Divide o dataset em duas partes: uma rotulada (20% por padrão) e outra não rotulada (80%).

**Por quê**: Para simular um cenário real onde temos poucos diagnósticos confirmados (dados rotulados) e muitos casos sem diagnóstico (dados não rotulados). O parâmetro `labeled_ratio=0.2` significa que usamos apenas 20% dos dados como se tivessem rótulos, enquanto "escondemos" os rótulos dos 80% restantes (embora na realidade tenhamos esses rótulos para avaliar).

### Modelos e métricas (`model.py`)

#### Definição dos modelos
```python
MODEL_MAPPINGS = {
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "BioGPTBART-base": "microsoft/biogpt",
    "T5": "t5-base"
}
```

**O que faz**: Define quais modelos de linguagem específicos vamos comparar.

**Por quê**: Cada modelo foi pré-treinado em diferentes tipos de textos (biomédicos, clínicos, científicos), e queremos descobrir qual é o mais adequado para a tarefa de classificação de sintomas.

#### Carregamento de modelos
```python
def get_model(model_name, num_labels=22):
    if model_name in MODEL_MAPPINGS:
        model_id = MODEL_MAPPINGS[model_name]
    else:
        model_id = model_name
    
    # Versão para classificação para todos os modelos
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    return model
```

**O que faz**: Carrega um modelo de linguagem pré-treinado e o adapta para a tarefa de classificação.

**Por quê**: Utilizamos modelos pré-treinados porque eles já possuem um conhecimento geral da linguagem, obtido através do treinamento em grandes volumes de texto. O parâmetro `num_labels` define quantas classes (diagnósticos possíveis) o modelo precisa prever.

#### Métricas de avaliação
```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}
```

**O que faz**: Calcula métricas para avaliar o desempenho do modelo.

**Por quê**: Precisamos de formas objetivas de medir quão bem o modelo está fazendo previsões. A acurácia mede a porcentagem de previsões corretas, enquanto o F1-score é uma média harmônica entre precisão e recall, sendo mais robusto para classes desbalanceadas (diagnósticos raros vs. comuns).

#### Geração de pseudo-rótulos
```python
def generate_pseudo_labels(model, dataloader, device):
    model.eval()
    pseudo_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Processamento e previsão
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            pseudo_labels.extend(predictions.cpu().numpy())
    
    return pseudo_labels
```

**O que faz**: Usa o modelo treinado para gerar "palpites" (pseudo-rótulos) para os dados não rotulados.

**Por quê**: É o coração do método de pseudo-rotulagem. O modelo faz previsões para os dados que não têm diagnóstico, e usamos essas previsões como se fossem diagnósticos reais para treinar um modelo melhor. `model.eval()` coloca o modelo em modo de avaliação (não treinamento), e `torch.no_grad()` desativa o cálculo de gradientes para economizar memória.

### Treinamento (`trainer.py`)

#### Coleta de dados para treinamento
```python
@dataclass
class DataCollatorForTextClassification:
    def __call__(self, features):
        # Processamento de lotes de dados
        return batch
```

**O que faz**: Formata os dados corretamente para o treinamento, garantindo que tensores estejam no formato correto.

**Por quê**: Durante o treinamento, os dados são processados em lotes (batches). O collator garante que cada lote seja formatado corretamente, com todos os tensores tendo as dimensões certas e tipos de dados corretos.

#### Argumentos de treinamento
```python
def get_training_args(output_dir, num_train_epochs=3, batch_size=16, eval_batch_size=64):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
    )
    return training_args
```

**O que faz**: Define como o treinamento será conduzido (número de épocas, tamanho do batch, etc.).

**Por quê**: Estes parâmetros controlam o processo de treinamento. Uma `epoch` é uma passagem completa pelos dados de treinamento. O `batch_size` determina quantos exemplos são processados de uma vez. O `weight_decay` é uma forma de regularização para evitar overfitting (quando o modelo "decora" os dados em vez de aprender padrões gerais).

#### Treinamento semi-supervisionado
```python
def train_semi_supervised_model(model, pseudo_labeled_dataset, labeled_train_dataset, labeled_val_dataset, compute_metrics, output_dir="./results_semi_supervised", combined_training=True):
    # Verificar se devemos combinar ou usar apenas dados rotulados
    if combined_training and pseudo_labeled_dataset is not None:
        combined_dataset = ConcatDataset([labeled_train_dataset, pseudo_labeled_dataset])
        train_dataset = combined_dataset
    else:
        train_dataset = labeled_train_dataset
    
    # Configurar e executar treinamento
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=labeled_val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    
    return trainer, eval_results
```

**O que faz**: Treina o modelo usando tanto dados rotulados quanto pseudo-rotulados.

**Por quê**: Esta é a implementação do aprendizado semi-supervisionado. Quando `combined_training=True`, combinamos os dados originais rotulados com os dados que receberam pseudo-rótulos. O modelo aprende com todos esses dados juntos. A API `Trainer` do HuggingFace simplifica o processo de treinamento, lidando com loops de treinamento, atualização de pesos, avaliação, etc.

## 5. Fluxo de execução passo a passo

Vamos seguir a execução do código desde o início até o fim:

### 1. Configuração inicial
- Definimos parâmetros como semente aleatória (para reprodutibilidade), dispositivo (CPU/GPU), etc.
- Carregamos o dataset de sintomas e diagnósticos
- Preparamos os rótulos (transformando diagnósticos textuais em índices numéricos)

### 2. Para cada modelo a ser comparado
- **Fase de preparação**:
  - Carregamos o modelo e seu tokenizador específico
  - Tokenizamos o dataset (transformando texto em sequências numéricas)
  - Dividimos os dados em conjuntos rotulados (20%) e não rotulados (80%)
  - Dividimos os dados rotulados em treino e validação

- **Fase 1: Treinamento inicial**:
  - Treinamos o modelo usando apenas os dados rotulados
  - Esta é a fase supervisionada, onde o modelo aprende a relação entre sintomas e diagnósticos

- **Fase 2: Pseudo-rotulagem**:
  - Aplicamos o modelo inicial aos dados não rotulados
  - Geramos pseudo-rótulos (diagnósticos previstos) para esses dados
  - Criamos um novo dataset que combina os dados não rotulados com seus pseudo-rótulos

- **Fase 3: Treinamento semi-supervisionado**:
  - Combinamos os dados rotulados originais com os dados pseudo-rotulados
  - Treinamos o modelo novamente com este conjunto combinado
  - Avaliamos o desempenho final do modelo

- **Avaliação e registro**:
  - Calculamos métricas como acurácia, F1-score, precisão e recall
  - Armazenamos os resultados para comparação posterior

### 3. Comparação final
- Compilamos os resultados de todos os modelos
- Criamos uma tabela comparativa
- Identificamos o melhor modelo baseado no F1-score
- Salvamos os resultados em CSV para análise futura

## 6. Modelos de linguagem utilizados

### BioBERT
**O que é**: Uma versão do BERT pré-treinada em artigos científicos biomédicos do PubMed.

**Por que usar**: Especializado em terminologia biomédica e conceitos médicos, o que pode ser valioso para entender descrições de sintomas.

### ClinicalBERT
**O que é**: Uma versão do BERT adaptada para texto clínico, treinada em notas de médicos e registros eletrônicos de saúde.

**Por que usar**: Melhor compreensão da linguagem usada em contextos clínicos reais, incluindo abreviações e termos específicos.

### SciBERT
**O que é**: Uma versão do BERT treinada em artigos científicos de diversas áreas.

**Por que usar**: Forte compreensão de linguagem científica em geral, não apenas biomédica.

### BioGPT
**O que é**: Um modelo baseado na arquitetura GPT, mas adaptado para textos biomédicos.

**Por que usar**: Potencialmente melhor em capturar relações de causa-efeito em descrições de doenças.

### T5
**O que é**: Um modelo "Text-to-Text Transfer Transformer" de propósito geral.

**Por que usar**: Altamente versátil e eficaz em várias tarefas de linguagem, servindo como base de comparação.

## 7. Perguntas frequentes

### Por que usar aprendizado semi-supervisionado em vez de supervisionado?
No mundo real, obter diagnósticos médicos confirmados é caro e demorado. Usando aprendizado semi-supervisionado, podemos aproveitar muitos dados de sintomas não diagnosticados, que são mais abundantes e baratos de obter.

### Como sabemos que os pseudo-rótulos estão corretos?
Não sabemos com certeza! Alguns pseudo-rótulos estarão errados. No entanto, se o modelo inicial for razoavelmente bom, a maioria dos pseudo-rótulos estará correta, e o benefício de ter mais dados de treinamento geralmente supera o problema de alguns rótulos incorretos.

### Por que comparar diferentes modelos?
Cada modelo tem suas forças e fraquezas baseadas em como e em quais dados foi pré-treinado. Comparando-os na mesma tarefa, podemos descobrir qual é mais adequado para o domínio específico de diagnóstico médico baseado em descrições de sintomas.

### O que significa quando um modelo tem maior F1-score mas menor acurácia?
Isso geralmente indica que o modelo é melhor em lidar com classes desbalanceadas. Por exemplo, pode ser melhor em identificar diagnósticos raros, mesmo que isso resulte em alguns erros adicionais em diagnósticos comuns.

### Como este sistema poderia ser usado na prática?
Este sistema poderia servir como uma ferramenta de suporte à decisão para profissionais de saúde, sugerindo possíveis diagnósticos com base em descrições de sintomas. No entanto, é importante ressaltar que a decisão final sempre deve ser tomada por um médico qualificado. 