# Classificação Morfológica de Galáxias com DenseNet-121

## 1. Título e Objetivo

- **Título do Projeto:** Classificação Morfológica de Galáxias (Dataset Galaxy10 DECals) utilizando Redes Neurais Convolucionais Densas (DenseNet-121).
- **Objetivo da Aplicação:** Desenvolver um modelo de Aprendizado de Máquina de ponta a ponta capaz de classificar imagens astronômicas de galáxias em 10 classes morfológicas distintas (ex.: galáxias espirais, elípticas, em fusão, irregulares, etc.). O sistema automatiza a identificação visual de estruturas celestes, auxiliando na análise de grandes volumes de dados astronômicos.

## 2. Integrantes

Todos os integrantes participaram ativamente do desenvolvimento do projeto e da gravação do vídeo explicativo:

- **Gabriel Argôlo Julião dos Santos** - GitHub: [@argologabriel](https://github.com/argologabriel) | Matrícula: 202100011978
- **João Pedro Oliveira Freitas** - GitHub: [@JPed108](https://github.com/JPed108) | Matrícula: 202500020851
- **Davi Oliveira Machado** - GitHub: [@DaviMachad0](https://github.com/DaviMachad0) | Matrícula: 202400017693

## 3. Definição da Tarefa e Fonte dos Dados

- **Tipo da Tarefa:** **Classificação Multiclasse**. O problema foi definido como classificação porque o atributo-alvo (`ans`) é categórico, representando 10 classes discretas que definem o formato e a morfologia das galáxias (Classes de 0 a 9).
- **Atributo-Alvo:** Rótulo da galáxia (inteiro de 0 a 9, ex.: _0: Disturbed Galaxies_, _1: Merging Galaxies_, _2: Round Smooth Galaxies_, etc.).
- **Atributos Preditivos:** Matrizes de pixels das imagens coloridas das galáxias (canais RGB), redimensionadas para a resolução de $224 \times 224$ pixels.
- **Fonte dos Dados:** Dataset público **Galaxy10 DECals** (~2.5 GB em formato `.h5`), obtido diretamente do repositório científico [Zenodo](https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5).

## 4. Organização dos Arquivos

O repositório está estruturado da seguinte forma para garantir reprodutibilidade:

```text
├── README.md                                # Documentação geral do projeto (este arquivo)
└── Galaxy10_DenseNet.ipynb                  # Notebook principal com todo o pipeline (Colab)
```

## 5. Instruções para Execução no Google Colab

O notebook foi desenvolvido para rodar perfeitamente no ambiente do Google Colab sem dependências locais:

1. Clique no link para abrir o notebook no Colab: [Abrir no Google Colab](https://colab.research.google.com/drive/1MU-zhhTwxQQiqFdkSsjPL_4SJNnMTZqW?usp=sharing).
2. **Configuração de Hardware:** No menu superior do Colab, vá em `Ambiente de execução` -> `Alterar o tipo de ambiente de execução` e selecione **GPU (ex.: T4)**. O uso de GPU é obrigatório para aceleração do treinamento e inferência.
3. **Download dos Dados:** Execute as células iniciais. O código possui um script automatizado (`download_dataset()`) que verifica se o dataset existe e, caso contrário, baixa o arquivo `Galaxy10_DECals.h5` automaticamente da fonte pública.
4. **Treinamento vs. Inferência:**
    - Para treinar do zero, execute a célula do "Loop de treinamento" (tempo estimado: ~2 horas).
    - Para avaliar imediatamente, pule a célula de treino e execute a célula **"Download de modelo pré-treinado"**, que baixará os pesos otimizados diretamente do Hugging Face Hub (`JPedro108/Galaxy10_DenseNet121`).
5. Execute as células de **Inferência e Avaliação** para gerar a Matriz de Confusão e o Relatório de Classificação.

## 6. Modelos Utilizados e Metodologia

Por se tratar de um problema complexo de visão computacional (imagens), utilizamos modelos de Aprendizado Profundo (Deep Learning):

- **Modelo Principal:** **DenseNet-121** (`torchvision.models.densenet121`). Esta arquitetura foi escolhida por sua conectividade densa, onde cada camada recebe entradas de todas as camadas anteriores, mitigando o problema de desvanecimento do gradiente e extraindo padrões morfológicos altamente complexos.
- **Ajuste Fino (Fine-Tuning):** A camada classificadora original foi substituída por uma sequência customizada: `Linear(1024, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 10)` para evitar _overfitting_.
- **Pré-processamento e Balanceamento (Data Augmentation):**
    - O dataset apresentava desbalanceamento severo. Utilizamos a biblioteca `Albumentations` exclusivamente no conjunto de **Treino (80%)** para gerar amostras sintéticas das classes minoritárias através de rotações aleatórias (`RandomRotate90`), espelhamentos (`Horizontal/Vertical Flip`) e translações (`ShiftScaleRotate`), equilibrando o número de amostras por classe.
    - O conjunto de **Validação/Teste (20%)** foi mantido estritamente isolado e original para evitar vazamento de dados (_data leakage_).
- **Otimização:** Treinamento realizado com otimizador **AdamW** ($lr = 1\times 10^{-4}$), agendador `ReduceLROnPlateau`, precisão mista (AMP/GradScaler) e função de perda `CrossEntropyLoss` com **Label Smoothing (0.1)** para penalizar excesso de confiança em classes visualmente ambíguas.

## 7. Principais Resultados

O modelo foi avaliado no conjunto de validação (3.548 imagens isoladas), obtendo métricas robustas e superando amplamente um _baseline_ aleatório (~10% de acurácia):

- **Acurácia Geral (Accuracy):** **90%** (0.90)
- **F1-Score Médio (Macro Avg):** **89%** (0.89)
- **F1-Score Ponderado (Weighted Avg):** **90%** (0.90)

### Destaques por Classe (Classification Report):

|    Classe    | Nome da Morfologia          | Precisão | Revocação (Recall) | F1-Score |
| :----------: | :-------------------------- | :------: | :----------------: | :------: |
| **Classe 9** | Edge-on Galaxies with Bulge | **0.97** |      **0.96**      | **0.97** |
| **Classe 1** | Merging Galaxies            |   0.94   |        0.95        |   0.95   |
| **Classe 2** | Round Smooth Galaxies       |   0.94   |        0.96        |   0.95   |
| **Classe 3** | In-between Round Smooth     |   0.93   |        0.97        |   0.95   |
| **Classe 0** | Disturbed Galaxies          |  _0.76_  |       _0.64_       |  _0.69_  |

- **Análise Crítica:** O modelo obteve excelente desempenho na identificação de galáxias com geometria bem definida, como as elípticas lisas e espirais vistas de perfil (F1-score $\ge 0.95$). A principal limitação se encontra na **Classe 0 (Disturbed Galaxies)** com F1-score de 0.69; erros analisados na Matriz de Confusão mostram que essas galáxias sofrem confusão com galáxias espirais soltas (Classe 7) e sem barra (Classe 6), devido à sua natureza visualmente irregular e difusa no espaço celestial.

## 8. Divisão das Contribuições

- **Gabriel Argôlo:** xxxxxxxxxxxxxxxxx
- **João Pedro:** xxxxxxxxxxxxxxxxxxxxx
- **Davi Oliveira:** xxxxxxxxxxxxxxxxxx

## 9. Link do Vídeo Explicativo

O vídeo contendo a apresentação detalhada do trabalho, com a identificação e participação na explicação do código e dos conceitos de **todos** os integrantes do grupo, está disponível abaixo:

🎥 **[CLIQUE AQUI PARA ASSISTIR AO VÍDEO EXPLICATIVO NO YOUTUBE/DRIVE](https://link-do-seu-video.com)**

---

## 10. Declaração de Uso de Ferramentas de Inteligência Artificial

Declaram-se os seguintes usos de ferramentas de IA durante o desenvolvimento deste projeto, visando total transparência de acordo com as normas da disciplina:

- **Ferramenta Utilizada:** Gemini (Google).
- **Finalidade:** Auxílio na sintaxe de transformações da biblioteca `Albumentations`, documentação de docstrings no código Python e revisão gramatical/formatação Markdown deste arquivo README.
