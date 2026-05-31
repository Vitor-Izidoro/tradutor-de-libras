# Sistema de Reconhecimento de Sinais em LIBRAS

Sistema de visão computacional e aprendizado de máquina para tradução automática da Língua Brasileira de Sinais (LIBRAS). O sistema detecta e reconhece gestos via webcam ou arquivo de vídeo, utilizando pontos de referência anatômicos das mãos para classificação.

---

## Como o sistema funciona

O pipeline completo segue quatro etapas:

```
Vídeos de exemplo
      ↓
[1] Extração de frames
      ↓
[2] Detecção de landmarks (MediaPipe)
      ↓
[3] Treinamento do modelo
      ↓
[4] Reconhecimento em novos vídeos
```

### 1. Detecção de mãos — MediaPipe Hand Landmarker

Cada frame de vídeo é processado pelo MediaPipe, que localiza **ambas as mãos** e retorna 21 pontos anatômicos (landmarks) por mão: ponta dos dedos, articulações, pulso. São **126 coordenadas por frame** no total (21 pontos × 3 eixos × 2 mãos).

Quando uma mão não está visível, o bloco correspondente é preenchido com zeros — o modelo aprende que zeros significa "mão ausente", o que é importante para gestos que usam apenas uma mão.

### 2. Normalização dos landmarks

As coordenadas brutas do MediaPipe variam com a distância da câmera e com o tamanho da mão do sinalizante. Para tornar o modelo independente desses fatores:

- **Eixos X e Y**: divididos pela largura da palma (distância entre a base do indicador e a base do mínimo). Remove a variação de distância câmera–mão.
- **Eixo Z (profundidade)**: apenas centralizado no pulso, sem escala adicional. Preserva a informação de profundidade relativa entre os dedos, essencial para gestos no eixo Z.

### 3. Três modelos de classificação

| Modelo | Tipo | Melhor para |
|---|---|---|
| **Random Forest** | Frame a frame (estático) | Gestos cujo formato da mão é suficiente para identificação |
| **LSTM** | Sequência temporal (dinâmico) | Gestos que dependem de movimento, trajetória ou velocidade |
| **KNN + K-Means** | Sequência temporal simplificada | Alternativa leve ao LSTM; baseado no trabalho de Caiafa et al. (SBrT 2023) |

### 4. Predição final

Para cada vídeo testado, o modelo faz múltiplas predições ao longo dos frames. A predição final é aquela que apareceu com maior frequência (voto majoritário).

---

## Estrutura de diretórios

```
tradutor-de-libras/
│
├── main.py                   ← ponto de entrada; menu interativo
├── feature_extraction.py     ← detecção MediaPipe e normalização
├── model_training.py         ← treinamento dos três modelos
├── sign_recognition.py       ← inferência em vídeo/webcam
├── data_preprocessing.py     ← extração de frames dos vídeos
├── landmark_augmentation.py  ← data augmentation nos landmarks
├── import_from_csv.py        ← carrega dataset do CSV para treino
│
├── hand_landmarker.task      ← modelo MediaPipe (necessário)
│
├── videos/
│   ├── treino/
│   │   ├── abelha/           ← um subdiretório por gesto
│   │   │   ├── video1.mp4
│   │   │   └── video2.mp4
│   │   └── abraco/
│   │       └── video1.mp4
│   └── teste/
│       └── abelha/
│           └── teste1.mp4
│
├── dataset/
│   ├── frames_treino/
│   │   └── abelha/
│   │       ├── v0000/        ← frames do primeiro vídeo
│   │       │   ├── frame_0.jpg
│   │       │   └── frame_1.jpg
│   │       └── v0001/        ← frames do segundo vídeo
│   ├── frames_teste/
│   ├── dataset_completo_rf.csv
│   └── dataset_completo_lstm.csv
│
└── models/
    ├── sign_model.pkl         ← Random Forest treinado
    ├── lstm_sign_model.h5     ← LSTM treinado
    ├── knn_sign_model.pkl     ← KNN treinado
    └── label_encoder.pkl      ← codificador de classes do LSTM
```

---

## Requisitos e instalação

### Versão do Python

O TensorFlow exige **Python 3.9, 3.10, 3.11 ou 3.12** (64-bit). Python 3.13+ não é suportado pelo TensorFlow.

### Instalação

```bash
# 1. Criar ambiente virtual com Python compatível (exemplo: 3.12)
py -3.12 -m venv venv

# 2. Ativar o ambiente
.\venv\Scripts\Activate.ps1          # Windows (PowerShell)
source venv/bin/activate             # Linux / Mac

# 3. Instalar dependências
pip install tensorflow scikit-learn numpy mediapipe opencv-python pandas matplotlib
```

---

## Manual do usuário — Menu principal

Execute o sistema com:

```bash
python main.py
```

O menu exibe as seguintes opções:

---

### `[1]` Treinar modelo

Treina um dos três modelos com os dados disponíveis.

**Passo a passo:**
1. Escolha o modelo: `1` Random Forest / `2` LSTM / `3` KNN
2. Escolha a origem dos dados:
   - `1` **Extrair agora** — processa as imagens em `dataset/frames_treino/` na hora (mais lento, não salva CSV)
   - `2` **Carregar CSV** — usa o arquivo gerado pela opção `[7]` (recomendado, muito mais rápido)
3. Escolha se quer **data augmentation** (`s/n`): gera variações sintéticas dos gestos (rotação, escala, espelhamento). Recomendado quando há poucos vídeos por gesto.
4. O modelo treinado é salvo automaticamente na pasta `models/`.

> **Dica:** sempre gere o CSV antes (`[7]`) e treine com a opção `2` (carregar CSV). É muito mais rápido do que reprocessar as imagens a cada treino.

---

### `[2]` Testar via Webcam

Abre a câmera e reconhece gestos em tempo real.

**Uso:** escolha o modelo (`1`, `2` ou `3`) e realize o gesto na frente da câmera. A predição aparece na tela. Pressione `Q` para sair.

> O modelo escolhido precisa ter sido treinado antes (opção `[1]`).

---

### `[3]` Testar via vídeo (pasta de teste)

Processa todos os vídeos de uma pasta e salva as predições em arquivo de texto.

**Uso:**
1. Escolha o modelo (`1`, `2` ou `3`)
2. Informe a pasta com os vídeos de teste (padrão: `videos/teste`)
3. As predições são salvas em `predicoes_modelo_X.txt` (onde X é o número do modelo)

O resultado de cada vídeo é o gesto com maior frequência de aparição ao longo das predições frame a frame.

---

### `[4]` Comparar pipeline direto vs CSV

Ferramenta de validação técnica. Treina o mesmo modelo duas vezes — uma extraindo features diretamente das imagens e outra carregando do CSV — e compara as acurácias. Útil para verificar se o CSV está íntegro e consistente com os dados originais.

---

### `[5]` Extrair frames de vídeos em lote

Converte todos os vídeos de `videos/treino/` e `videos/teste/` em frames `.jpg` salvos em `dataset/frames_treino/` e `dataset/frames_teste/`.

**Quando usar:** sempre que adicionar novos vídeos de treinamento ou teste.

Cada vídeo é salvo numa subpasta própria (`v0000/`, `v0001/`, ...) dentro da pasta do gesto, evitando que vídeos se sobreponham.

> **Fluxo correto ao adicionar novos vídeos:**
> `[5] Extrair frames → [7] Gerar Dataset → [1] Treinar`

---

### `[6]` Gerar Matriz de Confusão

Exibe um gráfico da matriz de confusão comparando as predições do modelo com o gabarito real.

**Pré-requisito:** a opção `[3]` deve ter sido executada antes para gerar o arquivo `predicoes_modelo_X.txt`.

**Uso:**
1. Escolha o modelo (`1`, `2` ou `3`)
2. Digite a lista de rótulos reais na ordem em que os vídeos foram analisados, separados por vírgula.
   - Exemplo: `abelha, abraco, acabar, aceitar`

---

### `[7]` Gerar Dataset (salvar CSV)

Extrai os landmarks de todas as imagens em `dataset/frames_treino/` e salva o resultado em arquivo CSV. **Este passo é obrigatório antes de treinar com a opção `[1] → [2] Carregar CSV`.**

**Uso:**
1. Escolha o modo: `1` para RF/KNN (dados estáticos) ou `2` para LSTM (sequências temporais)
2. Se o CSV já existir, o sistema pergunta antes de sobrescrever

O CSV gerado contém apenas os dados originais. A **data augmentation** (geração de variações sintéticas) é aplicada automaticamente durante o treino, evitando que amostras sintéticas contaminem a avaliação do modelo.

---

### `[0]` Sair

Encerra o programa.

---

## Fluxo de trabalho recomendado

### Primeira vez (configuração completa)

```
1. Grave vídeos de cada gesto e coloque em videos/treino/<nome_do_gesto>/
   (recomendado: mínimo 5 vídeos por gesto)

2. [5] Extrair frames de vídeos em lote

3. [7] Gerar Dataset → escolha o modo (RF ou LSTM) → aguarde

4. [1] Treinar modelo → escolha modelo → [2] Carregar CSV → augmentation: s

5. [3] Testar via vídeo   OU   [2] Testar via Webcam
```

### Ao adicionar novos vídeos

```
1. Adicione os vídeos em videos/treino/<nome_do_gesto>/

2. [5] Extrair frames

3. [7] Gerar Dataset (sobrescrever: s)

4. [1] Treinar novamente
```

---

## Observações importantes

- **Aranha não está no treino:** o gesto `aranha` foi removido do conjunto de treinamento. Não inclua vídeos de `aranha` em `videos/teste/` sem antes adicioná-lo ao treino.
- **Modelos incompatíveis:** se você alterar o número de gestos ou regenerar o dataset, os modelos antigos precisam ser retreinados — não é possível usar um modelo treinado com 20 gestos para reconhecer um dataset de 21 gestos.
- **Aviso do MediaPipe** (`Failed to send to clearcut`): mensagem inofensiva de telemetria interna do MediaPipe. Não afeta o funcionamento.
