# Sistema Integrado de Reconhecimento de Sinais (Libras)

Este projeto consiste em um sistema de visão computacional e aprendizado de máquina voltado para a tradução da Linguagem Brasileira de Sinais (Libras). O sistema utiliza a captura de movimentos das mãos através de webcam ou arquivos de vídeo, extraindo pontos de referência anatômicos (landmarks) para a classificação dos gestos.

## Evolução do Projeto: De Estático para Dinâmico

O sistema foi originalmente concebido com uma abordagem de classificação frame-a-frame utilizando o algoritmo Random Forest, adequado para sinais estáticos ou o alfabeto manual. 

Recentemente, a arquitetura foi expandida para suportar Séries Temporais através de uma rede neural LSTM (Long Short-Term Memory). Esta atualização permite que o sistema compreenda o movimento contínuo das mãos, diferenciando sinais que dependem de trajetória, velocidade e direção (parâmetros dinâmicos).

### Alterações Implementadas:

* Modelos Duplos (model_training.py): O sistema agora suporta duas arquiteturas de treinamento distintas.
    * Random Forest (Scikit-learn): Voltado para processamento de dados 2D e sinais estáticos.
    * LSTM (TensorFlow/Keras): Voltado para processamento de dados 3D (amostras, frames, coordenadas) e sinais dinâmicos.
* Extração Temporal (feature_extraction.py): O extrator foi atualizado para agrupar frames em sequências de movimento (janelas de tempo). Inclui lógica para ordenação numérica de arquivos e preenchimento de zeros (zero-padding) quando a detecção falha, garantindo a integridade da linha do tempo.
* Interface de Usuário (main.py): O menu principal foi reestruturado para permitir a escolha entre as arquiteturas de modelo antes de iniciar o fluxo de extração e treinamento.

## Requisitos de Sistema e Instalação

### Compatibilidade de Versão do Python
O TensorFlow, utilizado para a rede LSTM, requer versões específicas do Python para garantir a compatibilidade com seus binários pré-compilados.
* Versões Suportadas: Python 3.9, 3.10, 3.11 ou 3.12 (64-bit).
* Versões Não Suportadas: Atualmente, o Python 3.13 e 3.14 apresentam erros de instalação ("No matching distribution found for tensorflow") devido à falta de binários estáveis para o núcleo em C++.

### Instruções para Instalação

1. Crie um Ambiente Virtual com a versão correta (Exemplo para Python 3.12 no Windows):
   py -3.12 -m venv venv

2. Ative o ambiente virtual:
   No Windows (PowerShell): .\venv\Scripts\Activate.ps1
   No Linux/Mac: source venv/bin/activate

3. Instale as bibliotecas necessárias:
   pip install tensorflow scikit-learn numpy mediapipe opencv-python

## Estrutura de Arquivos

* main.py: Arquivo principal que gerencia o menu e o fluxo de execução do sistema.
* data_preprocessing.py: Responsável pela extração de frames individuais a partir de vídeos brutos.
* feature_extraction.py: Realiza a detecção de mãos via MediaPipe e converte as coordenadas para o formato 2D (RF) ou 3D (LSTM).
* model_training.py: Contém as rotinas de treinamento, avaliação e exportação dos modelos (.pkl para RF e .h5 para LSTM).
* sign_recognition.py: Implementa a lógica de inferência em tempo real para reconhecimento via câmera ou vídeo.

## Guia de Utilização

1. Organize seus vídeos de treinamento no diretório dataset/ e certifique-se de que as linhas de extração na função executar_treinamento do main.py estejam ativas.
2. Inicie o sistema:
   python main.py
3. Selecione a opção [1] para realizar o treinamento. O sistema solicitará que você escolha entre Random Forest ou LSTM.
4. Após a conclusão e geração dos arquivos na pasta models/, selecione a opção [2] para testar o reconhecimento utilizando sua webcam.
