import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import extract_frames
# Importamos as duas novas funções de treino
from model_training import train_random_forest, train_lstm
from sign_recognition import recognize_sign
from feature_extraction import extract_features_from_directory 

def executar_treinamento():
    """Lógica de extração e escolha do modelo de treinamento"""
    dataset_root = "dataset/frames"
    
    print("\n--- INICIANDO PROCESSO DE TREINAMENTO ---")
    print("1. Escolha a arquitetura do modelo primeiro:")
    print("[1] Random Forest (Reconhecimento estático frame-a-frame)")
    print("[2] LSTM (Reconhecimento contínuo de movimento)")
    
    tipo_modelo = input("\nEscolha (1 ou 2): ").strip()

    if tipo_modelo not in ['1', '2']:
        print("Opção inválida. Cancelando treinamento.")
        return

    # Mapeia a escolha para o 'mode' do extrator
    modo_extracao = "rf" if tipo_modelo == '1' else "lstm"

    print(f"\n2. Extraindo features das imagens no modo: {modo_extracao.upper()}...")
    # AGORA COM DADOS REAIS - Chamando o extrator com o modo selecionado
    features, labels = extract_features_from_directory(dataset_root, mode=modo_extracao)

    print("\n3. Iniciando Treinamento...")
    if tipo_modelo == '1':
        if len(features) < 2:
            print("Erro: Dados insuficientes para treino do Random Forest.")
        else:
            train_random_forest(features, labels)
            
    elif tipo_modelo == '2':
        if len(features) < 2:
            print("Erro: Dados insuficientes para treino do LSTM. Tire vídeos mais longos (min 20 frames por gesto).")
        else:
            train_lstm(features, labels)

if __name__ == "__main__":
    
    while True:
        print("\n" + "="*50)
        print("   SISTEMA DE RECONHECIMENTO DE SINAIS   ")
        print("="*50)
        print("[1] Treinar o modelo (RF ou LSTM)")
        print("[2] Testar reconhecimento via Câmera (Webcam)")
        print("[3] Testar reconhecimento via Vídeo")
        print("[0] Sair do programa")
        print("="*50)
        
        escolha = input("\nEscolha uma opção (0 a 3): ").strip()
        
        if escolha == '1':
            executar_treinamento()
            
        elif escolha == '2':
            fonte_de_video = 0
            print("\nIniciando teste de reconhecimento...")
            recognize_sign(fonte_de_video)
            
        elif escolha == '3':
            fonte_de_video = "dataset/video_teste.mp4" 
            if os.path.exists(fonte_de_video):
                recognize_sign(fonte_de_video)
            else:
                print(f"Erro: O arquivo '{fonte_de_video}' não foi encontrado.")
                
        elif escolha == '0':
            print("\nEncerrando o sistema. Até logo!")
            break
            
        else:
            print("\nErro: Opção inválida.")