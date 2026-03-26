import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import extract_frames
from model_training import train_model
from sign_recognition import recognize_sign
from feature_extraction import extract_features_from_directory 

def executar_treinamento():
    """Isola a lógica de extração e treinamento para manter o menu limpo"""
    dataset_root = "dataset/frames"
    
    # Definição dos caminhos dos vídeos
    video_path_A = "dataset/Agarrar_Articulador3.mp4"
    video_path_B = "dataset/Agora_Articulador1.mp4"
    video_path_C = "dataset/Aconselhar_Articulador1.mp4"
    
    print("\n--- INICIANDO PROCESSO DE TREINAMENTO ---")
    print("1. Extraindo frames dos vídeos...")
    # ATENÇÃO: Descomente as linhas abaixo quando tiver os vídeos na pasta
    # extract_frames(video_path_A, dataset_root, gesture_label="agarrar")
    # extract_frames(video_path_B, dataset_root, gesture_label="agora")
    # extract_frames(video_path_C, dataset_root, gesture_label="aconselhar")

    print("\n2. Extraindo features (coordenadas) das imagens...")
    # Descomente a linha abaixo para realizar a extração real
    # features, labels = extract_features_from_directory(dataset_root)
    
    # Simulação para evitar que o código quebre caso as variáveis estejam comentadas acima
    features, labels = [], [] 

    # 3. Treinamento
    # if len(features) < 10:
    #     print("Erro: Poucos dados extraídos. Tire mais fotos/frames ou verifique a detecção das mãos.")
    # else:
    #     print("\n3. Iniciando treinamento do modelo...")
    #     train_model(features, labels)
    #     print("\nModelo treinado e atualizado com sucesso!")


if __name__ == "__main__":
    
    # O loop 'while True' mantém o menu rodando até você escolher sair (opção 0)
    while True:
        print("\n" + "="*50)
        print("   SISTEMA DE RECONHECIMENTO DE SINAIS   ")
        print("="*50)
        print("[1] Treinar o modelo (Extrair dados e gerar novo .pkl)")
        print("[2] Testar reconhecimento via Câmera (Webcam)")
        print("[3] Testar reconhecimento via Vídeo")
        print("[0] Sair do programa")
        print("="*50)
        
        escolha = input("\nEscolha uma opção (0 a 3): ").strip()
        
        if escolha == '1':
            executar_treinamento()
            
        elif escolha == '2':
            fonte_de_video = 0
            print("\nIniciando teste de reconhecimento em tempo real via webcam...")
            recognize_sign(fonte_de_video)
            
        elif escolha == '3':
            # Você pode alterar a string abaixo para apontar para um vídeo de teste específico
            fonte_de_video = "dataset/video_teste.mp4" 
            print(f"\nIniciando teste de reconhecimento com o vídeo: {fonte_de_video}...")
            
            # Verificação rápida para evitar erros se o vídeo não existir
            if os.path.exists(fonte_de_video):
                recognize_sign(fonte_de_video)
            else:
                print(f"Erro: O arquivo '{fonte_de_video}' não foi encontrado.")
                
        elif escolha == '0':
            print("\nEncerrando o sistema. Até logo!")
            break # Quebra o loop e finaliza o script
            
        else:
            print("\nErro: Opção inválida. Por favor, digite 0, 1, 2 ou 3.")