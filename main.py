import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import extract_frames
# Importamos as duas novas funções de treino
from model_training import train_random_forest, train_lstm
from sign_recognition import recognize_sign
from feature_extraction import extract_features_from_directory 

from import_from_csv import import_from_csv

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
def comparar_pipelines():
    dataset_root = "dataset/frames_agarrar"

    print("\n--- COMPARAÇÃO DE PIPELINES ---")
    print("[1] Random Forest")
    print("[2] LSTM")

    tipo_modelo = input("\nEscolha o modelo (1 ou 2): ").strip()

    if tipo_modelo not in ['1', '2']:
        print("Opção inválida.")
        return

    modo_extracao = "rf" if tipo_modelo == '1' else "lstm"

    print(f"\n1. Extraindo features direto dos frames no modo {modo_extracao.upper()}...")
    features_direto, labels_direto = extract_features_from_directory(
        dataset_root,
        mode=modo_extracao
    )

    if len(features_direto) < 2:
        print("Erro: dados insuficientes no pipeline direto.")
        return

    print("\n2. Treinando pipeline direto...")
    if tipo_modelo == '1':
        acc_direto = train_random_forest(
            features_direto,
            labels_direto,
            model_path="models/sign_model_direto.pkl",
            return_accuracy=True
        )
        csv_path = input("Caminho do CSV RF [dataset/dataset_agarrar_rf.csv]: ").strip()
        if not csv_path:
            csv_path = "dataset/dataset_agarrar_rf.csv"
    else:
        acc_direto = train_lstm(
            features_direto,
            labels_direto,
            model_path="models/lstm_sign_model_direto.h5",
            encoder_path="models/label_encoder_direto.pkl",
            return_accuracy=True
        )
        csv_path = input("Caminho do CSV LSTM [dataset/dataset_agarrar_lstm.csv]: ").strip()
        if not csv_path:
            csv_path = "dataset/dataset_agarrar_lstm.csv"

    print("\n3. Lendo dados do CSV...")
    X_csv, y_csv = import_from_csv(csv_path, mode=modo_extracao)

    if len(X_csv) < 2:
        print("Erro: dados insuficientes no pipeline CSV.")
        return

    print("\n4. Treinando pipeline via CSV...")
    if tipo_modelo == '1':
        acc_csv = train_random_forest(
            X_csv,
            y_csv,
            model_path="models/sign_model_csv.pkl",
            return_accuracy=True
        )
    else:
        acc_csv = train_lstm(
            X_csv,
            y_csv,
            model_path="models/lstm_sign_model_csv.h5",
            encoder_path="models/label_encoder_csv.pkl",
            return_accuracy=True
        )

    print("\n--- RESULTADO DA COMPARAÇÃO ---")
    nome_modelo = "Random Forest" if tipo_modelo == '1' else "LSTM"
    print(f"Modelo: {nome_modelo}")
    print(f"Acurácia pipeline direto: {acc_direto * 100:.2f}%")
    print(f"Acurácia pipeline via CSV: {acc_csv * 100:.2f}%")

    if acc_direto > acc_csv:
        print("Melhor resultado: pipeline direto")
    elif acc_csv > acc_direto:
        print("Melhor resultado: pipeline via CSV")
    else:
        print("Resultado empatado")

    
    resultado_path = "results/comparacao_pipelines.csv"
    arquivo_existe = os.path.exists(resultado_path)

    with open(resultado_path, "a", encoding="utf-8") as f:
        if not arquivo_existe:
            f.write("modelo,pipeline,acuracia\n")

        f.write(f"{nome_modelo},direto,{acc_direto * 100:.2f}\n")
        f.write(f"{nome_modelo},csv,{acc_csv * 100:.2f}\n")

    print(f"Resultado salvo em {resultado_path}")

if __name__ == "__main__":
    
    while True:
        print("\n" + "="*50)
        print("   SISTEMA DE RECONHECIMENTO DE SINAIS   ")
        print("="*50)
        print("[1] Treinar o modelo (RF ou LSTM)")
        print("[2] Testar reconhecimento via Câmera (Webcam)")
        print("[3] Testar reconhecimento via Vídeo")
        print("[4] Comparar pipeline direto vs pipeline via CSV")
        print("[0] Sair do programa")
        print("="*50)
        
        escolha = input("\nEscolha uma opção (0 a 4): ").strip()
        
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
                
        elif escolha == '4':
            comparar_pipelines()
            
        elif escolha == '0':
            print("\nEncerrando o sistema. Até logo!")
            break
            
        else:
            print("\nErro: Opção inválida.")
