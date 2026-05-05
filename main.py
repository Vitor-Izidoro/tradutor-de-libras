import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import extract_frames
from model_training import train_random_forest, train_lstm, train_knn
from sign_recognition import recognize_sign
from feature_extraction import extract_features_from_directory 
from import_from_csv import import_from_csv

def extrair_dataset_completo(pasta_videos_brutos, pasta_destino_frames):
    """Função auxiliar que varre diretórios e extrai frames automaticamente"""
    if not os.path.exists(pasta_videos_brutos):
        print(f"Aviso: A pasta '{pasta_videos_brutos}' não foi encontrada.")
        return

    for gesto_label in os.listdir(pasta_videos_brutos):
        caminho_gesto = os.path.join(pasta_videos_brutos, gesto_label)
        
        if not os.path.isdir(caminho_gesto):
            continue
            
        print(f"\nProcessando vídeos do gesto: {gesto_label}...")
        
        for nome_video in os.listdir(caminho_gesto):
            if nome_video.endswith(('.mp4', '.avi', '.mov')):
                caminho_video = os.path.join(caminho_gesto, nome_video)
                print(f" -> Extraindo: {nome_video}")
                
                extract_frames(
                    video_path=caminho_video, 
                    output_root_dir=pasta_destino_frames, 
                    gesture_label=gesto_label
                )

def executar_extracao_de_frames():
    """Lógica automatizada para preparar os dados de Treino e Teste"""
    print("\n--- EXTRAÇÃO AUTOMÁTICA DE VÍDEOS ---")
    
    print("\n[Etapa 1/2] Extraindo pasta de TREINAMENTO...")
    extrair_dataset_completo("videos/treino", "dataset/frames_treino")
    
    print("\n[Etapa 2/2] Extraindo pasta de TESTE/VALIDAÇÃO...")
    extrair_dataset_completo("videos/teste", "dataset/frames_teste")
    
    print("\nProcesso de extração em lote finalizado!")

def executar_treinamento():
    """Lógica de extração e escolha do modelo de treinamento"""
    # ATUALIZADO: Agora aponta apenas para os dados de treino
    dataset_root = "dataset/frames_treino" 
    
    print("\n--- INICIANDO PROCESSO DE TREINAMENTO ---")
    print("1. Escolha a arquitetura do modelo primeiro:")
    print("[1] Random Forest (Reconhecimento estático frame-a-frame)")
    print("[2] LSTM (Reconhecimento contínuo de movimento)")
    print("[3] KNN (Reconhecimento estático frame-a-frame)")
    
    tipo_modelo = input("\nEscolha (1, 2 ou 3): ").strip()

    if tipo_modelo not in ['1', '2', '3']:
        print("Opção inválida. Cancelando treinamento.")
        return

    modo_extracao = "rf" if tipo_modelo in ['1', '3'] else "lstm"

    print(f"\n2. Extraindo features das imagens de TREINO no modo: {modo_extracao.upper()}...")
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

    elif tipo_modelo == '3':
        if len(features) < 2:
            print("Erro: Dados insuficientes para treino do KNN.")
        else:
            train_knn(features, labels)

def comparar_pipelines():
    # ATUALIZADO: Aponta apenas para os dados de treino
    dataset_root = "dataset/frames_treino"

    print("\n--- COMPARAÇÃO DE PIPELINES ---")
    print("[1] Random Forest")
    print("[2] LSTM")
    print("[3] KNN")

    tipo_modelo = input("\nEscolha o modelo (1, 2 ou 3): ").strip()

    if tipo_modelo not in ['1', '2', '3']:
        print("Opção inválida.")
        return

    modo_extracao = "rf" if tipo_modelo in ['1', '3'] else "lstm"

    print(f"\n1. Extraindo features de TREINO direto dos frames no modo {modo_extracao.upper()}...")
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
            features_direto, labels_direto,
            model_path="models/sign_model_direto.pkl", return_accuracy=True
        )
        csv_path = input("Caminho do CSV RF [dataset/dataset_agarrar_rf.csv]: ").strip() or "dataset/dataset_agarrar_rf.csv"
    elif tipo_modelo == '2':
        acc_direto = train_lstm(
            features_direto, labels_direto,
            model_path="models/lstm_sign_model_direto.h5", encoder_path="models/label_encoder_direto.pkl", return_accuracy=True
        )
        csv_path = input("Caminho do CSV LSTM [dataset/dataset_agarrar_lstm.csv]: ").strip() or "dataset/dataset_agarrar_lstm.csv"
    elif tipo_modelo == '3':
        acc_direto = train_knn(
            features_direto, labels_direto,
            model_path="models/knn_sign_model_direto.pkl", return_accuracy=True
        )
        csv_path = input("Caminho do CSV KNN [dataset/dataset_agarrar_rf.csv]: ").strip() or "dataset/dataset_agarrar_rf.csv"

    print("\n3. Lendo dados do CSV...")
    X_csv, y_csv = import_from_csv(csv_path, mode=modo_extracao)

    if len(X_csv) < 2:
        print("Erro: dados insuficientes no pipeline CSV.")
        return

    print("\n4. Treinando pipeline via CSV...")
    if tipo_modelo == '1':
        acc_csv = train_random_forest(X_csv, y_csv, model_path="models/sign_model_csv.pkl", return_accuracy=True)
    elif tipo_modelo == '2':
        acc_csv = train_lstm(X_csv, y_csv, model_path="models/lstm_sign_model_csv.h5", encoder_path="models/label_encoder_csv.pkl", return_accuracy=True)
    elif tipo_modelo == '3':
        acc_csv = train_knn(X_csv, y_csv, model_path="models/knn_sign_model_csv.pkl", return_accuracy=True)

    print("\n--- RESULTADO DA COMPARAÇÃO ---")
    nome_modelo = "Random Forest" if tipo_modelo == '1' else ("LSTM" if tipo_modelo == '2' else "KNN")
    print(f"Modelo: {nome_modelo}")
    print(f"Acurácia pipeline direto: {acc_direto * 100:.2f}%")
    print(f"Acurácia pipeline via CSV: {acc_csv * 100:.2f}%")

    if acc_direto > acc_csv:
        print("Melhor resultado: pipeline direto")
    elif acc_csv > acc_direto:
        print("Melhor resultado: pipeline via CSV")
    else:
        print("Resultado empatado")


if __name__ == "__main__":
    
    while True:
        print("\n" + "="*50)
        print("   SISTEMA DE RECONHECIMENTO DE SINAIS   ")
        print("="*50)
        print("[1] Treinar o modelo (RF ou LSTM)")
        print("[2] Testar reconhecimento via Câmera (Webcam)")
        print("[3] Testar reconhecimento via Vídeo (Dados de Teste)")
        print("[4] Comparar pipeline direto vs pipeline via CSV")
        print("[5] Extrair frames de vídeos em lote (Treino e Teste)")
        print("[0] Sair do programa")
        print("="*50)
        
        escolha = input("\nEscolha uma opção (0 a 5): ").strip()
        
        if escolha == '1':
            executar_treinamento()
            
        # No seu main.py, atualize as Opções 2 e 3 do menu:

        elif escolha == '2':
            print("\nQual modelo deseja usar na Webcam?")
            print("[1] Random Forest  [2] LSTM  [3] KNN")
            modelo_teste = input("Escolha (1, 2 ou 3): ").strip()
            
            fonte_de_video = 0
            print("\nIniciando teste de reconhecimento na Webcam...")
            recognize_sign(fonte_de_video, tipo_modelo=modelo_teste)
            
        elif escolha == '3':
            print("\n--- TESTE VIA VÍDEO ---")
            print("Qual modelo deseja usar para o teste?")
            print("[1] Random Forest  [2] LSTM  [3] KNN")
            modelo_teste = input("Escolha (1, 2 ou 3): ").strip()
            
            print("\nDica: Use um vídeo da sua nova pasta de testes.")
            fonte_de_video = input("Digite o caminho do vídeo (ex: videos/teste/agarrar/video_teste.mp4): ").strip()
            
            if os.path.exists(fonte_de_video):
                print(f"Analisando vídeo '{fonte_de_video}' com o modelo {modelo_teste}...")
                recognize_sign(fonte_de_video, tipo_modelo=modelo_teste)
            else:
                print(f"Erro: O arquivo '{fonte_de_video}' não foi encontrado.")
        elif escolha == '4':
            comparar_pipelines()

        elif escolha == '5':
            executar_extracao_de_frames()
            
        elif escolha == '0':
            print("\nEncerrando o sistema. Até logo!")
            break
            
        else:
            print("\nErro: Opção inválida.")