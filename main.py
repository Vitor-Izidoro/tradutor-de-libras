import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import extract_frames
from model_training import train_random_forest, train_lstm, train_knn
from sign_recognition import recognize_sign
from feature_extraction import extract_features_from_directory 
from import_from_csv import import_from_csv
from matriz_confusao import avaliar_rf_knn, avaliar_lstm


def extrair_dataset_completo(videos, pasta_destino_frames):
    """Função auxiliar que varre diretórios e extrai frames automaticamente"""
    if not os.path.exists(videos):
        print(f"Aviso: A pasta '{videos}' não foi encontrada.")
        return

    for gesto_label in os.listdir(videos):
        caminho_gesto = os.path.join(videos, gesto_label)
        
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

    # 1. Extração Gerando o CSV automaticamente
    print(f"\n1. Extraindo features e GERANDO CSV no modo {modo_extracao.upper()}...")
    features_direto, labels_direto = extract_features_from_directory(
        dataset_root,
        mode=modo_extracao,
        export_dataframe=True # Garante a criação do arquivo
    )

    if len(features_direto) < 2:
        print("Erro: dados insuficientes no pipeline direto.")
        return

    # 2. Define o nome do arquivo que ACABOU de ser criado pelo passo acima
    nome_padrao = f"dataset/dataset_completo_{modo_extracao}.csv"
    
    print(f"\n2. O arquivo CSV esperado é: {nome_padrao}")
    csv_path = input(f"Confirme o caminho do CSV [{nome_padrao}]: ").strip() or nome_padrao

    # 3. Treinando pipeline direto (Memória RAM)
    print("\n3. Treinando pipeline direto (RAM)...")
    if tipo_modelo == '1':
        acc_direto = train_random_forest(features_direto, labels_direto, model_path="models/model_direto.pkl", return_accuracy=True)
    elif tipo_modelo == '2':
        acc_direto = train_lstm(features_direto, labels_direto, model_path="models/lstm_direto.h5", encoder_path="models/encoder_direto.pkl", return_accuracy=True)
    elif tipo_modelo == '3':
        acc_direto = train_knn(features_direto, labels_direto, model_path="models/knn_direto.pkl", return_accuracy=True)

    # 4. Lendo dados do CSV que foi gerado no Passo 1
    print("\n4. Lendo dados do CSV para comparação...")
    if not os.path.exists(csv_path):
        print(f"ERRO CRÍTICO: O arquivo {csv_path} não foi encontrado!")
        return

    X_csv, y_csv = import_from_csv(csv_path, mode=modo_extracao)

    # 5. Treinando pipeline via CSV
    print("\n5. Treinando pipeline via CSV...")
    if tipo_modelo == '1':
        acc_csv = train_random_forest(X_csv, y_csv, model_path="models/model_csv.pkl", return_accuracy=True)
    elif tipo_modelo == '2':
        acc_csv = train_lstm(X_csv, y_csv, model_path="models/lstm_csv.h5", encoder_path="models/encoder_csv.pkl", return_accuracy=True)
    elif tipo_modelo == '3':
        acc_csv = train_knn(X_csv, y_csv, model_path="models/knn_csv.pkl", return_accuracy=True)

    print("\n" + "="*30)
    print("--- RESULTADO DA COMPARAÇÃO ---")
    print(f"Acurácia Direto: {acc_direto * 100:.2f}%")
    print(f"Acurácia via CSV: {acc_csv * 100:.2f}%")
    print("="*30)

def executar_matriz_confusao():
    print("\n--- MATRIZ DE CONFUSÃO ---")
    print("[1] Random Forest")
    print("[2] LSTM")
    print("[3] KNN")
    print("[4] Gerar todas")

    escolha = input("\nEscolha uma opção (1 a 4): ").strip()

    if escolha == "1":
        avaliar_rf_knn("Random Forest", "models/sign_model.pkl")
    elif escolha == "2":
        avaliar_lstm()
    elif escolha == "3":
        avaliar_rf_knn("KNN", "models/knn_sign_model.pkl")
    elif escolha == "4":
        avaliar_rf_knn("Random Forest", "models/sign_model.pkl")
        avaliar_rf_knn("KNN", "models/knn_sign_model.pkl")
        avaliar_lstm()
    else:
        print("Opção inválida.")



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
        print("[6] Gerar matriz de confusão")
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
            print("\n--- TESTE VIA VÍDEO (EM LOTE) ---")
            print("Qual modelo deseja usar para o teste?")
            print("[1] Random Forest  [2] LSTM  [3] KNN")
            modelo_teste = input("Escolha (1, 2 ou 3): ").strip()
            
            print("\nDica: Pressione ENTER para usar a pasta padrão (videos/teste).")
            pasta_teste = input("Digite o caminho da pasta (ex: videos/teste): ").strip()
            
            # Se o usuário apenas apertar Enter, assume a pasta padrão
            if not pasta_teste:
                pasta_teste = "videos/teste"
            
            # Verifica se o caminho existe e é um diretório
            if os.path.exists(pasta_teste) and os.path.isdir(pasta_teste):
                print(f"\nBuscando vídeos no diretório: {pasta_teste}")
                videos_encontrados = []
                
                # os.walk percorre a pasta principal e todas as subpastas
                for root, dirs, files in os.walk(pasta_teste):
                    for file in files:
                        if file.lower().endswith(('.mp4', '.avi', '.mov')):
                            videos_encontrados.append(os.path.join(root, file))
                
                if not videos_encontrados:
                    print(f"Aviso: Nenhum vídeo foi encontrado dentro de '{pasta_teste}'.")
                else:
                    print(f"Total de {len(videos_encontrados)} vídeo(s) encontrado(s). Iniciando testes...\n")
                    
                    # Itera sobre cada vídeo encontrado na pasta
                    for caminho_video in videos_encontrados:
                        print(f"\n" + "-"*40)
                        print(f"-> Analisando vídeo: {caminho_video}")
                        
                        # Chama a sua função de reconhecimento para o vídeo atual
                        recognize_sign(caminho_video, tipo_modelo=modelo_teste)
            else:
                print(f"Erro: O diretório '{pasta_teste}' não foi encontrado ou não é uma pasta válida.")
        elif escolha == '4':
            comparar_pipelines()

        elif escolha == '5':
            executar_extracao_de_frames()
        
        elif escolha == '6':
            executar_matriz_confusao()
            
        elif escolha == '0':
            print("\nEncerrando o sistema. Até logo!")
            break
            
        else:
            print("\nErro: Opção inválida.")