import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import extract_frames
from model_training import train_random_forest, train_lstm, train_knn
from sign_recognition import recognize_sign
from feature_extraction import extract_features_from_directory
from import_from_csv import import_from_csv


def extrair_dataset_completo(videos, pasta_destino_frames):
    if not os.path.exists(videos):
        print(f"Aviso: A pasta '{videos}' nao foi encontrada.")
        return
    for gesto_label in os.listdir(videos):
        caminho_gesto = os.path.join(videos, gesto_label)
        if not os.path.isdir(caminho_gesto):
            continue
        print(f"\nProcessando videos do gesto: {gesto_label}...")
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
    print("\n--- EXTRACAO AUTOMATICA DE VIDEOS ---")
    print("\n[Etapa 1/2] Extraindo pasta de TREINAMENTO...")
    extrair_dataset_completo("videos/treino", "dataset/frames_treino")
    print("\n[Etapa 2/2] Extraindo pasta de TESTE/VALIDACAO...")
    extrair_dataset_completo("videos/teste", "dataset/frames_teste")
    print("\nProcesso de extracao em lote finalizado!")


def _perguntar_augmentation():
    """Pergunta ao usuario se quer augmentation e quantas variacoes."""
    usar = input("\nUsar data augmentation nos landmarks? (s/n) [recomendado com poucos videos]: ").strip().lower()
    if usar == 's':
        try:
            n = int(input("Quantas variacoes por amostra? [padrao: 5, recomendado: 5-10]: ").strip() or "5")
        except ValueError:
            n = 5
        print(f"Augmentation ativada: {n} variacoes por amostra (~{n+1}x mais dados)")
        return True, n
    return False, 0


def executar_treinamento():
    dataset_root = "dataset/frames_treino"

    print("\n--- INICIANDO PROCESSO DE TREINAMENTO ---")
    print("1. Escolha a arquitetura do modelo:")
    print("[1] Random Forest (estatico, frame-a-frame)")
    print("[2] LSTM (dinamico, sequencias temporais)")
    print("[3] KNN (estatico, frame-a-frame)")

    tipo_modelo = input("\nEscolha (1, 2 ou 3): ").strip()
    if tipo_modelo not in ['1', '2', '3']:
        print("Opcao invalida. Cancelando.")
        return

    modo_extracao = "rf" if tipo_modelo in ['1', '3'] else "lstm"
    augmentar, n_aumentos = _perguntar_augmentation()

    print(f"\n2. Extraindo features no modo {modo_extracao.upper()}...")
    features, labels = extract_features_from_directory(
        dataset_root,
        mode=modo_extracao,
        augmentar=augmentar,
        n_aumentos=n_aumentos
    )

    print("\n3. Iniciando Treinamento...")
    if len(features) < 2:
        print("Erro: dados insuficientes para treino.")
        return

    if tipo_modelo == '1':
        train_random_forest(features, labels)
    elif tipo_modelo == '2':
        train_lstm(features, labels)
    elif tipo_modelo == '3':
        train_knn(features, labels)


def comparar_pipelines():
    dataset_root = "dataset/frames_treino"

    print("\n--- COMPARACAO DE PIPELINES ---")
    print("[1] Random Forest  [2] LSTM  [3] KNN")
    tipo_modelo = input("\nEscolha o modelo (1, 2 ou 3): ").strip()

    if tipo_modelo not in ['1', '2', '3']:
        print("Opcao invalida.")
        return

    modo_extracao = "rf" if tipo_modelo in ['1', '3'] else "lstm"
    augmentar, n_aumentos = _perguntar_augmentation()

    print(f"\n1. Extraindo features e gerando CSV no modo {modo_extracao.upper()}...")
    features_direto, labels_direto = extract_features_from_directory(
        dataset_root,
        mode=modo_extracao,
        export_dataframe=True,
        augmentar=augmentar,
        n_aumentos=n_aumentos
    )

    if len(features_direto) < 2:
        print("Erro: dados insuficientes.")
        return

    nome_padrao = f"dataset/dataset_completo_{modo_extracao}.csv"
    csv_path = input(f"\nConfirme o caminho do CSV [{nome_padrao}]: ").strip() or nome_padrao

    print("\n2. Treinando pipeline direto (RAM)...")
    if tipo_modelo == '1':
        acc_direto = train_random_forest(features_direto, labels_direto, model_path="models/model_direto.pkl", return_accuracy=True)
    elif tipo_modelo == '2':
        acc_direto = train_lstm(features_direto, labels_direto, model_path="models/lstm_direto.h5", encoder_path="models/encoder_direto.pkl", return_accuracy=True)
    elif tipo_modelo == '3':
        acc_direto = train_knn(features_direto, labels_direto, model_path="models/knn_direto.pkl", return_accuracy=True)

    print("\n3. Lendo dados do CSV...")
    if not os.path.exists(csv_path):
        print(f"ERRO: {csv_path} nao encontrado!")
        return
    X_csv, y_csv = import_from_csv(csv_path, mode=modo_extracao)

    print("\n4. Treinando pipeline via CSV...")
    if tipo_modelo == '1':
        acc_csv = train_random_forest(X_csv, y_csv, model_path="models/model_csv.pkl", return_accuracy=True)
    elif tipo_modelo == '2':
        acc_csv = train_lstm(X_csv, y_csv, model_path="models/lstm_csv.h5", encoder_path="models/encoder_csv.pkl", return_accuracy=True)
    elif tipo_modelo == '3':
        acc_csv = train_knn(X_csv, y_csv, model_path="models/knn_csv.pkl", return_accuracy=True)

    print("\n" + "="*30)
    print("--- RESULTADO ---")
    print(f"Acuracia Direto:   {acc_direto * 100:.2f}%")
    print(f"Acuracia via CSV:  {acc_csv * 100:.2f}%")
    print("="*30)


if __name__ == "__main__":
    while True:
        print("\n" + "="*50)
        print("   SISTEMA DE RECONHECIMENTO DE SINAIS (LIBRAS)")
        print("="*50)
        print("[1] Treinar modelo")
        print("[2] Testar via Webcam")
        print("[3] Testar via Video (pasta de teste)")
        print("[4] Comparar pipeline direto vs CSV")
        print("[5] Extrair frames de videos em lote")
        print("[0] Sair")
        print("="*50)

        escolha = input("\nEscolha (0 a 5): ").strip()

        if escolha == '1':
            executar_treinamento()

        elif escolha == '2':
            print("\n[1] Random Forest  [2] LSTM  [3] KNN")
            modelo_teste = input("Modelo (1, 2 ou 3): ").strip()
            recognize_sign(0, tipo_modelo=modelo_teste)

        elif escolha == '3':
            print("\n[1] Random Forest  [2] LSTM  [3] KNN")
            modelo_teste = input("Modelo (1, 2 ou 3): ").strip()
            pasta_teste = input("Pasta dos videos de teste [videos/teste]: ").strip() or "videos/teste"

            if os.path.exists(pasta_teste) and os.path.isdir(pasta_teste):
                videos_encontrados = []
                for root, dirs, files in os.walk(pasta_teste):
                    for file in files:
                        if file.lower().endswith(('.mp4', '.avi', '.mov')):
                            videos_encontrados.append(os.path.join(root, file))

                if not videos_encontrados:
                    print(f"Nenhum video encontrado em '{pasta_teste}'.")
                else:
                    print(f"{len(videos_encontrados)} video(s) encontrado(s).")
                    for caminho_video in videos_encontrados:
                        print(f"\n-> Analisando: {caminho_video}")
                        recognize_sign(caminho_video, tipo_modelo=modelo_teste)
            else:
                print(f"Diretorio '{pasta_teste}' nao encontrado.")

        elif escolha == '4':
            comparar_pipelines()

        elif escolha == '5':
            executar_extracao_de_frames()

        elif escolha == '0':
            print("\nEncerrando. Ate logo!")
            break
        else:
            print("\nOpcao invalida.")