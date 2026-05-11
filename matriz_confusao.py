import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from feature_extraction import extract_features_from_directory


DATASET_TREINO = "dataset/frames_treino"
PASTA_SAIDA = "models/matrizes_confusao"


def avaliar_rf_knn(nome_modelo, caminho_modelo):
    
    if not os.path.exists(caminho_modelo):
        print(f"Erro: modelo não encontrado em {caminho_modelo}")
        return

    print(f"\nExtraindo features para avaliar {nome_modelo}...")
    features, labels = extract_features_from_directory(DATASET_TREINO, mode="rf")

    if len(features) < 2:
        print("Erro: dados insuficientes para gerar matriz de confusao.")
        return

    _, X_test, _, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
    )

    with open(caminho_modelo, "rb") as arquivo:
        modelo = pickle.load(arquivo)

    y_pred = modelo.predict(X_test)
    classes = sorted(set(y_test) | set(y_pred))

    gerar_matriz(nome_modelo, y_test, y_pred, classes)


def avaliar_lstm():
    
    caminho_modelo = "models/lstm_sign_model.h5"
    caminho_encoder = "models/label_encoder.pkl"

    if not os.path.exists(caminho_modelo):
        print(f"Erro: modelo LSTM nao encontrado em {caminho_modelo}")
        return

    if not os.path.exists(caminho_encoder):
        print(f"Erro: LabelEncoder nao encontrado em {caminho_encoder}")
        return

    print("\nExtraindo sequencias para avaliar LSTM...")
    features, labels = extract_features_from_directory(DATASET_TREINO, mode="lstm")

    if len(features) < 2:
        print("Erro: dados insuficientes para gerar matriz de confusao.")
        return

    _, X_test, _, y_test = train_test_split(
        np.array(features),
        np.array(labels),
        test_size=0.2,
        random_state=42,
    )

    from tensorflow.keras.models import load_model

    with open(caminho_encoder, "rb") as arquivo:
        label_encoder = pickle.load(arquivo)

    modelo = load_model(caminho_modelo)
    probabilidades = modelo.predict(X_test, verbose=0)
    indices_previstos = np.argmax(probabilidades, axis=1)
    y_pred = label_encoder.inverse_transform(indices_previstos)

    gerar_matriz("LSTM", y_test, y_pred, list(label_encoder.classes_))


def gerar_matriz(nome_modelo, y_real, y_pred, classes):
    """Mostra e salva a matriz de confusao."""
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    print(f"\nRelatorio de classificacao - {nome_modelo}")
    print(classification_report(y_real, y_pred, labels=classes, zero_division=0))

    matriz = confusion_matrix(y_real, y_pred, labels=classes)

    largura = max(8, len(classes) * 0.6)
    altura = max(6, len(classes) * 0.6)
    fig, ax = plt.subplots(figsize=(largura, altura))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=classes,
    )
    display.plot(ax=ax, cmap="Blues", xticks_rotation=90, values_format="d")

    ax.set_title(f"Matriz de Confusao - {nome_modelo}")
    ax.set_xlabel("Gesto previsto")
    ax.set_ylabel("Gesto real")
    plt.tight_layout()

    nome_arquivo = nome_modelo.lower().replace(" ", "_")
    caminho_saida = os.path.join(PASTA_SAIDA, f"matriz_{nome_arquivo}.png")
    plt.savefig(caminho_saida, dpi=200)

    print(f"Matriz salva em: {caminho_saida}")
    plt.show()


def main():
    print("=" * 50)
    print("        MATRIZ DE CONFUSAO DOS MODELOS")
    print("=" * 50)
    print("[1] Random Forest")
    print("[2] LSTM")
    print("[3] KNN")
    print("[4] Gerar todas")
    print("=" * 50)

    escolha = input("\nEscolha uma opcao (1 a 4): ").strip()

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
        print("Opcao invalida.")


if __name__ == "__main__":
    main()
