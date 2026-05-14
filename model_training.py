import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


def train_random_forest(features, labels, model_path="models/sign_model.pkl", return_accuracy=False):
    """
    Treina um modelo Random Forest para gestos estáticos (frame-a-frame).
    Espera features no formato 2D: (amostras, features_por_frame)
    """
    # CORREÇÃO: stratify garante que todas as classes apareçam no treino e no teste.
    # Com poucos dados, sem isso, uma classe pode ficar só no teste e o modelo nunca a aprende.
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # CORREÇÃO: n_estimators=200 e class_weight='balanced' ajudam com datasets pequenos e desbalanceados.
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Acurácia do Random Forest: {accuracy * 100:.2f}%")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo Random Forest salvo em {model_path}")

    if return_accuracy:
        return accuracy


def train_knn(features, labels, model_path="models/knn_sign_model.pkl", return_accuracy=False):
    """
    Treina um modelo KNN para gestos estáticos (frame-a-frame).
    Espera features no formato 2D: (amostras, features_por_frame)
    """
    # CORREÇÃO: stratify para garantir representação balanceada no split.
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # CORREÇÃO: KNN é sensível à escala das features. O StandardScaler garante que
    # coordenadas com valores maiores não dominem o cálculo de distância.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # CORREÇÃO: n_neighbors=3 é mais adequado com datasets pequenos.
    # Com poucos dados por classe, k=5 pode consultar vizinhos da classe errada.
    # Usamos pesos por distância: vizinhos mais próximos têm mais influência.
    num_classes = len(set(labels))
    k = max(1, min(3, len(X_train) // num_classes))
    print(f"KNN usando k={k} (ajustado ao tamanho do dataset)")

    model = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    print(f"Acurácia do KNN: {accuracy * 100:.2f}%")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # CORREÇÃO: salva o scaler junto com o modelo, pois ele é necessário na inferência.
    with open(model_path, "wb") as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print(f"Modelo KNN (+ scaler) salvo em {model_path}")

    if return_accuracy:
        return accuracy


def train_lstm(
    features,
    labels,
    model_path="models/lstm_sign_model.h5",
    encoder_path="models/label_encoder.pkl",
    return_accuracy=False
):
    """
    Treina um modelo LSTM para reconhecimento de gestos dinâmicos (sequências temporais).
    Espera features no formato 3D: (amostras, frames, features_por_frame)
    """
    X = np.array(features)
    y = np.array(labels)

    if X.ndim != 3:
        raise ValueError(
            f"LSTM requer dados 3D (amostras, frames, coordenadas). "
            f"Formato recebido: {X.shape}"
        )

    num_amostras = X.shape[0]
    num_classes = len(set(labels))

    # CORREÇÃO: com poucos dados, um test_size fixo de 0.2 pode deixar menos de
    # 1 amostra por classe no teste. Garantimos pelo menos 1 por classe.
    test_size = max(num_classes / num_amostras, 0.2)
    test_size = min(test_size, 0.3)  # nunca mais que 30%

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=test_size, random_state=42, stratify=y_encoded
    )

    # CORREÇÃO: batch_size deve ser <= número de amostras de treino.
    # Com datasets pequenos, usar o total de amostras ou um valor pequeno como 8.
    batch_size = min(8, len(X_train))
    print(f"LSTM usando batch_size={batch_size} (ajustado ao tamanho do dataset)")

    # CORREÇÃO: removida a activation='relu' das camadas LSTM.
    # LSTMs usam tanh internamente — forçar relu degrada o aprendizado de sequências.
    # Adicionado recurrent_dropout para regularização dentro da célula recorrente.
    model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2]),
             recurrent_dropout=0.1),
        Dropout(0.3),
        LSTM(128, return_sequences=False,
             recurrent_dropout=0.1),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    # CORREÇÃO: EarlyStopping evita overfitting em datasets pequenos.
    # Com 50 epochs fixos e poucos dados, o modelo memoriza ao invés de aprender.
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    print("\nIniciando treinamento do LSTM...")
    model.fit(
        X_train, y_train,
        epochs=100,          # mais epochs, mas o EarlyStopping para cedo se necessário
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAcurácia do modelo LSTM: {accuracy * 100:.2f}%")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"Modelo LSTM salvo em {model_path}")
    print(f"Label Encoder salvo em {encoder_path}")

    if return_accuracy:
        return accuracy