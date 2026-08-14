#criador -> vitor izidoro
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm  # <--- Nova importação para a barra de progresso

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K


def train_random_forest(features, labels, model_path="models/sign_model.pkl", return_accuracy=False):
    """
    Treina um modelo Random Forest usando Leave-One-Out Cross-Validation.
    """
    X = np.array(features)
    y = np.array(labels)
    loo = LeaveOneOut()
    
    y_true = []
    y_pred = []

    print("Iniciando validação Leave-One-Out para Random Forest...")
    
    # Adicionando o tqdm aqui
    for train_index, test_index in tqdm(loo.split(X), total=len(X), desc="Avaliando Random Forest"):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        y_pred.append(pred[0])
        y_true.append(y_test[0])

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAcurácia do Random Forest (LOOCV): {accuracy * 100:.2f}%")

    # Treinamento final com 100% dos dados para salvar em produção
    print("Treinando modelo final com todos os dados...")
    final_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"Modelo Random Forest final salvo em {model_path}")

    if return_accuracy:
        return accuracy


def train_knn(features, labels, model_path="models/knn_sign_model.pkl", return_accuracy=False):
    """
    Treina um modelo KNN usando Leave-One-Out Cross-Validation.
    """
    X = np.array(features)
    y = np.array(labels)
    loo = LeaveOneOut()
    
    y_true = []
    y_pred = []

    num_classes = len(set(labels))
    k = max(1, min(3, (len(X) - 1) // num_classes))
    print(f"KNN usando k={k}. Iniciando validação Leave-One-Out...")
#criador -> vitor izidoro
    # Adicionando o tqdm aqui
    for train_index, test_index in tqdm(loo.split(X), total=len(X), desc="Avaliando KNN"):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        model.fit(X_train_scaled, y_train)
        
        pred = model.predict(X_test_scaled)
        y_pred.append(pred[0])
        y_true.append(y_test[0])

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAcurácia do KNN (LOOCV): {accuracy * 100:.2f}%")

    # Treinamento final com 100% dos dados
    print("Treinando modelo final com todos os dados...")
    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X)
    
    final_model = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    final_model.fit(X_full_scaled, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({'model': final_model, 'scaler': final_scaler}, f)
    print(f"Modelo KNN (+ scaler) final salvo em {model_path}")

    if return_accuracy:
        return accuracy


def build_lstm_model(input_shape, num_classes):
    """Função auxiliar para reconstruir o modelo LSTM a cada iteração do LOOCV."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, recurrent_dropout=0.1),
        Dropout(0.3),
        LSTM(128, return_sequences=False, recurrent_dropout=0.1),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def train_lstm(
    features,
    labels,
    model_path="models/lstm_sign_model.h5",
    encoder_path="models/label_encoder.pkl",
    return_accuracy=False
):
    """
    Treina um modelo LSTM usando Leave-One-Out Cross-Validation.
    """
    X = np.array(features)
    y = np.array(labels)

    if X.ndim != 3:
        raise ValueError(f"LSTM requer dados 3D. Formato recebido: {X.shape}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    num_classes = y_categorical.shape[1]

    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    batch_size = min(8, len(X) - 1)
    
    print(f"Atenção: LOOCV em Redes Neurais pode demorar. Validando {len(X)} iterações...")
    
    # Adicionando o tqdm aqui
    for train_index, test_index in tqdm(loo.split(X), total=len(X), desc="Avaliando LSTM", unit="amostra"):
        K.clear_session()
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_categorical[train_index], y_categorical[test_index]

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]), num_classes)
        
        # Mantenha verbose=0 no .fit() para que a saída do Keras não quebre a visualização da barra do tqdm
        model.fit(X_train, y_train, epochs=30, batch_size=batch_size, verbose=0)
        
        pred = model.predict(X_test, verbose=0)
        y_pred.append(np.argmax(pred[0]))
        y_true.append(np.argmax(y_test[0]))

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAcurácia do modelo LSTM (LOOCV): {accuracy * 100:.2f}%")

    # Treinamento final com 100% dos dados para uso real
    print("Treinando modelo LSTM final com todos os dados...")
    K.clear_session()
    final_model = build_lstm_model((X.shape[1], X.shape[2]), num_classes)
    
    final_model.fit(X, y_categorical, epochs=50, batch_size=batch_size, verbose=1)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    final_model.save(model_path)

    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"Modelo LSTM final salvo em {model_path}")
    print(f"Label Encoder salvo em {encoder_path}")

    if return_accuracy:
        return accuracy
#criador -> vitor izidoro
