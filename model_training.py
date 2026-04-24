import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Novos imports necessários para o modelo LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def train_random_forest(features, labels, model_path="models/sign_model.pkl", return_accuracy=False):
    """
    Treina um modelo de classificação de gestos estáticos usando Random Forest.
    Espera features no formato 2D: (amostras, caracteristicas_por_frame)
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    
    print(f"Acurácia do Random Forest: {accuracy * 100:.2f}%")

    # Cria a pasta 'models' se ela não existir
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo Random Forest salvo com sucesso em {model_path}")
    
    if return_accuracy:
        return accuracy


def train_lstm(features, labels, model_path="models/lstm_sign_model.h5", encoder_path="models/label_encoder.pkl", return_accuracy=False):
    """
    Treina um modelo LSTM para séries temporais (movimento contínuo).
    Espera features no formato 3D: (amostras, quantidade_de_frames, caracteristicas_por_frame)
    """
    # Converte listas para arrays numpy para manipulação de matrizes
    X = np.array(features)
    y = np.array(labels)

    # Verificação de segurança para garantir que os dados são sequenciais (3D)
    if len(X.shape) != 3:
        raise ValueError(f"Erro: O modelo LSTM exige dados em 3 dimensões (amostras, frames, coordenadas). Formato recebido: {X.shape}. Você precisa adaptar o extrator de features para retornar sequências temporais.")

    # 1. Tratamento das labels (Ex: transforma 'agarrar', 'agora' em 0, 1 e depois em matriz binária)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded) # One-hot encoding

    # 2. Divisão de Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # 3. Construção da Arquitetura LSTM
    # X_train.shape[1] = número de frames da sequência
    # X_train.shape[2] = número de coordenadas extraídas do mediapipe
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2)) # Ajuda a evitar overfitting
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(label_encoder.classes_), activation='softmax')) # Camada de saída baseada nas categorias

    # 4. Compilação do Modelo
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    print("\nIniciando treinamento do LSTM (isso pode levar alguns minutos)...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # 5. Avaliação do Modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nAcurácia do modelo LSTM: {accuracy * 100:.2f}%")

    # 6. Salvando o Modelo e o Decodificador de Labels
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Modelos Keras/TensorFlow são salvos diretamente pelo método do próprio objeto
    model.save(model_path)
    
    # Salvamos o encoder com pickle para conseguir converter "0" de volta para "agarrar" no momento de testar
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
        
    print(f"Modelo LSTM salvo em {model_path}")
    print(f"Label Encoder salvo em {encoder_path}")
    
    if return_accuracy:
        return accuracy