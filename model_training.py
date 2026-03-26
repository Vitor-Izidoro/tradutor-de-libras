import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_model(features, labels, model_path="models/sign_model.pkl"):
    """
    Treina um modelo de classificação de gestos.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    
    print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

    # Cria a pasta 'models' (e qualquer outra pasta no caminho) se ela não existir
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo salvo com sucesso em {model_path}")