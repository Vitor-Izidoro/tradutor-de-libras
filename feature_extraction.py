import os
import mediapipe as mp

# Atalhos da API do MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def extract_features_from_directory(dataset_root_dir, model_asset_path="hand_landmarker.task"):
    """
    Varre os diretórios de imagens, extrai as coordenadas das mãos e retorna features e labels.
    Espera-se que o dataset_root_dir contenha subpastas com os nomes dos gestos.
    Ex: 
    dataset/frames/gesto_A/frame_0.jpg
    dataset/frames/gesto_B/frame_0.jpg
    """
    features = []
    labels = []

    # Configura o MediaPipe para modo IMAGEM (estática)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_asset_path),
        running_mode=VisionRunningMode.IMAGE
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        # Varre cada subpasta (que representa a classe/gesto)
        for label_name in os.listdir(dataset_root_dir):
            class_dir = os.path.join(dataset_root_dir, label_name)
            
            if not os.path.isdir(class_dir):
                continue
                
            print(f"Extraindo features do gesto: {label_name}...")
            
            for file_name in os.listdir(class_dir):
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image_path = os.path.join(class_dir, file_name)
                
                # Carrega a imagem no formato do MediaPipe
                try:
                    mp_image = mp.Image.create_from_file(image_path)
                    results = landmarker.detect(mp_image)

                    # Se encontrou uma mão, extrai as coordenadas (x e y)
                    if results.hand_landmarks:
                        for hand_landmarks in results.hand_landmarks:
                            landmarks = []
                            for lm in hand_landmarks:
                                landmarks.append(lm.x)
                                landmarks.append(lm.y)
                            
                            features.append(landmarks)
                            labels.append(label_name) # O nome da pasta vira o rótulo (label)
                except Exception as e:
                    print(f"Erro ao processar imagem {image_path}: {e}")

    print(f"Extração concluída! Total de amostras válidas: {len(features)}")
    return features, labels