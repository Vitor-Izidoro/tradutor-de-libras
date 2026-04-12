import os
import mediapipe as mp
import pandas as pd
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def extract_features_from_directory(
    dataset_root_dir,
    model_asset_path="hand_landmarker.task",
    mode="lstm",
    sequence_length=20,
    step=5,
    export_dataframe=False    
):
    """
    Varre os diretórios de imagens e extrai as coordenadas.
    - mode="rf": Retorna 2D (frame-a-frame)
    - mode="lstm": Retorna 3D (sequências temporais de tamanho 'sequence_length')
    """
    features = []
    labels = []

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_asset_path),
        running_mode=VisionRunningMode.IMAGE
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        for label_name in os.listdir(dataset_root_dir):
            class_dir = os.path.join(dataset_root_dir, label_name)
            
            if not os.path.isdir(class_dir):
                continue
                
            print(f"Extraindo features do gesto: {label_name}...")
            
            # 1. Obtém e ORDENA os arquivos numericamente (para manter a linha do tempo correta)
            file_names = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            try:
                # Ordena pegando o número depois de 'frame_' e antes de '.jpg'
                file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            except Exception:
                file_names.sort() # Fallback de segurança

            class_landmarks = [] # Guarda todos os landmarks desta pasta em ordem

            for file_name in file_names:
                image_path = os.path.join(class_dir, file_name)
                
                try:
                    mp_image = mp.Image.create_from_file(image_path)
                    results = landmarker.detect(mp_image)

                    if results.hand_landmarks:
                        # Pega a primeira mão detectada
                        hand_landmarks = results.hand_landmarks[0]
                        landmarks = []
                        for lm in hand_landmarks:
                            landmarks.append(lm.x)
                            landmarks.append(lm.y)
                        
                        if mode == "rf":
                            features.append(landmarks)
                            labels.append(label_name)
                        else:
                            class_landmarks.append(landmarks)
                    else:
                        # Se não detectou a mão e for LSTM, preenche com zeros para manter a contagem de tempo
                        if mode == "lstm":
                            class_landmarks.append([0.0] * 42)
                except Exception as e:
                    print(f"Erro ao processar imagem {image_path}: {e}")

            # 2. Se for LSTM, agrupa os frames em sequências (Janela Deslizante)
            if mode == "lstm":
                # Exemplo: Com sequence_length=20 e step=5, se tivermos 30 frames, 
                # geramos sequências do [0 ao 20], [5 ao 25], [10 ao 30].
                for i in range(0, len(class_landmarks) - sequence_length + 1, step):
                    sequence = class_landmarks[i : i + sequence_length]
                    features.append(sequence)
                    labels.append(label_name)

        if export_dataframe:
            columns = []
            if mode=='lstm': 
                # Adiciona colunas identificadoras da amostra e do frame
                # para transformacao de 3D para 2D
                print(f'Numero de labels: {len(labels)}')
                columns.append('target')
                columns.append('sample_idx')
                columns.append('frame_idx')
                flatten_features = []
                for sample_idx in range(len(features)):
                    for frame_idx in range(len(features[sample_idx])):

                        temp_vec = [labels[sample_idx], sample_idx, frame_idx]
                        temp_vec.extend(features[sample_idx][frame_idx])
                        flatten_features.append(temp_vec)

                features = flatten_features
  
            for i in range(1, 22):
                columns.append(f'x_{i}')
                columns.append(f'y_{i}')

            #Vincula a label respectiva e as coordenadas encontradas
            df = pd.DataFrame(features, columns=columns)
            if mode == 'rf':
                df['target'] = labels
                df = df[['target'] + columns]
            df.to_csv(f'./dataset/dataset_{label_name}_{mode}.csv', index=False)
            print(f'Landmarks de {label_name} no modo {mode} exportados para csv')

    print(f"Extração concluída! Total de amostras ({mode}): {len(features)}")



    return features, labels

if __name__ == "__main__":
    dataset_root = "dataset/frames"
    features, labels = extract_features_from_directory(
        dataset_root_dir=dataset_root,
        mode='lstm',
        export_dataframe=True
    )
    features, labels = extract_features_from_directory(
        dataset_root_dir=dataset_root,
        mode='rf',
        export_dataframe=True
    )