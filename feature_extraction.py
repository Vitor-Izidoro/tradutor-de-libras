#criador -> vitor izidoro
import os
import mediapipe as mp
import pandas as pd
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

NUM_FEATURES = 126  # 21 landmarks × 3 coords × 2 mãos

_N_MAO = 63               # features de uma mão
_ZEROS_MAO = [0.0] * _N_MAO  # placeholder para mão ausente


def _normalizar_mao(hand_landmarks):
    """
    Normaliza os landmarks de UMA mão (63 features).

    1. Subtrai o pulso (lm 0) em X, Y e Z → remove posição absoluta.
    2. Divide X e Y pela largura da palma em 2D (distância XY entre a base do
       indicador lm 5 e a base do mínimo lm 17) → remove variação de escala/câmera
       sem tocar no eixo Z.
    3. Z permanece apenas centrado no pulso → preserva profundidade relativa
       entre os dedos, informação crucial para gestos no eixo Z.
    """
    pulso = hand_landmarks[0]
    base_ind = hand_landmarks[5]
    base_min = hand_landmarks[17]
#criador -> vitor izidoro
    escala_xy = (
        (base_ind.x - base_min.x) ** 2 +
        (base_ind.y - base_min.y) ** 2
    ) ** 0.5

    if escala_xy < 1e-6:
        escala_xy = 1.0

    landmarks = []
    for lm in hand_landmarks:
        landmarks.append((lm.x - pulso.x) / escala_xy)
        landmarks.append((lm.y - pulso.y) / escala_xy)
        landmarks.append(lm.z - pulso.z)
    return landmarks


def _extrair_ambas_maos(hand_landmarks_list, handedness_list):
    """
    Extrai e concatena as features das duas mãos em ordem consistente:
        [mão direita (63 features)] + [mão esquerda (63 features)] = 126 features

    Mão não detectada → bloco de zeros (o modelo aprende que zeros = ausente).
    A ordem direita/esquerda é determinada pela label de lateralidade do MediaPipe,
    garantindo consistência entre frames independente de qual mão foi detectada primeiro.

    Nota: o MediaPipe reporta "Right"/"Left" do ponto de vista da câmera (espelhado).
    Usamos a convenção direita/esquerda da câmera de forma consistente.
    """
    feats_direita = _ZEROS_MAO
    feats_esquerda = _ZEROS_MAO

    for hand_lms, handed in zip(hand_landmarks_list, handedness_list):
        label = handed[0].category_name  # "Right" ou "Left"
        feats = _normalizar_mao(hand_lms)
        if label == "Right":
            feats_direita = feats
        else:
            feats_esquerda = feats

    return feats_direita + feats_esquerda


def _listar_frames(directory):
    """Retorna lista de imagens ordenada numericamente."""
    file_names = [
        f for f in os.listdir(directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    try:
        file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    except Exception:
        file_names.sort()
    return file_names


def _video_dirs_de_classe(class_dir):
    """
    Retorna os diretórios de vídeo para uma classe.

    Novo formato (1 subpasta por vídeo): class_dir/v0000/, class_dir/v0001/, ...
    Formato legado (frames direto na pasta): class_dir/frame_0.jpg, ...

    A separação por subpasta é necessária para evitar que janelas LSTM
    cruzem a fronteira entre vídeos diferentes.
    """
    subdirs = sorted([
        os.path.join(class_dir, d)
        for d in os.listdir(class_dir)
        if os.path.isdir(os.path.join(class_dir, d))
    ])
    return subdirs if subdirs else [class_dir]
#criador -> vitor izidoro

def extract_features_from_directory(
    dataset_root_dir,
    model_asset_path="hand_landmarker.task",
    mode="lstm",
    sequence_length=20,
    step=1,
    export_dataframe=False,
):
    """
    Varre os diretórios de imagens e extrai coordenadas normalizadas de AMBAS as mãos.

    Retorna features com 126 features por frame (63 mão direita + 63 mão esquerda).
    Mão não detectada → 63 zeros naquele bloco.

    Parâmetros
    ----------
    mode : "rf" → 2D (amostras, 126) | "lstm" → 3D (amostras, frames, 126)
    step : passo da janela deslizante (padrão 1 → máximo de sequências por vídeo)
    export_dataframe : salva CSV em ./dataset/

    Sobre augmentation
    ------------------
    Não é feita aqui. Ocorre dentro de train_* (model_training.py) APÓS o split,
    evitando data leakage. O CSV exportado contém apenas dados originais.
    """
    features = []
    labels = []

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_asset_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,   # detectar até 2 mãos por frame
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        for label_name in sorted(os.listdir(dataset_root_dir)):
            class_dir = os.path.join(dataset_root_dir, label_name)

            if not os.path.isdir(class_dir):
                continue

            print(f"Extraindo features do gesto: {label_name}...")

            video_dirs = _video_dirs_de_classe(class_dir)
            sequencias_classe = 0

            for video_dir in video_dirs:
                file_names = _listar_frames(video_dir)
                if not file_names:
                    continue

                video_landmarks = []
                ultimo_valido = [0.0] * NUM_FEATURES

                for file_name in file_names:
                    image_path = os.path.join(video_dir, file_name)
                    try:
                        mp_image = mp.Image.create_from_file(image_path)
                        results = landmarker.detect(mp_image)

                        if results.hand_landmarks:
                            landmarks = _extrair_ambas_maos(
                                results.hand_landmarks,
                                results.handedness,
                            )
                            ultimo_valido = landmarks

                            if mode == "rf":
                                features.append(landmarks)
                                labels.append(label_name)
                            else:
                                video_landmarks.append(landmarks)
                        else:
                            # Forward-fill com o último frame válido
                            if mode == "lstm":
                                video_landmarks.append(ultimo_valido)

                    except Exception as e:
                        print(f"Erro ao processar {image_path}: {e}")

                if mode == "lstm":
                    for i in range(0, len(video_landmarks) - sequence_length + 1, step):
                        features.append(video_landmarks[i: i + sequence_length])
                        labels.append(label_name)
                        sequencias_classe += 1

            if mode == "lstm":
                if sequencias_classe == 0:
                    print(
                        f"  AVISO: '{label_name}' gerou 0 sequências. "
                        f"Verifique se os vídeos têm >= {sequence_length} frames detectáveis."
                    )
                else:
                    print(f"  -> {sequencias_classe} sequência(s) para '{label_name}'")

    print(f"\nExtração concluída! Total ({mode}): {len(features)} amostras, {NUM_FEATURES} features/frame")

    if export_dataframe:
        _exportar_csv(features, labels, mode)

    return features, labels


def _exportar_csv(features, labels, mode):
    """Salva o dataset em CSV."""
    # Colunas: d_x_1, d_y_1, d_z_1, ..., d_x_21, d_y_21, d_z_21,
    #          e_x_1, e_y_1, e_z_1, ..., e_x_21, e_y_21, e_z_21
    coord_cols = []
    for prefixo in ('d', 'e'):  # d = direita, e = esquerda
        for i in range(1, 22):
            coord_cols += [f'{prefixo}_x_{i}', f'{prefixo}_y_{i}', f'{prefixo}_z_{i}']

    os.makedirs('./dataset', exist_ok=True)

    if mode == 'rf':
        df = pd.DataFrame(features, columns=coord_cols)
        df.insert(0, 'target', labels)

    elif mode == 'lstm':
        print(f'Exportando {len(labels)} amostras...')
        rows = []
        for sample_idx, (seq, label) in enumerate(zip(features, labels)):
            for frame_idx, frame in enumerate(seq):
                row = [label, sample_idx, frame_idx] + list(frame)
                rows.append(row)
        all_cols = ['target', 'sample_idx', 'frame_idx'] + coord_cols
        df = pd.DataFrame(rows, columns=all_cols)

    output_path = f'./dataset/dataset_completo_{mode}.csv'
    df.to_csv(output_path, index=False)
    print(f'Dataset exportado: {output_path}')


if __name__ == "__main__":
    dataset_root = "dataset/frames_treino"

    print("=== Modo LSTM ===")
    extract_features_from_directory(dataset_root_dir=dataset_root, mode='lstm', export_dataframe=True)

    print("\n=== Modo RF/KNN ===")
    extract_features_from_directory(dataset_root_dir=dataset_root, mode='rf', export_dataframe=True)
#criador -> vitor izidoro
