import os
import mediapipe as mp
import pandas as pd
import numpy as np
from landmark_augmentation import gerar_amostras_aumentadas

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

NUM_FEATURES = 63  # 21 landmarks × (x, y, z)


def _extrair_landmarks(hand_landmarks):
    """
    Extrai e normaliza os landmarks relativos ao pulso (landmark 0), com eixo Z.
    A normalização garante que o modelo aprende o formato do gesto,
    não a posição da mão na tela.
    """
    pulso = hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks:
        landmarks.append(lm.x - pulso.x)
        landmarks.append(lm.y - pulso.y)
        landmarks.append(lm.z - pulso.z)
    return landmarks


def extract_features_from_directory(
    dataset_root_dir,
    model_asset_path="hand_landmarker.task",
    mode="lstm",
    sequence_length=20,
    step=5,
    export_dataframe=False,
    augmentar=False,
    n_aumentos=5
):
    """
    Varre os diretórios de imagens e extrai coordenadas normalizadas.

    Parâmetros
    ----------
    dataset_root_dir : str
        Pasta raiz com subpastas por gesto (ex: dataset/frames_treino).
    mode : "rf" ou "lstm"
        "rf"   -> features 2D (amostras, 63)        para RF e KNN
        "lstm" -> features 3D (amostras, frames, 63) para LSTM
    sequence_length : int
        Tamanho da janela temporal para o LSTM.
    step : int
        Passo da janela deslizante para o LSTM.
    export_dataframe : bool
        Se True, salva o dataset em CSV em ./dataset/.
    augmentar : bool
        Se True, aplica data augmentation nos landmarks apos a extracao.
        Recomendado quando ha poucos videos por gesto (< 10).
    n_aumentos : int
        Numero de amostras sinteticas geradas por amostra original.
        n_aumentos=5 -> ~6x mais dados. n_aumentos=10 -> ~11x mais dados.
    """
    features = []
    labels = []

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_asset_path),
        running_mode=VisionRunningMode.IMAGE
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        for label_name in sorted(os.listdir(dataset_root_dir)):
            class_dir = os.path.join(dataset_root_dir, label_name)

            if not os.path.isdir(class_dir):
                continue

            print(f"Extraindo features do gesto: {label_name}...")

            file_names = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            try:
                file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            except Exception:
                file_names.sort()

            class_landmarks = []
            ultimo_valido = [0.0] * NUM_FEATURES

            for file_name in file_names:
                image_path = os.path.join(class_dir, file_name)
                try:
                    mp_image = mp.Image.create_from_file(image_path)
                    results = landmarker.detect(mp_image)

                    if results.hand_landmarks:
                        landmarks = _extrair_landmarks(results.hand_landmarks[0])
                        ultimo_valido = landmarks

                        if mode == "rf":
                            features.append(landmarks)
                            labels.append(label_name)
                        else:
                            class_landmarks.append(landmarks)
                    else:
                        if mode == "lstm":
                            class_landmarks.append(ultimo_valido)

                except Exception as e:
                    print(f"Erro ao processar {image_path}: {e}")

            if mode == "lstm":
                sequencias_geradas = 0
                for i in range(0, len(class_landmarks) - sequence_length + 1, step):
                    features.append(class_landmarks[i: i + sequence_length])
                    labels.append(label_name)
                    sequencias_geradas += 1

                if sequencias_geradas == 0:
                    print(
                        f"  AVISO: '{label_name}' gerou 0 sequencias. "
                        f"{len(class_landmarks)} frames disponiveis, "
                        f"minimo necessario: {sequence_length}."
                    )
                else:
                    print(f"  -> {sequencias_geradas} sequencias geradas para '{label_name}'")

    print(f"\nExtracao concluida! Amostras reais ({mode}): {len(features)}")

    # ----------------------------------------------------------------
    # Augmentation — aplicada ANTES do export para o CSV ja conter
    # os dados aumentados, mantendo consistencia entre treino direto
    # e treino via CSV.
    #
    # IMPORTANTE: augmentation so deve ocorrer nos dados de TREINO.
    # Nunca passe dados de teste/validacao por aqui.
    # ----------------------------------------------------------------
    if augmentar and len(features) > 0:
        features, labels = gerar_amostras_aumentadas(
            features, labels,
            mode=mode,
            n_aumentos=n_aumentos
        )

    if export_dataframe:
        _exportar_csv(features, labels, mode, NUM_FEATURES)

    return features, labels


def _exportar_csv(features, labels, mode, num_features):
    """Salva o dataset em CSV (util para inspecao e para o pipeline via CSV)."""
    coord_cols = []
    for i in range(1, (num_features // 3) + 1):
        coord_cols += [f'x_{i}', f'y_{i}', f'z_{i}']

    os.makedirs('./dataset', exist_ok=True)

    if mode == 'rf':
        df = pd.DataFrame(features, columns=coord_cols)
        df.insert(0, 'target', labels)

    elif mode == 'lstm':
        print(f'Exportando {len(labels)} amostras (incluindo augmentadas)...')
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

    print("=== Modo LSTM (com augmentation) ===")
    extract_features_from_directory(
        dataset_root_dir=dataset_root,
        mode='lstm',
        augmentar=True,
        n_aumentos=5,
        export_dataframe=True
    )

    print("\n=== Modo RF/KNN (com augmentation) ===")
    extract_features_from_directory(
        dataset_root_dir=dataset_root,
        mode='rf',
        augmentar=True,
        n_aumentos=5,
        export_dataframe=True
    )