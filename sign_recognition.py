import mediapipe as mp
import cv2
import pickle
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

# Deve ser igual ao frame_rate usado em data_preprocessing.extract_frames().
# O treino usa 1 frame a cada FRAME_RATE_TREINO frames do vídeo original.
# A inferência precisa da mesma subamostragem para que as janelas LSTM
# cubram o mesmo intervalo de tempo real que cobriram durante o treino.
# Sem isso, 20 frames de treino cobrem ~3,3s mas 20 frames de inferência
# (a 30fps) cobrem apenas ~0,67s — o modelo vê o gesto 5x mais rápido.
FRAME_RATE_TREINO = 5


def _kmeans_temporal_single(seq_array, n_clusters):
    """
    Aplica K-Means temporal a uma única sequência de frames para inferência.
    Mesma lógica de _aplicar_kmeans_temporal em model_training.py.
    """
    n_frames = len(seq_array)
    m = min(n_clusters, n_frames)

    km = KMeans(n_clusters=m, random_state=42, n_init=10)
    km.fit(seq_array)
    assignments = km.labels_

    centroid_order = []
    for c in range(m):
        frame_idxs = np.where(assignments == c)[0]
        median_idx = float(np.median(frame_idxs)) if len(frame_idxs) > 0 else float(c)
        centroid_order.append((median_idx, c))
    centroid_order.sort(key=lambda x: x[0])

    sorted_centroids = km.cluster_centers_[[c for _, c in centroid_order]]

    if m < n_clusters:
        pad = np.zeros((n_clusters - m, seq_array.shape[1]))
        sorted_centroids = np.vstack([sorted_centroids, pad])

    return sorted_centroids.flatten()


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


_ZEROS_MAO = [0.0] * 63


def _normalizar_mao(hand_landmarks):
    """Normaliza uma mão: centraliza no pulso, escala XY pela largura da palma."""
    pulso = hand_landmarks[0]
    base_ind = hand_landmarks[5]
    base_min = hand_landmarks[17]

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


def extrair_landmarks(hand_landmarks_list, handedness_list):
    """
    Extrai 126 features de até 2 mãos: [direita (63)] + [esquerda (63)].
    Mão ausente → 63 zeros. Ordem determinada pela lateralidade do MediaPipe.
    Deve ser idêntico a feature_extraction._extrair_ambas_maos.
    """
    feats_direita = _ZEROS_MAO
    feats_esquerda = _ZEROS_MAO

    for hand_lms, handed in zip(hand_landmarks_list, handedness_list):
        label = handed[0].category_name
        feats = _normalizar_mao(hand_lms)
        if label == "Right":
            feats_direita = feats
        else:
            feats_esquerda = feats

    return feats_direita + feats_esquerda


def recognize_sign(video_path, tipo_modelo='1'):
    """
    Reconhece gestos em um vídeo usando o modelo selecionado (RF, LSTM ou KNN).
    """
    scaler = None
    if tipo_modelo == '1':
        print("Carregando modelo Random Forest...")
        with open("models/sign_model.pkl", "rb") as f:
            model = pickle.load(f)
        modo_avaliacao = "estatico"

    elif tipo_modelo == '2':
        print("Carregando modelo LSTM...")
        from tensorflow.keras.models import load_model
        model = load_model("models/lstm_sign_model.h5")
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        modo_avaliacao = "continuo"
        sequence_length = 20
        sequence_buffer = []

    elif tipo_modelo == '3':
        print("Carregando modelo KNN...")
        with open("models/knn_sign_model.pkl", "rb") as f:
            pacote_knn = pickle.load(f)
            model = pacote_knn['model']
            scaler = pacote_knn['scaler']
            n_clusters = pacote_knn.get('n_clusters', 10)
            usa_kmeans = pacote_knn.get('usa_kmeans', False)
            sequence_length = pacote_knn.get('sequence_length', 20) or 20

        if usa_kmeans:
            modo_avaliacao = "knn_kmeans"
            sequence_buffer = []
            print(f"KNN com K-Means temporal: M={n_clusters} centróides, janela={sequence_length} frames")
        else:
            modo_avaliacao = "estatico"

    else:
        print("Modelo não reconhecido!")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERRO: O OpenCV não conseguiu abrir o caminho: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30

    frame_idx = 0
    gesto_atual = "Aguardando sinal..."
    historico_predicoes = []

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,   # detectar até 2 mãos, igual ao treinamento
    )

    with HandLandmarker.create_from_options(options) as hand_landmarker:
        last_known_landmarks = [0.0] * 126  # 63 direita + 63 esquerda

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((frame_idx / fps) * 1000)
            results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks:
                landmarks = extrair_landmarks(results.hand_landmarks, results.handedness)
                last_known_landmarks = landmarks

                if modo_avaliacao == "estatico":
                    if scaler is not None:
                        landmarks_processados = scaler.transform([landmarks])
                        prediction = model.predict(landmarks_processados)
                    else:
                        prediction = model.predict([landmarks])
                    gesto_atual = prediction[0]
                    print(f"Frame {frame_idx:03d} | Gesto detectado: {gesto_atual}")
                    historico_predicoes.append(gesto_atual)

            # Para modelos sequenciais (LSTM, KNN+K-Means), adiciona ao buffer
            # apenas a cada FRAME_RATE_TREINO frames — mesmo intervalo usado
            # na extração de frames de treino. Isso garante que a janela de
            # 20 frames cobre o mesmo tempo real durante treino e inferência.
            if modo_avaliacao in ("continuo", "knn_kmeans"):
                if frame_idx % FRAME_RATE_TREINO == 0:
                    if results.hand_landmarks:
                        sequence_buffer.append(landmarks)
                    else:
                        sequence_buffer.append(last_known_landmarks)

            if modo_avaliacao == "continuo" and len(sequence_buffer) == sequence_length:
                input_data = np.expand_dims(sequence_buffer, axis=0)
                res = model.predict(input_data, verbose=0)[0]
                predicted_idx = np.argmax(res)
                gesto_atual = label_encoder.inverse_transform([predicted_idx])[0]
                print(f"Frame {frame_idx:03d} | Movimento traduzido: {gesto_atual}")
                historico_predicoes.append(gesto_atual)
                sequence_buffer.pop(0)

            elif modo_avaliacao == "knn_kmeans" and len(sequence_buffer) == sequence_length:
                seq_array = np.array(sequence_buffer, dtype=float)
                features_flat = _kmeans_temporal_single(seq_array, n_clusters)
                features_scaled = scaler.transform([features_flat])
                prediction = model.predict(features_scaled)
                gesto_atual = prediction[0]
                print(f"Frame {frame_idx:03d} | KNN+K-Means detectou: {gesto_atual}")
                historico_predicoes.append(gesto_atual)
                sequence_buffer.pop(0)

            frame_idx += 1

            cv2.putText(frame, f"Traducao: {gesto_atual}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Reconhecimento de Sinais - Tradutor", frame)

            tempo_espera_ms = int(1000 / fps)
            if cv2.waitKey(tempo_espera_ms) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if historico_predicoes:
        total_chutes = len(historico_predicoes)
        contagem = Counter(historico_predicoes)

        print("\n" + "=" * 40)
        print("  RESULTADO CONSOLIDADO DO VÍDEO  ")
        print("=" * 40)
        print(f"Total de predições realizadas: {total_chutes}")

        for gesto, qtd in contagem.most_common():
            porcentagem = (qtd / total_chutes) * 100
            print(f" -> {gesto}: {porcentagem:.2f}% de predominância ({qtd} frames)")
        print("=" * 40 + "\n")

        gesto_vencedor = contagem.most_common(1)[0][0]
        return gesto_vencedor
    else:
        print("\nNenhuma predição pôde ser feita neste vídeo.")
        return None
