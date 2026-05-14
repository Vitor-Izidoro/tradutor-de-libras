import mediapipe as mp
import cv2
import pickle
import numpy as np
from collections import Counter

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def extrair_landmarks(hand_landmarks):
    """
    Extrai e normaliza landmarks com eixo Z, relativo ao pulso.
    Deve ser idêntico ao usado no feature_extraction.py.
    """
    pulso = hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks:
        landmarks.append(lm.x - pulso.x)
        landmarks.append(lm.y - pulso.y)
        landmarks.append(lm.z - pulso.z)
    return landmarks


def recognize_sign(video_path, tipo_modelo='1'):
    """
    Reconhece gestos em um vídeo usando o modelo selecionado (RF, LSTM ou KNN).
    """
    scaler = None
    # 1. CARREGAMENTO DO MODELO ESPECÍFICO
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
            model = pickle.load(f)
        modo_avaliacao = "estatico"
        
    else:
        print("Modelo não reconhecido!")
        return

    # 2. CONFIGURAÇÃO DA CÂMERA/VÍDEO
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERRO: O OpenCV não conseguiu abrir o caminho: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) 
    if fps == 0 or np.isnan(fps): 
        fps = 30
        
    frame_idx = 0
    gesto_atual = "Aguardando sinal..."
    
    # --- NOVIDADE 1: Variável para guardar todos os chutes do modelo ---
    historico_predicoes = []

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO
    )
    
    with HandLandmarker.create_from_options(options) as hand_landmarker:
        
        # Variável de estado para o Forward Fill (mantendo a sua última atualização do código)
        last_known_landmarks = [0.0] * 63 
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            timestamp_ms = int((frame_idx / fps) * 1000)
            results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            # 3. LÓGICA DE EXTRAÇÃO E PREVISÃO
            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                landmarks = []
                
                base_x = hand_landmarks[0].x
                base_y = hand_landmarks[0].y
                base_z = hand_landmarks[0].z
                
                for lm in hand_landmarks:
                    landmarks.append(lm.x - base_x)
                    landmarks.append(lm.y - base_y)
                    landmarks.append(lm.z - base_z)
                
                last_known_landmarks = landmarks 
                
                if modo_avaliacao == "estatico":
                    prediction = model.predict([landmarks])
                    gesto_atual = prediction[0]
                    print(f"Frame {frame_idx:03d} | Gesto detectado: {gesto_atual}")
                    
                    # --- NOVIDADE 2: Salva o chute estático no histórico ---
                    historico_predicoes.append(gesto_atual)

                elif modo_avaliacao == "continuo":
                    sequence_buffer.append(landmarks)
            else:
                if modo_avaliacao == "continuo":
                    sequence_buffer.append(last_known_landmarks)

            if modo_avaliacao == "continuo" and len(sequence_buffer) == sequence_length:
                input_data = np.expand_dims(sequence_buffer, axis=0)
                res = model.predict(input_data, verbose=0)[0]
                predicted_idx = np.argmax(res)
                
                gesto_atual = label_encoder.inverse_transform([predicted_idx])[0]
                print(f"Frame {frame_idx:03d} | Movimento traduzido: {gesto_atual}")
                
                # --- NOVIDADE 3: Salva o chute contínuo (LSTM) no histórico ---
                historico_predicoes.append(gesto_atual)
                
                sequence_buffer.pop(0)

            frame_idx += 1

            # 4. RENDERIZAÇÃO NA TELA
            cv2.putText(frame, f"Traducao: {gesto_atual}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Reconhecimento de Sinais - Tradutor", frame)
            
            tempo_espera_ms = int(1000 / fps)
            if cv2.waitKey(tempo_espera_ms) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # --- NOVIDADE 4: Cálculo e impressão da porcentagem de certeza ao final do vídeo ---
    if historico_predicoes:
        total_chutes = len(historico_predicoes)
        contagem = Counter(historico_predicoes)
        
        print("\n" + "="*40)
        print("  RESULTADO CONSOLIDADO DO VÍDEO  ")
        print("="*40)
        print(f"Total de predições realizadas: {total_chutes}")
        
        # O most_common() já ordena do gesto que apareceu mais vezes para o que apareceu menos
        for gesto, qtd in contagem.most_common():
            porcentagem = (qtd / total_chutes) * 100
            print(f" -> {gesto}: {porcentagem:.2f}% de predominância ({qtd} frames)")
        print("="*40 + "\n")
    else:
        print("\nNenhuma predição pôde ser feita neste vídeo.")