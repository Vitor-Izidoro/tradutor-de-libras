import mediapipe as mp
import cv2
import pickle
import numpy as np 

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def recognize_sign(video_path, tipo_modelo='1'):
    """
    Reconhece gestos em um vídeo usando o modelo selecionado (RF, LSTM ou KNN).
    """
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

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO
    )
    
    with HandLandmarker.create_from_options(options) as hand_landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Verificação se o vídeo acabou ou falhou
            if not ret:
                if frame_idx == 0:
                    print("\n[!] ERRO DE CODEC: O OpenCV encontrou o arquivo, mas não conseguiu decodificar as imagens.")
                    print("Tente converter o vídeo para .mp4 padrão (H.264) ou testar com outro arquivo.")
                else:
                    print(f"\nFim do vídeo alcançado. Total de frames lidos: {frame_idx}")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            timestamp_ms = int((frame_idx / fps) * 1000)
            
            results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            # 3. LÓGICA DE EXTRAÇÃO E PREVISÃO
            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                
                if modo_avaliacao == "estatico":
                    prediction = model.predict([landmarks])
                    gesto_atual = prediction[0]
                    # Print no terminal para facilitar a depuração
                    print(f"Frame {frame_idx:03d} | Gesto detectado: {gesto_atual}")

                elif modo_avaliacao == "continuo":
                    sequence_buffer.append(landmarks)
            else:
                if modo_avaliacao == "continuo":
                    sequence_buffer.append([0.0] * 42)

            if modo_avaliacao == "continuo" and len(sequence_buffer) == sequence_length:
                input_data = np.expand_dims(sequence_buffer, axis=0)
                res = model.predict(input_data, verbose=0)[0]
                predicted_idx = np.argmax(res)
                
                gesto_atual = label_encoder.inverse_transform([predicted_idx])[0]
                print(f"Frame {frame_idx:03d} | Movimento traduzido: {gesto_atual}")
                
                sequence_buffer.pop(0)

            frame_idx += 1

            # 4. RENDERIZAÇÃO NA TELA COM VELOCIDADE CORRIGIDA
            cv2.putText(frame, f"Traducao: {gesto_atual}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("Reconhecimento de Sinais - Tradutor", frame)
            
            # Calcula o tempo de espera correto para o vídeo não rodar "acelerado"
            tempo_espera_ms = int(1000 / fps)
            if cv2.waitKey(tempo_espera_ms) & 0xFF == ord('q'):
                print("Teste interrompido pelo usuário.")
                break

    cap.release()
    cv2.destroyAllWindows()