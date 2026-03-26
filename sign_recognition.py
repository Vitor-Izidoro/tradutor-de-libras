import mediapipe as mp
import cv2
import pickle

# Simplificando os imports usando as rotas oficiais e recomendadas pelo MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def recognize_sign(video_path, model_path="models/sign_model.pkl"):
    """
    Reconhece gestos em um vídeo usando um modelo treinado.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(video_path)
    
    # Pegamos o FPS do vídeo para calcular os timestamps necessários para a nova API
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_idx = 0

    # Configuração correta do HandLandmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO
    )
    hand_landmarker = HandLandmarker.create_from_options(options)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Converte o frame (array numpy) para o formato mp.Image exigido
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # 2. Calcula o timestamp do frame atual em milissegundos
        timestamp_ms = int((frame_idx / fps) * 1000)
        frame_idx += 1

        # 3. Processa o frame usando a função específica para o modo de vídeo
        results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                
                # O método predict do scikit-learn sempre espera um array 2D
                prediction = model.predict([landmarks])
                print(f"Gesto reconhecido: {prediction[0]}")

        cv2.imshow("Reconhecimento de Gestos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()