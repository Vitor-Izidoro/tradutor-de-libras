import cv2
import os

def extract_frames(video_path, output_root_dir, gesture_label, frame_rate=5):
    """
    Extrai frames de um vídeo em uma taxa específica e salva em uma subpasta com o nome do gesto.
    """
    # Cria a pasta específica para o gesto dentro do diretório raiz
    output_dir = os.path.join(output_root_dir, gesture_label)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        count += 1

    cap.release()
    print(f"Frames da classe '{gesture_label}' salvos em {output_dir}")