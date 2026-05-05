import cv2
import os
def extrair_dataset_completo(pasta_videos_brutos, pasta_destino_frames):
    """
    Varre a estrutura de pastas e extrai os vídeos automaticamente.
    Ex: extrair_dataset_completo("videos_brutos/treino", "dataset/frames_treino")
    """
    # Lista as pastas dos gestos (ex: 'agarrar', 'agora')
    for gesto_label in os.listdir(pasta_videos_brutos):
        caminho_gesto = os.path.join(pasta_videos_brutos, gesto_label)
        
        if not os.path.isdir(caminho_gesto):
            continue
            
        print(f"\nProcessando vídeos do gesto: {gesto_label}...")
        
        # Pega todos os mp4 dentro da pasta do gesto
        for nome_video in os.listdir(caminho_gesto):
            if nome_video.endswith('.mp4'): # ou avi, mov...
                caminho_video = os.path.join(caminho_gesto, nome_video)
                
                print(f" -> Extraindo: {nome_video}")
                
                # Chama a sua função atual
                extract_frames(
                    video_path=caminho_video, 
                    output_root_dir=pasta_destino_frames, 
                    gesture_label=gesto_label
                )
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