import os
from data_preprocessing import extract_frames

def extrair_dataset_completo(pasta_videos_brutos, pasta_destino_frames):
    for gesto_label in os.listdir(pasta_videos_brutos):
        caminho_gesto = os.path.join(pasta_videos_brutos, gesto_label)
        
        if not os.path.isdir(caminho_gesto):
            continue
            
        print(f"\nProcessando vídeos do gesto: {gesto_label}...")
        
        for nome_video in os.listdir(caminho_gesto):
            if nome_video.endswith(('.mp4', '.avi')): 
                caminho_video = os.path.join(caminho_gesto, nome_video)
                print(f" -> Extraindo: {nome_video}")
                
                extract_frames(
                    video_path=caminho_video, 
                    output_root_dir=pasta_destino_frames, 
                    gesture_label=gesto_label
                )

if __name__ == "__main__":
    print("Iniciando Teste de Extração...")
    # Apontamos para a estrutura miniatura criada no Passo 1
    # E mandamos salvar em uma pasta de destino falsa
    extrair_dataset_completo(
        pasta_videos_brutos="videos/treino", 
        pasta_destino_frames="dataset/frames_treino_teste"
    )
    print("\nTeste finalizado!")