from data_preprocessing import extract_frames

extract_frames(
    video_path="dataset/Agarrar_Articulador3.mp4",
    output_root_dir="dataset/frames_agarrar",
    gesture_label="agarrar",
    frame_rate=1
)
