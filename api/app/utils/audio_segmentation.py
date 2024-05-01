import librosa
import librosa.display
import soundfile as sf
import numpy as np
import os
import matplotlib.pyplot as plt

def get_segments_folder(segments_id):
    return "segments/" + str(segments_id) + "/";

def remove_segments_folder(segments_id):
    folder = get_segments_folder(segments_id)
    os.rmdir(folder)
def create_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(3, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def segment_audio(segments_id, song_path):
    min_duration = 0.1  # duración mínima en segundos
    min_rms = 0.01  # mínimo RMS para considerar un segmento no silencioso
    output_folder = get_segments_folder(segments_id)
    y, sr = librosa.load(song_path)

    # Detectar segmentos usando silencios
    non_mute_sections = librosa.effects.split(y, top_db=35)

    # Detectar onsets
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='samples', delta=0.02, post_avg=3, pre_avg=3)

    # Combina onsets y segmentos no mudos
    all_changes = sorted(set(onsets.tolist() + [item for sublist in non_mute_sections for item in sublist]))

    # Crear segmentos finales
    refined_segments = []
    last_pos = all_changes[0]
    for change in all_changes[1:]:
        if (change - last_pos) > sr * min_duration:  # Asegurarse de que el segmento tenga una duración mínima
            refined_segments.append((last_pos, change))
            last_pos = change

    # Filtrar segmentos por energía
    refined_segments = [(start, end) for start, end in refined_segments if
                        librosa.feature.rms(y=y[start:end]).max() > min_rms]

    # Crear y guardar espectrogramas para cada segmento refinado
    file_paths = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, (start, end) in enumerate(refined_segments):
        segment_save_path = os.path.join(output_folder, f'segment_{i}.png')
        segment_audio_path = os.path.join(output_folder, f'segment_{i}.wav')
        segment_data = y[start:end]
        if np.any(segment_data):
            sf.write(segment_audio_path, segment_data, sr)
            create_spectrogram(segment_audio_path, segment_save_path)
            file_paths.append(segment_save_path)

    return file_paths
