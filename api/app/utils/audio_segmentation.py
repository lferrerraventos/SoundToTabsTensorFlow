import librosa
import librosa.display
import soundfile as sf
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

def get_segments_folder(segments_id):
    return "segments/" + str(segments_id) + "/";

def remove_segments_folder(segments_id):
    folder = get_segments_folder(segments_id)
    shutil.rmtree(folder)
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
    output_folder = get_segments_folder(segments_id)
    y, sr = librosa.load(song_path)

    # Detect silent sections
    non_mute_sections = librosa.effects.split(y, top_db=25)  # Decrease top_db to consider quieter sounds as silence

    # Harmonic content analysis using the chromagram
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_std = np.std(chroma, axis=0)  # Calculate the standard deviation of chroma

    # Generate potential segment changes
    changes = []
    for idx, val in enumerate(chroma_std):
        if val > np.percentile(chroma_std, 75):  # Use a high percentile to ensure only significant changes are noted
            changes.append(idx * 512)  # Converting frames to samples

    # Combine silence boundaries with significant harmonic changes
    segments = sorted(set(changes + [start for start, end in non_mute_sections]))

    # Filter segments by minimum duration and minimum RMS
    final_segments = []
    min_duration = 0.5  # Increase duration to avoid too short segments
    min_rms = 0.01
    last_pos = segments[0]
    for pos in segments[1:]:
        if (pos - last_pos) > sr * min_duration and librosa.feature.rms(y=y[last_pos:pos]).max() > min_rms:
            final_segments.append((last_pos, pos))
            last_pos = pos

    # Create and save spectrograms
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_paths = []
    for i, (start, end) in enumerate(final_segments):
        segment_audio_path = os.path.join(output_folder, f'segment_{i}.wav')
        segment_save_path = os.path.join(output_folder, f'segment_{i}.png')
        sf.write(segment_audio_path, y[start:end], sr)
        create_spectrogram(segment_audio_path, segment_save_path)
        file_paths.append(segment_save_path)

    return file_paths
