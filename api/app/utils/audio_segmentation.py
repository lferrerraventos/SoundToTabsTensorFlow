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
    silences = librosa.effects.split(y, top_db=32)  # lower for more sensitivity

    # Detect significant RMS energy changes
    rms = librosa.feature.rms(y=y)[0]
    rms_diff = np.diff(rms)
    significant_energy_changes = np.where(np.abs(rms_diff) > np.percentile(np.abs(rms_diff), 90))[0] * 512

    # Pitch tracking
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    dominant_pitches = [np.max(pitch[np.nonzero(pitch)]) if np.any(pitch) else 0 for pitch in pitches.T]
    pitch_changes = np.diff(dominant_pitches)
    significant_pitch_changes = np.where(np.abs(pitch_changes) > np.percentile(np.abs(pitch_changes), 75))[0] * 512

    # Combine all segment boundaries
    segments = sorted(set([s[0] for s in silences] + [s[1] for s in silences] + significant_energy_changes.tolist() + significant_pitch_changes.tolist()))

    # Filter segments by minimum duration and minimum RMS
    final_segments = []
    min_duration = 1.5 # Minimum duration of a segment in seconds
    last_pos = segments[0]
    for pos in segments[1:]:
        if (pos - last_pos) > sr * min_duration and librosa.feature.rms(y=y[last_pos:pos]).max() > 0.01:
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
