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



def segment_audio(segment_id, song_path):
    output_folder = get_segments_folder(segment_id)
    y, sr = librosa.load(song_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    min_duration_sec = 0.75

    total_audio_duration = librosa.get_duration(y=y, sr=sr)
    file_paths = []
    start_frame = 0
    total_duration_processed = 0

    for onset_frame in onset_frames:
        start_time = librosa.frames_to_time(start_frame, sr=sr)
        end_time = librosa.frames_to_time(onset_frame, sr=sr)
        duration = end_time - start_time
        total_duration_processed += duration
        if duration >= min_duration_sec:
            start_sample = librosa.frames_to_samples(start_frame)
            end_sample = librosa.frames_to_samples(onset_frame)

            segment_audio_path = os.path.join(output_folder, f'segment_{len(file_paths)}.wav')
            segment_save_path = os.path.join(output_folder, f'segment_{len(file_paths)}.png')
            # Write segment audio
            sf.write(segment_audio_path, y[start_sample:end_sample], sr)

            # Create and save spectrogram
            create_spectrogram(segment_audio_path, segment_save_path)
            file_paths.append(segment_save_path)

        # Update start frame for next segment
        start_frame = onset_frame

    # Handle the last segment if there's remaining audio after the last onset
    if total_audio_duration > total_duration_processed:
        end_frame = len(y)  # the end of the audio
        end_time = librosa.frames_to_time(end_frame, sr=sr)
        duration = end_time - librosa.frames_to_time(start_frame, sr=sr)
        if duration >= min_duration_sec:
            start_sample = librosa.frames_to_samples(start_frame)
            end_sample = len(y)  # end of the audio data
            segment_audio_path = os.path.join(output_folder, f'segment_{len(file_paths)}.wav')
            segment_save_path = os.path.join(output_folder, f'segment_{len(file_paths)}.png')
            sf.write(segment_audio_path, y[start_sample:end_sample], sr)
            create_spectrogram(segment_audio_path, segment_save_path)
            file_paths.append(segment_save_path)


    return file_paths

