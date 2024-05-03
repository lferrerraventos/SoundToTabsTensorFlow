import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from .audio_segmentation import remove_segments_folder, segment_audio

model_path = "app/models/"
def get_note_classes():
    notes_classes = {'1_0': 0,
                     '1_1': 1,
                     '1_2': 2,
                     '1_3': 3,
                     '2_0': 4,
                     '2_1': 5,
                     '2_2': 6,
                     '2_3': 7,
                     '3_0': 8,
                     '3_1': 9,
                     '3_2': 10,
                     '3_3': 11,
                     '4_0': 12,
                     '4_1': 13,
                     '4_2': 14,
                     '4_3': 15,
                     '5_0': 16,
                     '5_1': 17,
                     '5_2': 18,
                     '5_3': 19,
                     '6_0': 20,
                     '6_1': 21,
                     '6_2': 22,
                     '6_3': 23}

    return {v: k for k, v in notes_classes.items()}

def get_chord_classes():
    chords_classes = {'A': 0,
                      'AMin': 1,
                      'B': 2,
                      'BMin': 3,
                      'C': 4,
                      'D': 5,
                      'DMin': 6,
                      'E': 7,
                      'EMin': 8,
                      'F': 9,
                      'G': 10}

    return {v: k for k, v in chords_classes.items()}


def load_spectogram_image(path):
    target_size=(224, 224)
    img = image.load_img(path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_chord_or_note(spectrogram):
    # Load the binary model to classify between chord and note
    chordvsnote_model = tf.keras.models.load_model(model_path + "chordvsnote.h5")


    # Predict if it's a chord or a note
    prediction = chordvsnote_model.predict(load_spectogram_image(spectrogram))
    is_chord = prediction[0][0] < 0.5

    if is_chord:
        index_to_chords = get_chord_classes()
        classification = 'chord'
        # Load the chord model and predict
        chords_model = tf.keras.models.load_model(model_path + "chords.h5")
        chord_prediction = chords_model.predict(load_spectogram_image(spectrogram))
        chord_key = np.argmax(chord_prediction, axis=1)
        chord_name = index_to_chords[chord_key[0]]
        result = (classification, chord_name)
    else:
        index_to_notes = get_note_classes()
        classification = 'note'
        # Load the note model and predict
        notes_model = tf.keras.models.load_model(model_path + "notes.h5")
        note_prediction = notes_model.predict(load_spectogram_image(spectrogram))
        note_key = np.argmax(note_prediction, axis=1)
        note_name = index_to_notes[note_key[0]]
        result = (classification, note_name)

    return result


def predict_from_wav(segments_id, wav_file_path):
    segment_files = segment_audio(segments_id, wav_file_path)
    predictions = []
    last_prediction = None

    for spectrogram_path in segment_files:
        prediction = predict_chord_or_note(spectrogram_path)
        current_prediction = {
            "type": prediction[0],
            "value": prediction[1]
        }

        # Check if the current prediction is the same as the last one
        if last_prediction and current_prediction["value"] == last_prediction["value"]:
            continue  # Skip this prediction if it's the same as the last one

        predictions.append(current_prediction)
        last_prediction = current_prediction

    remove_segments_folder(segments_id)
    return predictions

