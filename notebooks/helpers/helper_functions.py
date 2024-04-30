"""
 @author: Luis Ferrer
 @description: Libreria que contiene funciones para convertir audio a espectograma,
                visualizar/reproducir datos de audio/espectogramas y varias funciones para trabajar con los datasets y visualizar
                el historial de los modelos entrenados.
                Algunas de las funciones provienen de: https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/helper_functions.py
"""

import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import os
from pathlib import Path
import random
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import itertools
from IPython.display import Audio, display


def predict_and_plot_from_wav_file(wav_file_path, model, class_indices, target_size=(224, 224)):
    # Convert the .wav file to a spectrogram image
    create_spectrogram(wav_file_path, 'temp_spectrogram.png')

    # Display the sound player
    display(Audio(wav_file_path))

    # Load and preprocess the spectrogram image
    img = image.load_img('temp_spectrogram.png', target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class_name = list(class_indices.keys())[predicted_class_index[0]]

    # Plotting
    plt.figure(figsize=(10, 4))

    # Display spectrogram
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Spectrogram of the Uploaded Chord")
    plt.axis('off')

    # Display prediction
    plt.subplot(1, 2, 2)
    plt.bar(range(len(predictions[0])), predictions[0], color='skyblue')
    plt.title(f"Predicted: {predicted_class_name}")
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.xticks(range(len(predictions[0])), list(class_indices.keys()), rotation=90)

    plt.tight_layout()
    plt.show()

    print(f"Predicted class: {predicted_class_name}")
    return predicted_class_name

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def show_images_from_path(path, label):
    images = []
    labels = []
    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        labels.append((label))

    random.shuffle(images)
    first10_images = images[:10]
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(first10_images[i] / 255)



def create_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(3, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_spectograms_from_dir(audio_dir, target_dir):
    audio_dir = Path(audio_dir)
    target_dir = Path(target_dir)
    for audio_file in audio_dir.rglob('*.mp3'):
        save_path = target_dir / audio_file.relative_to(audio_dir).with_suffix('.png')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        create_spectrogram(audio_file, save_path)



def unzip_data(filename):
    """
    Unzips filename into the current working directory.

    Args:
      filename (str): a filepath to a target zip folder to be unzipped.
    """
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()



def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.

    Args:
      dir_path (str): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} files in '{dirpath}'.")



def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();


