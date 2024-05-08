# Sound to Tabs TensorFlow
Author: [Luis Ferrer Raventós](https://www.linkedin.com/in/luis-ferrer-raventos/)

This project is part of my final assignment for TFG - UOC.

## Description

This repository includes the [TensorFlow notebooks](/notebooks) used to train 3 CNN models to classify guitar sound to chords and notes,
it also includes a notebook in order to segregate a guitar recording into multiple sound segments (this is done in order to classify each chord/note individually).

All the [datasets](/datasets) used to train and validate the models are also included, they have been recorded using 3 guitars:
1. Acoustic guitar (name: Hope)
2. Acoustic Parlor (name: Dreamer)
3. Electro Acoustic guitar (name: Whimsical)

Hope and Dreamer have been used for the training data, Whimsical was used for the test data.

I also included a small **Flask API** to deploy the models, please refer to the following readme to set it up:

- [SoundToTabs Flask API Example](api/README.md)


Finally, you can find a **Flutter APP** implementation that aims to convert guitar sounds to tabs using the Flask API in the following repository:
- [SoundToTabs Flutter APP](https://github.com/lferrerraventos/SoundToTabs)

## Important

**Note:** Due to the fact that this is an academic research work, the datasets do not cover all the notes and chords from a guitar. 

Regarding notes, it covers all the first 3 frets. 

For the chords, it covers those using the first 4 frets: 
- A & Amin
- B & Bmin
- C
- D & Dmin
- E & Emin
- F
- G



## Link to TensorFlow notebooks

* [SoundToTabs POC](notebooks/TFG_75_679_TensorFlow_POC_GuitarSoundToTabs.ipynb)
* [SoundToTabs Notes CNN](notebooks/TFG_75_679_TensorFlow_SoundToTabs_NotesCNN.ipynb) This notebook creates a categorical cross entropy CNN model to detect guitar single notes
* [SoundToTabs Chords CNN](notebooks/TFG_75_679_TensorFlow_SoundToTabs_ChordsCNN.ipynb) This notebook creates a categorical cross entropy CNN model to detect guitar chords.
* [SoundToTabs Note vs Chord CNN](notebooks/TFG_75_679_TensorFlow_SoundToTabs_NoteVsChordCNN.ipynb) This notebook creates a binary CNN model to differentiate between a guitar chord and a note.
* [SoundToTabs Audio Note Segregation](notebooks/TFG_75_679_TensorFlow_SoundToTabs_AudioNotesSegregation.ipynb) This aims to segregate into multiple segments a guitar recording.

## Link to datasets
* [Guitar Notes Dataset](https://github.com/lferrerraventos/SoundToTabsTensorFlow/raw/main/datasets/Notes.zip)
* [Guitar Chords Dataset](https://github.com/lferrerraventos/SoundToTabsTensorFlow/raw/main/datasets/Chords.zip)
* [Guitar Note vs Chord Light Dataset](https://github.com/lferrerraventos/SoundToTabsTensorFlow/raw/main/datasets/ChordVsNoteLight.zip)
* [Guitar Note vs Chord Full Dataset](https://github.com/lferrerraventos/SoundToTabsTensorFlow/raw/main/datasets/ChordsVsNotesFull.zip)
* [Guitar song to test 1](datasets/song_to_test1.wav)
* [Guitar song to test 2](datasets/song_to_test2.wav)
* [Guitar 1 chord song](datasets/1_chord_song.wav)