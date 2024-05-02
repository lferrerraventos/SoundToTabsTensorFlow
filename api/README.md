# SoundToTabs Flask API Example

This Flask API uses TensorFlow models that have been trained using the [notebooks](/notebooks) provided in this repository
to facilitate the classification of guitar recordings.


## Getting Started

### Prerequisites

Before you run the API, you need to ensure you have Python installed on your machine. Additionally, you will need to install the necessary Python libraries which can be found in the `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Downloading the Models
The trained model files are not included in the repository due to their large size (each over 150MB).

You can download the models from the following link:
- [Download SoundToTabs models](https://drive.google.com/file/d/1QsAhzKn9kERBkjWmwzMhJ5vzi8c8mJni/view?usp=sharing)

## Installation Instructions
After downloading, place the model files in this folder `api/app/models/` to ensure the Flask API functions correctly. The expected models are:
- `chords.h5`
- `chordvsnote.h5`
- `notes.h5`

## Model Prediction Format

The model outputs predictions in two formats, depending on whether the input corresponds to a chord or a note:

- **Chords**: The prediction for chords is represented using letters that denote the chord name. Examples include `E`, `G`, `D`, etc. Each letter corresponds to a specific chord played on the guitar.

- **Notes**: The prediction for individual notes follows a specific syntax: `{STRING_NUMBER}_{FRET_NUMBER}`. This format specifies which string and fret of the guitar are being played. For example:
  - `1_2` represents the note played on the first string at the second fret.
  - `6_0` represents the note played on the sixth string at the open position (fret 0).


## Running the API

Navigate to the root directory of the project and run the following command:

```bash
python run.py
```

## Using the API
To use the API, send a POST request to the `/upload/transcription` endpoint with a .wav file. Below is an example of how you might do this using curl:

```bash
curl -X POST -F 'file=@path_to_your_audio_file.wav' http://localhost:5000/upload/transcription
```


## Response

The API will return a JSON response that includes the classification of the audio file. Here is an example of a successful response:


```json
{
  "segments_id": "4741e100-6c69-4318-82df-49e4bc049c04",
  "audio_path": "uploads/4741e100-6c69-4318-82df-49e4bc049c04.wav",
  "predictions": 
  [
    {
      "type": "chord",
      "value": "D"
    },
     {
      "type": "chord",
      "value": "G"
    },
     {
      "type": "note",
      "value": "1_2"
    },
       {
      "type": "note",
      "value": "6_0"
    }
    
  ]
  
}

```



## Deployment

For deployment considerations, refer to the Flask deployment options on the official Flask website or consult deployment services like Heroku, AWS, or GCP for hosting your API.
This is just a simple example API implementation which has not been tested in production environments.
