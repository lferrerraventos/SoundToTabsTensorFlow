# SoundToTabs Flask API Examples

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
// todo: to be implemented
```

## Deployment

For deployment considerations, refer to the Flask deployment options on the official Flask website or consult deployment services like Heroku, AWS, or GCP for hosting your API.
This is just a simple example API implementation which has not been tested in production environments.
