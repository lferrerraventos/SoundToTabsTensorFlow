import uuid

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from api.app.utils.audio_classification import predict_from_wav

main = Blueprint('main', __name__)

@main.route('/upload/transcription', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.wav'):
        segments_id = uuid.uuid4()
        filename = str(segments_id) + ".wav"
        file_path = f"./uploads/{filename}"
        file.save(file_path)

        predictions = predict_from_wav(segments_id, file_path)

        return jsonify(predictions)

    return jsonify({"error": "Unsupported file type"}), 400


