import uuid

from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from app.utils import predict_from_wav

main = Blueprint('main', __name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@main.route('/uploads/<path:filename>', methods=['GET'])
def download_file(filename):
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    return send_from_directory(os.path.join(location, '..', UPLOAD_FOLDER), filename)

@main.route('/upload/transcription', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        segments_id = uuid.uuid4()
        filename = str(segments_id) + os.path.splitext(file.filename)[1]
        file_path = os.path.join("./uploads/", filename)
        file.save(file_path)

        predictions = predict_from_wav(segments_id, file_path)

        return jsonify({
            "segments_id": segments_id,
            "audio_path": "/uploads/" + filename,
            "predictions": predictions
        })

    return jsonify({"error": "Unsupported file type"}), 400


