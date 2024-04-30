from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)

@main.route('/upload/transcription', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.wav'):
        filename = secure_filename(file.filename)
        file_path = f"./uploads/{filename}"
        file.save(file_path)




        return jsonify({"prediction": ""})

    return jsonify({"error": "Unsupported file type"}), 400
