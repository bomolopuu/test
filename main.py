import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from transformers import pipeline

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the Hugging Face pipeline for the MMS model
model_id = "facebook/mms-1b-all"
pipe = pipeline(task="automatic-speech-recognition", model=model_id, model_kwargs={"target_lang": "moa", "ignore_mismatched_sizes": True})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the file is part of the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        # Check if the file has a valid filename
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Transcribe the file
                transcription = transcribe_audio(filepath)
                
                # Clean up the uploaded file after transcription
                os.remove(filepath)
                
                # Return transcription result as JSON
                return jsonify({"transcription": transcription})
            except Exception as e:
                return jsonify({"error": str(e)}), 500  # Return error as JSON
    else:
        # Serve the index.html page for GET requests
        return render_template('index.html')

def transcribe_audio(file_path):
    try:
        # Use the Hugging Face pipeline to transcribe the audio
        transcription_result = pipe(file_path)
        return transcription_result['text'] if 'text' in transcription_result else 'No transcription available'
    except Exception as e:
        # Handle any errors during transcription and return as JSON
        return f"Error during transcription: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
