import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from app import helper

# Initialize Flask app
app = Flask(__name__, template_folder='../templates')

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for document data (for demonstration purposes)
document_data = {
    "chunks": None,
    "embeddings": None
}

def allowed_file(filename):
    """Check if the uploaded file has a .pdf extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """Handle PDF upload, processing, and embedding generation."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the PDF
        try:
            # 1. Extract text
            text = helper.extract_text_from_pdf(filepath)
            # 2. Chunk text
            chunks = helper.chunk_text(text)
            # 3. Generate embeddings
            model = helper.get_model()
            embeddings = model.encode(chunks, show_progress_bar=True)

            # Store data in memory
            document_data["chunks"] = chunks
            document_data["embeddings"] = embeddings

            return jsonify({"message": "PDF processed successfully!"}), 200
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

@app.route('/ask', methods=['POST'])
def ask():
    """Handle user questions and return relevant document chunks."""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query not provided"}), 400

    query = data['query']

    # Check if a document has been processed
    if document_data["chunks"] is None or document_data["embeddings"] is None:
        return jsonify({"error": "Please upload a PDF first."}), 400

    # Perform search
    try:
        results = helper.search(query, document_data["chunks"], document_data["embeddings"], top_k=3)
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred during search: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
