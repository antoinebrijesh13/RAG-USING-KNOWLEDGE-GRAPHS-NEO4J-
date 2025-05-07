from flask import Flask, request, render_template, jsonify
from rag_pipeline import GraphRAGPipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
pipeline = GraphRAGPipeline()

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['pdf']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            try:
                # Save the file
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(pdf_path)
                
                # Process the PDF
                texts = pipeline.process_pdf(pdf_path)
                pipeline.create_vector_store(texts)
                
                # Clean up the file after processing
                os.remove(pdf_path)
                
                return jsonify({'message': 'PDF processed successfully'}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    try:
        result = pipeline.query(question)
        return jsonify({'answer': result['result']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 