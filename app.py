from flask import Flask, request, render_template, jsonify
from rag_pipeline import GraphRAGPipeline
import os

app = Flask(__name__)
pipeline = GraphRAGPipeline()

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
                # Create uploads directory if it doesn't exist
                os.makedirs('uploads', exist_ok=True)
                
                # Save the file
                pdf_path = os.path.join('uploads', file.filename)
                file.save(pdf_path)
                
                # Process the PDF
                texts = pipeline.process_pdf(pdf_path)
                pipeline.create_vector_store(texts)
                
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
    app.run(debug=True) 