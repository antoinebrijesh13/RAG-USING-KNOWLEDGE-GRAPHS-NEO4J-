# PDF RAG Query System

A powerful RAG (Retrieval-Augmented Generation) system that allows users to upload PDF documents and ask questions about their content using advanced AI technology. The system uses Neo4j as a vector database and Google's Gemini model for generating responses.

## Features

- PDF document upload and processing
- Advanced text chunking with context preservation
- Vector-based semantic search using Neo4j
- Interactive web interface
- Real-time question answering
- Modern, clean UI design

## Prerequisites

- Python 3.8+
- Neo4j Database
- Google API Key (for Gemini)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-rag-query.git
cd pdf-rag-query
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your credentials:
```
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
GOOGLE_API_KEY=your_google_api_key
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://127.0.0.1:5000`

3. Upload a PDF document using the web interface

4. Ask questions about the content of your PDF

## Project Structure

```
pdf-rag-query/
├── app.py                 # Flask application
├── rag_pipeline.py        # RAG pipeline implementation
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface template
└── .env                  # Environment variables (not tracked by git)
```

## Technologies Used

- **Backend**: Python, Flask
- **Database**: Neo4j
- **AI Model**: Google Gemini
- **Frontend**: HTML, CSS, JavaScript
- **Vector Store**: Neo4jVector
- **Text Processing**: LangChain

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the RAG implementation
- Neo4j for vector database capabilities
- Google for the Gemini AI model 
