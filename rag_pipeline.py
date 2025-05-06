import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from neo4j import GraphDatabase
import PyPDF2
from pathlib import Path
import hashlib

# Load environment variables
load_dotenv()

class GraphRAGPipeline:
    def __init__(self):
        # Initialize Neo4j connection
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            task_type="retrieval_document"
        )
        
        # Initialize text splitter with better parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Increased chunk size
            chunk_overlap=400,  # Increased overlap
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # More natural separators
            keep_separator=True  # Keep separators to maintain context
        )
        
        # Initialize vector store
        self.vector_store = None

    def clear_neo4j_database(self):
        """Clear all data from Neo4j database"""
        try:
            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password)
            )
            with driver.session() as session:
                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")
                # Drop the specific index we use
                session.run("DROP INDEX document_embeddings IF EXISTS")
            driver.close()
            print("Neo4j database cleared successfully")
        except Exception as e:
            print(f"Warning: Error clearing Neo4j database: {str(e)}")

    def process_pdf(self, pdf_path: str) -> List[str]:
        """Process a PDF file and extract text content"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Clear all previous data from Neo4j
        self.clear_neo4j_database()
        
        texts = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    # Clean and normalize the text
                    text = text.replace('\n', ' ').strip()  # Replace newlines with spaces
                    text = ' '.join(text.split())  # Normalize whitespace
                    texts.append(text)
        return texts

    def create_vector_store(self, texts: List[str]):
        """Create or update the vector store in Neo4j"""
        # Split texts into chunks
        chunks = self.text_splitter.create_documents(texts)
        
        # Create vector store
        self.vector_store = Neo4jVector.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name="document_embeddings"
        )
        return self.vector_store

    def setup_qa_chain(self):
        """Set up the QA chain with custom prompt"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please process documents first.")

        prompt_template = """
        You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        If the context seems incomplete or irrelevant to the question, please mention that the context might not be sufficient.

        Context: {context}

        Question: {question}
        Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 4}  # Retrieve more context chunks
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        return qa_chain

    def query(self, question: str) -> Dict:
        """Query the RAG pipeline"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please process documents first.")
            
        qa_chain = self.setup_qa_chain()
        result = qa_chain.invoke({"query": question})
        return result

def main():
    pipeline = GraphRAGPipeline()
    pdf_path = r"B:\RAGWITHKNOWLEDGEGRAPHS\A Study on Translating Logical Specifications to Natural Language using Large Language Models(1).pdf"
    try:
        texts = pipeline.process_pdf(pdf_path)
        pipeline.create_vector_store(texts)
        print("You can now ask questions about the PDF. Type 'exit' to quit.")
        while True:
            question = input("Your question: ")
            if question.strip().lower() in ["exit", "quit"]:
                print("Exiting.")
                break
            try:
                result = pipeline.query(question)
                print(f"Answer: {result['result']}")
            except Exception as e:
                print(f"Error: {str(e)}")
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 