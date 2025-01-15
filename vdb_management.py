from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from datetime import datetime
from pathlib import Path
import json
import os
import logging

class vdb:
    def __init__(self, persist_directory="db", tracking_file="db_files.json"):  
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.persist_directory = persist_directory
        self.tracking_file = tracking_file
        #initializes the embedder/vectorizer
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ["GOOGLE_API_KEY"])
        
        # Create persist directory if it doesn't exist
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        #initilizes the vdb
        self.chromadb = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def add_file(self, file_path):
        try:
            self.logger.info(f"Attempting to add file: {file_path}")
            tracking_data = self._load_tracking()
            resolved_path = str(Path(file_path).resolve())
            
            # Check if file already exists
            if resolved_path in tracking_data:
                self.logger.warning(f"File already exists in database: {file_path}")
                return f"File {file_path} already exists in the database"
            
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return f"Error: File {file_path} does not exist"

            loader = TextLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            self.logger.info(f"Adding {len(texts)} chunks to vector database")
            ids = self.chromadb.add_documents(texts)
            # Remove persist() call - ChromaDB handles persistence automatically
            
            tracking_data[resolved_path] = {
                'chunk_ids': ids,
                'added_date': datetime.now().isoformat(),
                'file_name': Path(file_path).name
            }
            self._save_tracking(tracking_data)
            
            self.logger.info(f"Successfully added file: {file_path}")
            return f"Added {len(texts)} chunks from {file_path}"
        except Exception as e:
            self.logger.error(f"Error adding file {file_path}: {str(e)}", exc_info=True)
            return f"Error adding file: {str(e)}"

    def bulk_add_files(self, file_paths):
        self.logger.info(f"Starting bulk addition of {len(file_paths)} files")
        results = {}
        for file_path in file_paths:
            self.logger.info(f"Processing file: {file_path}")
            results[file_path] = self.add_file(file_path)
        self.logger.info("Bulk addition completed")
        return results

    def remove_file(self, file_path):
        try:
            tracking_data = self._load_tracking()
            resolved_path = str(Path(file_path).resolve())
            
            if resolved_path not in tracking_data:
                return f"File {file_path} not found in database"
            
            chunk_ids = tracking_data[resolved_path]['chunk_ids']
            self.chromadb.delete(chunk_ids)
            # Remove persist() call - ChromaDB handles persistence automatically
            
            del tracking_data[resolved_path]
            self._save_tracking(tracking_data)
            
            return f"Removed file {file_path} and its {len(chunk_ids)} chunks"
        except Exception as e:
            return f"Error removing file: {str(e)}"

    def list_files(self, detailed=False):
        tracking_data = self._load_tracking()
        if not detailed:
            return list(tracking_data.keys())
        
        return {
            path: {
                'file_name': data.get('file_name', Path(path).name),
                'chunk_count': len(data['chunk_ids']),
                'added_date': data['added_date']
            }
            for path, data in tracking_data.items()
        }

    def search_similar(self, query, k=3):
        docs = self.chromadb.similarity_search(query, k=k)
        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity': getattr(doc, 'similarity', None)
            }
            for doc in docs
        ]

    def get_collection_stats(self):
        collection = self.chromadb.get()
        tracking_data = self._load_tracking()
        
        return {
            'total_documents': len(collection['ids']),
            'total_files': len(tracking_data),
            'files': self.list_files(detailed=True)
        }

    def clear_database(self):
        try:
            self.chromadb.delete_collection()
            # Remove persist() call - ChromaDB handles persistence automatically
            if os.path.exists(self.tracking_file):
                os.remove(self.tracking_file)
            return "Database cleared successfully"
        except Exception as e:
            return f"Error clearing database: {str(e)}"

    def get_document_by_id(self, doc_id):
        try:
            collection = self.chromadb.get([doc_id])
            if collection['ids']:
                return {
                    'id': doc_id,
                    'content': collection['documents'][0],
                    'metadata': collection['metadatas'][0]
                }
            return None
        except Exception as e:
            return f"Error retrieving document: {str(e)}"

    def get_retriever(self, search_kwargs={'k': 5}):
        """Get a retriever instance for use with LangChain chains"""
        return self.chromadb.as_retriever(search_kwargs=search_kwargs)

    def initialize_from_directory(self, data_dir, glob_pattern="**/*.txt"):
        """Initialize the vector store with all matching files from a directory"""
        try:
            file_paths = []
            for file in Path(data_dir).glob(glob_pattern):
                if file.is_file():
                    file_paths.append(str(file))
            return self.bulk_add_files(file_paths)
        except Exception as e:
            return f"Error initializing from directory: {str(e)}"

    def _load_tracking(self):
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_tracking(self, tracking_data):
        with open(self.tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)