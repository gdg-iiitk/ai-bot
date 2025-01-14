from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from datetime import datetime
from pathlib import Path
import json
import os

class VectorStoreManager:
    def __init__(self, persist_directory="db", tracking_file="db_files.json", embedding_model="models/embedding-001"):
        self.persist_directory = persist_directory
        self.tracking_file = tracking_file
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        
        # Create persist directory if it doesn't exist
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
    def _load_tracking(self):
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_tracking(self, tracking_data):
        with open(self.tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)

    def add_file(self, file_path):
        try:
            tracking_data = self._load_tracking()
            resolved_path = str(Path(file_path).resolve())
            
            # Check if file already exists
            if resolved_path in tracking_data:
                return f"File {file_path} already exists in the database"
            
            loader = TextLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            ids = self.vectorstore.add_documents(texts)
            self.vectorstore.persist()
            
            tracking_data[resolved_path] = {
                'chunk_ids': ids,
                'added_date': datetime.now().isoformat(),
                'file_name': Path(file_path).name
            }
            self._save_tracking(tracking_data)
            
            return f"Added {len(texts)} chunks from {file_path}"
        except Exception as e:
            return f"Error adding file: {str(e)}"

    def bulk_add_files(self, file_paths):
        results = {}
        for file_path in file_paths:
            results[file_path] = self.add_file(file_path)
        return results

    def remove_file(self, file_path):
        try:
            tracking_data = self._load_tracking()
            resolved_path = str(Path(file_path).resolve())
            
            if resolved_path not in tracking_data:
                return f"File {file_path} not found in database"
            
            chunk_ids = tracking_data[resolved_path]['chunk_ids']
            self.vectorstore.delete(chunk_ids)
            self.vectorstore.persist()
            
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
        docs = self.vectorstore.similarity_search(query, k=k)
        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity': getattr(doc, 'similarity', None)
            }
            for doc in docs
        ]

    def get_collection_stats(self):
        collection = self.vectorstore.get()
        tracking_data = self._load_tracking()
        
        return {
            'total_documents': len(collection['ids']),
            'total_files': len(tracking_data),
            'files': self.list_files(detailed=True)
        }

    def clear_database(self):
        try:
            self.vectorstore.delete_collection()
            self.vectorstore.persist()
            if os.path.exists(self.tracking_file):
                os.remove(self.tracking_file)
            return "Database cleared successfully"
        except Exception as e:
            return f"Error clearing database: {str(e)}"

    def get_document_by_id(self, doc_id):
        try:
            collection = self.vectorstore.get([doc_id])
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
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

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
