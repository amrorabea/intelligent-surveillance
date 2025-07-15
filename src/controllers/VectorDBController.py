import chromadb
import os
import logging
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
from .BaseController import BaseController

logger = logging.getLogger(__name__)

class VectorDBController(BaseController):
    def __init__(self, collection_name="surveillance_data"):
        super().__init__()
        
        # Set up ChromaDB client with persistent storage
        db_dir = os.path.join(self.files_dir, "chromadb")
        os.makedirs(db_dir, exist_ok=True)
        
        self.client = None
        self.encoder = None
        self.collection = None
        self.collection_name = collection_name
        
        # Initialize connections
        self._initialize_client(db_dir)
        self._initialize_encoder()
        self._initialize_collection()
        
    def _initialize_client(self, db_dir: str):
        """Initialize ChromaDB client"""
        try:
            self.client = chromadb.PersistentClient(path=db_dir)
            logger.info(f"ChromaDB client initialized at {db_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            self.client = None
    
    def _initialize_encoder(self):
        """Initialize sentence transformer encoder"""
        try:
            # TODO: You may want to use a different embedding model
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.encoder = None
    
    def _initialize_collection(self):
        """Initialize or get ChromaDB collection"""
        if not self.client:
            logger.error("Cannot initialize collection: ChromaDB client not available")
            return
            
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except Exception as e:
            logger.info(f"Collection {self.collection_name} not found: {e}, creating it")
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            except Exception as create_error:
                logger.error(f"Failed to create collection: {create_error}")
                self.collection = None
        
    def is_available(self) -> bool:
        """Check if vector database is available for operations"""
        return all([self.client, self.encoder, self.collection])
        
    def store_embedding(self, document_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Store text embeddings in vector database
        
        Args:
            document_id (str): Unique identifier for the document
            text (str): Text content to embed
            metadata (dict): Additional metadata for retrieval
            
        Returns:
            bool: Success status
        """
        if not self.is_available():
            logger.error("Vector database not available for storing embeddings")
            return False
            
        try:
            # Validate inputs
            if not document_id or not text:
                logger.error("Document ID and text are required")
                return False
                
            # Generate embedding
            embedding = self.encoder.encode(text)
            
            # Convert numpy array to list for ChromaDB
            embedding_list = embedding.tolist()
            
            # Ensure metadata is serializable
            clean_metadata = self._clean_metadata(metadata)
            
            # Check if document already exists
            try:
                existing = self.collection.get(ids=[document_id])
                if existing['ids']:
                    # Update existing document
                    self.collection.update(
                        ids=[document_id],
                        embeddings=[embedding_list],
                        metadatas=[clean_metadata],
                        documents=[text]
                    )
                    logger.debug(f"Updated existing document: {document_id}")
                else:
                    # Add new document
                    self.collection.add(
                        ids=[document_id],
                        embeddings=[embedding_list],
                        metadatas=[clean_metadata],
                        documents=[text]
                    )
                    logger.debug(f"Added new document: {document_id}")
            except Exception:
                # If check fails, try to add (will fail if exists)
                self.collection.add(
                    ids=[document_id],
                    embeddings=[embedding_list],
                    metadatas=[clean_metadata],
                    documents=[text]
                )
                logger.debug(f"Added document: {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing embedding for {document_id}: {e}")
            return False
        
    def semantic_search(self, query: str, limit: int = 10, 
                       filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar content using vector similarity
        
        Args:
            query (str): Natural language query
            limit (int): Maximum number of results
            filter_criteria (dict): Optional metadata filters
            
        Returns:
            list: Search results with documents and metadata
        """
        if not self.is_available():
            logger.error("Vector database not available for search")
            return []
            
        try:
            # Validate inputs
            if not query or limit <= 0:
                logger.error("Valid query and positive limit are required")
                return []
                
            # Generate query embedding
            query_embedding = self.encoder.encode(query)
            
            # Convert numpy array to list for ChromaDB
            query_embedding_list = query_embedding.tolist()
            
            # Prepare search parameters
            search_params = {
                "query_embeddings": [query_embedding_list],
                "n_results": min(limit, 100),  # Cap at 100 results
            }
            
            # Add filters if provided
            if filter_criteria:
                clean_filters = self._clean_metadata(filter_criteria)
                search_params["where"] = clean_filters
            
            # Execute search
            results = self.collection.query(**search_params)
            
            # Format results
            formatted_results = []
            if results and len(results['ids']) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    result_item = {
                        'id': doc_id,
                        'document': results['documents'][0][i] if 'documents' in results else None,
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                        'distance': results['distances'][0][i] if 'distances' in results else None,
                    }
                    formatted_results.append(result_item)
            
            logger.debug(f"Search for '{query}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in semantic search for query '{query}': {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector database"""
        if not self.is_available():
            logger.error("Vector database not available for deletion")
            return False
            
        try:
            self.collection.delete(ids=[document_id])
            logger.debug(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        if not self.is_available():
            return {"error": "Vector database not available"}
            
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "available": True
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e), "available": False}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection (use with caution)"""
        if not self.is_available():
            logger.error("Vector database not available for clearing")
            return False
            
        try:
            # Get all document IDs
            all_docs = self.collection.get()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
                logger.info(f"Cleared {len(all_docs['ids'])} documents from collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata to ensure it's serializable for ChromaDB"""
        clean_data = {}
        
        for key, value in metadata.items():
            # ChromaDB supports strings, numbers, and booleans
            if isinstance(value, (str, int, float, bool)):
                clean_data[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                clean_data[key] = ",".join(str(item) for item in value)
            else:
                # Convert other types to strings
                clean_data[key] = str(value)
        
        return clean_data
            
    def delete_embeddings(self, document_ids):
        """
        Delete embeddings from the database
        
        Args:
            document_ids (list): List of document IDs to delete
            
        Returns:
            bool: Success status
        """
        try:
            self.collection.delete(ids=document_ids)
            return True
        except Exception as e:
            print(f"Error deleting embeddings: {e}")
            return False
            
    def update_embedding(self, document_id, new_text, new_metadata=None):
        """
        Update an existing embedding
        
        Args:
            document_id (str): ID of document to update
            new_text (str): New text content
            new_metadata (dict): New metadata (optional)
            
        Returns:
            bool: Success status
        """
        try:
            # Delete existing entry
            self.collection.delete(ids=[document_id])
            
            # Create new embedding
            new_embedding = self.encoder.encode(new_text)
            
            # Add updated entry
            self.collection.add(
                ids=[document_id],
                embeddings=[new_embedding.tolist()],
                metadatas=[new_metadata] if new_metadata else None,
                documents=[new_text]
            )
            
            return True
        except Exception as e:
            print(f"Error updating embedding: {e}")
            return False