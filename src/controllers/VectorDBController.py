import chromadb
import os
import logging
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
try:
    # Try absolute imports first (for Celery running from project root)
    from src.controllers.BaseController import BaseController
except ImportError:
    # Fall back to relative imports (for FastAPI running from src/)
    from .BaseController import BaseController

logger = logging.getLogger(__name__)

class VectorDBController(BaseController):
    # Class-level cache for shared model and client
    _shared_encoder = None
    _shared_client = None
    _shared_db_dir = None
    
    def __init__(self, collection_name: str = "surveillance_collection", load_encoder: bool = True):
        super().__init__()
        
        # Disable telemetry to avoid errors
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        self.collection_name = collection_name
        self.load_encoder = load_encoder

        # Set up ChromaDB client with persistent storage
        db_dir = os.path.join(self.files_dir, "chromadb")
        os.makedirs(db_dir, exist_ok=True)
        
        self.client = None
        self.encoder = None
        self.collection = None
        
        # Initialize connections (using cached versions when possible)
        self._initialize_client(db_dir)
        if load_encoder:
            self._initialize_encoder()
        self._initialize_collection()
        
    def _initialize_client(self, db_dir: str):
        """Initialize ChromaDB client using cached version when possible"""
        try:
            # Use cached client if available and same directory
            if (VectorDBController._shared_client is not None and 
                VectorDBController._shared_db_dir == db_dir):
                self.client = VectorDBController._shared_client
                logger.info(f"Using cached ChromaDB client for {db_dir}")
                return
            
            # Create new client and cache it
            self.client = chromadb.PersistentClient(path=db_dir)
            VectorDBController._shared_client = self.client
            VectorDBController._shared_db_dir = db_dir
            logger.info(f"ChromaDB client initialized and cached at {db_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            self.client = None
    
    def _initialize_encoder(self):
        """Initialize sentence transformer encoder using cached version when possible"""
        try:
            # Use cached encoder if available
            if VectorDBController._shared_encoder is not None:
                self.encoder = VectorDBController._shared_encoder
                logger.info("Using cached sentence transformer encoder")
                return
            
            # Load new encoder and cache it
            logger.info("Loading sentence transformer encoder (this may take a moment)...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            VectorDBController._shared_encoder = self.encoder
            logger.info("Sentence transformer encoder loaded successfully and cached")
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
    
    def switch_collection(self, collection_name: str) -> bool:
        """
        Switch to a different collection without reloading the sentence transformer model.
        This is much more efficient than creating a new VectorDBController instance.
        
        Args:
            collection_name (str): Name of the collection to switch to
            
        Returns:
            bool: Success status
        """
        if not self.client:
            logger.error("Cannot switch collection: ChromaDB client not available")
            return False
            
        try:
            old_collection = self.collection_name
            self.collection_name = collection_name
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Switched from collection '{old_collection}' to '{collection_name}'")
            return True
        except Exception as e:
            logger.warning(f"Collection '{collection_name}' not found or cannot be accessed: {e}")
            return False
        
    def get_collection_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """
        Get basic stats for a collection without loading the full data.
        
        Args:
            collection_name (str, optional): Collection to get stats for. Uses current if None.
            
        Returns:
            dict: Collection statistics
        """
        target_collection = collection_name or self.collection_name
        
        if not self.client:
            return {"error": "ChromaDB client not available", "count": 0}
            
        try:
            collection = self.client.get_collection(target_collection)
            count = collection.count()
            return {
                "name": target_collection,
                "count": count,
                "available": True
            }
        except Exception as e:
            logger.warning(f"Cannot get stats for collection '{target_collection}': {e}")
            return {
                "name": target_collection,
                "count": 0,
                "available": False,
                "error": str(e)
            }
    
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
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the vector database
        
        Returns:
            dict: Health status and diagnostic information
        """
        try:
            health_status = {
                'status': 'healthy',
                'collection_name': self.collection_name,
                'available': self.is_available()
            }
            
            if self.is_available():
                # Test basic functionality
                try:
                    # Get collection stats
                    stats = self.get_collection_stats()
                    health_status.update(stats)
                    
                    # Test client connection
                    heartbeat = self.client.heartbeat()
                    health_status['heartbeat'] = heartbeat
                    
                except Exception as e:
                    health_status['status'] = 'warning'
                    health_status['warning'] = f"Basic operations failed: {e}"
            else:
                health_status['status'] = 'unhealthy'
                health_status['error'] = 'Vector database components not available'
                
            return health_status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'collection_name': self.collection_name,
                'available': False
            }

    def cleanup(self):
        """Clean up resources"""
        try:
            # ChromaDB doesn't require explicit cleanup, but we can log it
            self.logger.info("Vector database resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def ensure_encoder_loaded(self) -> bool:
        """
        Ensure the sentence transformer encoder is loaded.
        Call this before operations that require embeddings.
        
        Returns:
            bool: True if encoder is available
        """
        if self.encoder is not None:
            return True
            
        if not self.load_encoder:
            logger.warning("Encoder loading is disabled for this controller instance")
            return False
            
        self._initialize_encoder()
        return self.encoder is not None