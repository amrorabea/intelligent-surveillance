try:
    # Try absolute imports first (for Celery running from project root)
    from src.controllers.BaseController import BaseController
    from src.controllers.VectorDBController import VectorDBController
except ImportError:
    # Fall back to relative imports (for FastAPI running from src/)
    from .BaseController import BaseController
    from .VectorDBController import VectorDBController
import re
import os
import logging

logger = logging.getLogger(__name__)

class QueryController(BaseController):
    def __init__(self, project_id: str = None, vector_controller: VectorDBController = None):
        """Initialize query controller with vector database connection
        
        Args:
            project_id (str): Default project ID for filtering (optional)
            vector_controller (VectorDBController): Pre-initialized vector controller
                If None, creates a new one (for backward compatibility)
        """
        super().__init__()

        # Handle None or empty project_id
        if not project_id:
            project_id = "default"
        
        self.project_id = project_id
        
        # Use provided vector controller or create new one
        if vector_controller is not None:
            # Use the pre-initialized global vector controller
            self.vector_db = vector_controller
            logger.info(f"Using pre-initialized global vector controller for project {project_id}")
            
            # But we need to switch to the project-specific collection
            if project_id and project_id != "default":
                collection_name = f'surveillance_{project_id}'
                try:
                    # Switch to the project-specific collection
                    import chromadb
                    self.vector_db.collection = self.vector_db.client.get_collection(name=collection_name)
                    logger.info(f"Switched to project collection: {collection_name}")
                except Exception as e:
                    logger.warning(f"Could not switch to collection {collection_name}: {e}")
                    # Fall back to creating a new project-specific controller
                    self.vector_db = VectorDBController(collection_name=collection_name)
                    logger.info(f"Created new project-specific controller for {collection_name}")
        else:
            # Fallback: create project-specific controller (for backward compatibility)
            self.collection_name = f'surveillance_{project_id}'
            self.vector_db = VectorDBController(collection_name=self.collection_name)
            logger.info(f"Created new project-specific vector controller for {project_id}")
        
    def process_query(self, query_text, max_results=10, project_id=None):
        """
        Process natural language query about surveillance footage
        
        Args:
            query_text (str): The natural language query
            max_results (int): Maximum number of results to return
            project_id (str, optional): Filter results to specific project
                If None, uses the default project_id from initialization
                
        Returns:
            dict: Search results with matched frames and metadata
        """
        # Use provided project_id or fall back to instance default
        effective_project_id = project_id or self.project_id
        
        # DEBUG: Check what's in the vector database first
        try:
            total_count = self.vector_db.collection.count()
            logger.info(f"🔍 Total documents in vector database: {total_count}")
            
            # Get a sample of documents to see what project_ids exist
            sample_docs = self.vector_db.collection.get(limit=5, include=['metadatas'])
            if sample_docs and sample_docs.get('metadatas'):
                project_ids = [meta.get('project_id', 'unknown') for meta in sample_docs['metadatas']]
                logger.info(f"🔍 Sample project_ids in database: {set(project_ids)}")
            
        except Exception as e:
            logger.warning(f"Could not get database stats: {e}")
        
        # Analyze query for potential filters
        filters = self.extract_filters_from_query(query_text, effective_project_id)
        logger.info(f"🔍 Using filters: {filters}")
        
        # First try without filters to see if we get any results
        search_results_no_filter = self.vector_db.semantic_search(
            query=query_text, 
            limit=max_results,
            filter_criteria=None  # No filters
        )
        logger.info(f"🔍 Search results WITHOUT filters: {len(search_results_no_filter)} results")
        
        # Now try with filters
        search_results = self.vector_db.semantic_search(
            query=query_text, 
            limit=max_results,
            filter_criteria=filters
        )
        
        logger.info(f"🔍 Search results WITH filters for query '{query_text}' in project '{effective_project_id}': {len(search_results)} results")
        
        # If filtered search returns no results, use unfiltered results but log the issue
        if len(search_results) == 0 and len(search_results_no_filter) > 0:
            logger.warning(f"🔍 Filtered search returned 0 results, but unfiltered returned {len(search_results_no_filter)}. Using unfiltered results.")
            search_results = search_results_no_filter
            # Update effective_project_id to indicate we're showing all projects
            effective_project_id = "all_projects"
        
        # Extract frame paths and timestamps
        processed_results = []
        for result in search_results:
            metadata = result.get('metadata', {})
            
            # Calculate similarity score (ChromaDB uses distance, lower = more similar)
            distance = result.get('distance', 1.0)
            similarity_score = max(0.0, 1.0 - distance) if distance is not None else 0.0
            
            # Handle detected_objects - ensure it's a list
            detected_objects = metadata.get('detected_objects', [])
            if isinstance(detected_objects, str):
                # If it's a string, split it into a list
                if detected_objects:
                    detected_objects = [obj.strip() for obj in detected_objects.split(',') if obj.strip()]
                else:
                    detected_objects = []
            elif not isinstance(detected_objects, list):
                # If it's neither string nor list, make it an empty list
                detected_objects = []
            
            # Ensure all metadata values are properly typed
            processed_results.append({
                'id': str(result['id']),
                'caption': str(result['document']) if result.get('document') else 'No caption available',
                'file_id': str(metadata.get('file_id', 'unknown')),
                'video_filename': str(metadata.get('video_filename', metadata.get('filename', 'unknown'))),
                'project_id': str(metadata.get('project_id', 'unknown')),
                'timestamp': float(metadata.get('timestamp', 0)),
                'frame_number': str(metadata.get('frame_number', 'unknown')),
                'frame_path': str(metadata.get('frame_path', '')) if metadata.get('frame_path') else '',
                'detected_objects': detected_objects,
                'score': float(similarity_score)
            })
            
        return {
            'query': query_text,
            'results': processed_results,
            'total_results': len(processed_results),
            'project_id': effective_project_id
        }
        
    def extract_filters_from_query(self, query, project_id=None):
        """
        Extract filter conditions from natural language query
        
        Args:
            query (str): The natural language query
            project_id (str, optional): Project ID to filter by
            
        Returns:
            dict: Filter conditions for vector search
        """
        filters = {}
        
        # NOTE: Don't add project_id filter here since we're already in the correct collection
        # The QueryController should be initialized with the right collection for the project
        logger.info("🔍 Not adding project filter since we're in project-specific collection")
            
        # Extract time-based filters
        time_patterns = [
            (r'after (\d{1,2}:\d{2})', 'min_time'),
            (r'before (\d{1,2}:\d{2})', 'max_time'),
            (r'at night', 'night'),
            (r'after midnight', 'after_midnight'),
            (r'during day', 'day'),
            (r'today', 'today'),
            (r'yesterday', 'yesterday')
        ]
        
        for pattern, filter_type in time_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                if filter_type == 'min_time' or filter_type == 'max_time':
                    time_str = re.search(pattern, query, re.IGNORECASE).group(1)
                    # Convert time string to seconds for comparison
                    hours, minutes = map(int, time_str.split(':'))
                    time_seconds = hours * 3600 + minutes * 60
                    
                    if filter_type == 'min_time':
                        filters['timestamp'] = {'$gte': time_seconds}
                    else:
                        filters['timestamp'] = {'$lte': time_seconds}
        
        return filters
        
    def get_frame_for_result(self, result_id):
        """
        Retrieve the actual frame image path for a search result
        
        Args:
            result_id (str): Result ID from search results
            
        Returns:
            str: Path to the frame image file, or None if not found
        """
        try:
            # Try to get the document by ID from vector database
            all_docs = self.vector_db.collection.get(
                ids=[result_id],
                include=['metadatas']
            )
            
            if all_docs and all_docs['ids'] and len(all_docs['ids']) > 0:
                metadata = all_docs['metadatas'][0] if all_docs['metadatas'] else {}
                frame_path = metadata.get('frame_path')
                
                logger.info(f"Found frame path for {result_id}: {frame_path}")
                
                # Try the direct path first
                if frame_path and os.path.exists(frame_path):
                    return frame_path
                
                # If direct path doesn't work, try to construct alternative paths
                if frame_path:
                    # Extract filename from path
                    frame_filename = os.path.basename(frame_path)
                    
                    # Try common locations
                    base_dirs = [
                        "/home/amro/Desktop/intelligent-surveillance/src/assets/files",
                        "/home/amro/Desktop/intelligent-surveillance/assets/files",
                        os.path.dirname(frame_path)  # Try original directory
                    ]
                    
                    project_id = metadata.get('project_id', 'test')
                    
                    for base_dir in base_dirs:
                        potential_paths = [
                            os.path.join(base_dir, project_id, "frames", frame_filename),
                            os.path.join(base_dir, "test", "frames", frame_filename),
                            os.path.join(base_dir, "frames", frame_filename),
                        ]
                        
                        for path in potential_paths:
                            if os.path.exists(path):
                                logger.info(f"Found alternative frame path: {path}")
                                return path
                
                logger.warning(f"Frame path does not exist: {frame_path}")
            else:
                logger.warning(f"No metadata found for result ID: {result_id}")
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting frame for result {result_id}: {e}")
            return None