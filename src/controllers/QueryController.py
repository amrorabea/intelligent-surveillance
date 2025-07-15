from .BaseController import BaseController
from .VectorDBController import VectorDBController
import re
import os

class QueryController(BaseController):
    def __init__(self):
        """Initialize query controller with vector database connection"""
        super().__init__()
        self.vector_db = VectorDBController(collection_name="surveillance_data")
        
    def process_query(self, query_text, max_results=10, project_id=None):
        """
        Process natural language query about surveillance footage
        
        Args:
            query_text (str): The natural language query
            max_results (int): Maximum number of results to return
            project_id (str, optional): Filter results to specific project
            
        Returns:
            dict: Search results with matched frames and metadata
        """
        # Analyze query for potential filters
        filters = self.extract_filters_from_query(query_text, project_id)
        
        # Search for relevant content
        search_results = self.vector_db.semantic_search(
            query=query_text, 
            limit=max_results,
            filter_criteria=filters
        )
        
        # Extract frame paths and timestamps
        processed_results = []
        for result in search_results:
            metadata = result.get('metadata', {})
            
            processed_results.append({
                'id': result['id'],
                'caption': result['document'],
                'file_id': metadata.get('file_id', 'unknown'),
                'timestamp': metadata.get('timestamp', 0),
                'frame_path': metadata.get('frame_path', None),
                'score': 1.0 - (result.get('distance', 0) if result.get('distance') else 0)
            })
            
        return {
            'query': query_text,
            'results': processed_results,
            'total_results': len(processed_results)
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
        
        # Add project filter if specified
        if project_id:
            filters['project_id'] = project_id
            
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
        Retrieve the actual frame image for a search result
        
        Args:
            result_id (str): Result ID from search results
            
        Returns:
            str: Path to the frame image file
        """
        # Get the detailed result from vector DB
        results = self.vector_db.semantic_search(
            query="",  # Empty query for exact ID match
            filter_criteria={"id": result_id},
            limit=1
        )
        
        if results and len(results) > 0:
            metadata = results[0].get('metadata', {})
            frame_path = metadata.get('frame_path')
            
            if frame_path and os.path.exists(frame_path):
                return frame_path
                
        return None