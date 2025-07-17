class Document:
    """
    A simple document class similar to LangChain's Document.
    Represents a document with content and metadata.
    """
    def __init__(self, page_content: str, metadata: dict = None):
        """
        Initialize a Document.
        
        Args:
            page_content (str): The text content of the document
            metadata (dict, optional): Metadata associated with the document
        """
        self.page_content = page_content
        self.metadata = metadata or {}
