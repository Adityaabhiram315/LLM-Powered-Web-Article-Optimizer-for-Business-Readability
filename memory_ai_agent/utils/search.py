import time
from typing import List, Dict, Any
from duckduckgo_search import DDGS

class SearchTool:
    def __init__(self):
        """Initialize search tool."""
        self.ddgs = DDGS()
        
    def search(self, query: str, max_results: int = 5) -> tuple[List[Dict[str, Any]], float]:
        """Search the web using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Tuple of search results and time taken
        """
        start_time = time.time()
        
        try:
            results = list(self.ddgs.text(query, max_results=max_results))
            end_time = time.time()
            return results, end_time - start_time
        except Exception as e:
            end_time = time.time()
            return [{"title": f"Error searching: {str(e)}", "body": "", "href": ""}], end_time - start_time
            
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for the AI.
        
        Args:
            results: Search results
            
        Returns:
            Formatted search results
        """
        if not results:
            return "No search results found."
            
        formatted = "Search results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result.get('title', 'No title')}\n"
            formatted += f"{result.get('body', 'No description')[:200]}...\n"
            formatted += f"Source: {result.get('href', 'No link')}\n\n"
            
        return formatted