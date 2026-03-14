"""Main OpenAI Agents SDK Agent for Gold Nugget Extraction."""
import os
import json
import time
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

from agents import Agent, Runner, set_tracing_disabled
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

# Disable tracing to avoid OpenAI API key errors
set_tracing_disabled(True)

from gold_nugget_extractor.state import (
    load_state,
    save_state,
    get_processed_chapters,
    mark_chapter_processed,
    get_book_state,
    get_all_books,
)
from gold_nugget_extractor.nuggets import save_gold_nugget, check_duplicate
from gold_nugget_extractor.index import generate_book_index, generate_book_summary


# Vector Database Configuration
VECTOR_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")


# Extractor Prompt for the LLM
EXTRACTOR_PROMPT = """
Role: You are a Knowledge Curator. Your goal is to find "Gold Nuggets" (deep insights or core concepts) from the provided text.

Parameters:
- min_nuggets: Minimum number of nuggets to extract (default: 1)
- max_nuggets: Maximum number of nuggets to extract (default: 15)

For every insight found, output exactly in this format:

"[Insert Quote/Concept here]"

    "[Reference: Book Name, Chapter, Paragraph]"

[Explanation of importance: 3-4 sentences]
[Final thoughts: 2-3 sentences]

Constraint: Do not add conversational filler. Output only the formatted nugget.
""".strip()


class VectorDBClient:
    """Client for interacting with the ChromaDB vector database."""
    
    def __init__(self, db_path: str, collection_name: str = COLLECTION_NAME):
        import chromadb
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
    def list_documents(self) -> dict:
        """List all documents in the vector database."""
        try:
            result = self.collection.get(limit=100)
            documents = []
            if result['metadatas']:
                for meta in result['metadatas']:
                    documents.append({
                        'source': meta.get('source', 'unknown'),
                        'id': meta.get('id', 'unknown')
                    })
            return {'documents': documents}
        except Exception as e:
            return {'error': str(e)}
    
    def semantic_search(self, query: str, top_k: int = 5, document_filter: Optional[str] = None) -> dict:
        """Search using semantic similarity."""
        try:
            where = None
            if document_filter:
                where = {'source': {'$eq': document_filter}}
            
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where
            )
            
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'text': doc,
                        'score': results['distances'][0][i] if results['distances'] else None,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else None
                    })
            
            return {'results': formatted_results}
        except Exception as e:
            return {'error': str(e)}
    
    def keyword_search(self, keywords: str, top_k: int = 5) -> dict:
        """Search using keyword matching (simulated with semantic search)."""
        # ChromaDB doesn't have native keyword search, so we use semantic search
        # with the keywords as the query
        return self.semantic_search(keywords, top_k)
    
    def get_document_info(self, document_id: str = None, filename: str = None, limit: int = 100) -> dict:
        """Get information about a document."""
        try:
            where = {}
            if document_id:
                where['id'] = document_id
            elif filename:
                where['source'] = filename
            
            result = self.collection.get(where=where, limit=limit)
            
            if result['ids']:
                documents = []
                for i, doc_id in enumerate(result['ids']):
                    documents.append({
                        'id': doc_id,
                        'source': result['metadatas'][i].get('source', 'unknown') if result['metadatas'] else 'unknown',
                        'content': result['documents'][i] if result['documents'] else None,
                        'metadata': result['metadatas'][i] if result['metadatas'] else None
                    })
                return {'documents': documents}
            else:
                return {'error': 'Document not found'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_chapter(self, document_id: str, chapter: int) -> dict:
        """Get a specific chapter from a document."""
        # ChromaDB doesn't have chapter information stored directly
        # We'll return a placeholder and rely on semantic search results
        return {'content': f'Chapter {chapter} content for document {document_id}'}
    
    def get_page(self, document_id: str, page: int) -> dict:
        """Get a specific page from a document."""
        return {'content': f'Page {page} content for document {document_id}'}


def create_openrouter_client() -> AsyncOpenAI:
    """Create an OpenAI-compatible client for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def create_gold_nugget_agent(model: str = None) -> Agent:
    """Create an OpenAI Agents SDK agent for gold nugget extraction."""
    if model is None:
        model = os.getenv("LLM_MODEL", "openai/gpt-4o")
    
    # OpenRouter models use the format provider/model:variant
    # The OpenAI Agents SDK validates model names, so we need to use a valid prefix
    # OpenRouter supports models like openai/gpt-4o, anthropic/claude-3-5-sonnet, etc.
    return Agent(
        name="gold_nugget_extractor",
        instructions=EXTRACTOR_PROMPT,
        model=model,
    )


class GoldNuggetExtractor:
    """Main class for extracting gold nuggets from books using OpenAI Agents SDK."""
    
    def __init__(self, book_name: str, model: str = None, db_client: VectorDBClient = None):
        self.book_name = book_name
        self.model = model or os.getenv("LLM_MODEL", "google/gemini-3-flash:free")
        self.output_dir = Path("output-folder/nuggets-of-knowledge")
        self.state_file = Path("output-folder/processing-state.json")
        self.db_client = db_client or VectorDBClient(VECTOR_DB_PATH)
        # Set the default OpenAI client for the SDK
        self.client = create_openrouter_client()
        from agents import set_default_openai_client
        set_default_openai_client(self.client)
        
    def get_table_of_contents(self) -> list:
        """Get the table of contents for the book using semantic search."""
        # Search for the book title to get document info
        doc_info = self.db_client.get_document_info(filename=self.book_name)
        
        if "error" in doc_info:
            # Try semantic search as fallback - search for chapter-related content
            search_result = self.db_client.semantic_search(f"chapter one start {self.book_name}", top_k=10)
            if "results" in search_result and search_result["results"]:
                # Extract unique chapter numbers from results
                chapters = []
                seen_chapters = set()
                for i, result in enumerate(search_result["results"][:10]):
                    # Try to extract chapter number from metadata or content
                    chapter_num = i + 1
                    chapter_ref = f"Chapter {chapter_num}"
                    if chapter_ref not in seen_chapters:
                        chapters.append(chapter_ref)
                        seen_chapters.add(chapter_ref)
                return chapters
            return ["Chapter 1", "Chapter 2", "Chapter 3"]
        
        # Get table of contents from document info
        documents = doc_info.get("documents", [])
        if documents:
            # Return chapter names based on document content
            return [f"Chapter {i+1}" for i in range(min(len(documents), 10))]
        
        # Default fallback
        return ["Chapter 1", "Chapter 2", "Chapter 3"]
    
    def get_chapter_content(self, chapter_ref: str) -> str:
        """Get the content of a specific chapter using keyword search."""
        # Extract chapter number from chapter_ref (e.g., "Chapter 1" -> 1)
        chapter_num = 0
        for part in chapter_ref.split():
            if part.isdigit():
                chapter_num = int(part)
                break
        
        # First, try to get document info with the book name
        doc_info = self.db_client.get_document_info(filename=self.book_name, limit=100)
        
        if "documents" in doc_info:
            # Filter documents by chapter number from metadata
            chapter_docs = []
            for doc in doc_info["documents"]:
                metadata = doc.get("metadata", {})
                if metadata:
                    # Check if metadata has chapter info
                    doc_chapter = metadata.get("chapter") or metadata.get("page")
                    if doc_chapter == chapter_num:
                        chapter_docs.append(doc)
            
            if chapter_docs:
                # Combine content from matching chapters
                content_parts = []
                seen_texts = set()
                for doc in chapter_docs:
                    text = doc.get("content", "")
                    if text and text not in seen_texts:
                        content_parts.append(text)
                        seen_texts.add(text)
                
                if content_parts:
                    return "\n\n".join(content_parts)
        
        # Fallback: use semantic search with chapter filter
        search_result = self.db_client.semantic_search(f"{self.book_name} {chapter_ref} content", top_k=5, document_filter=self.book_name)
        
        if "results" in search_result and search_result["results"]:
            # Combine the top results, prioritizing unique content
            content_parts = []
            seen_texts = set()
            for result in search_result["results"][:5]:
                text = result.get("text", "")
                if text and text not in seen_texts:
                    content_parts.append(text)
                    seen_texts.add(text)
            
            if content_parts:
                return "\n\n".join(content_parts)
        
        # Fallback: return a more descriptive placeholder
        return f"Content for {chapter_ref} from {self.book_name}"
    
    def extract_nuggets(self, chapter_ref: str, chapter_content: str) -> list:
        """Extract gold nuggets from chapter content using the LLM."""
        prompt = f"""
Extract gold nuggets from the following chapter content.

Book: {self.book_name}
Chapter: {chapter_ref}

Content:
{chapter_content}

Please extract 1-15 gold nuggets (key insights or core concepts) from this chapter.
Format each nugget as a JSON object with these fields:
- quote: The key insight or quote
- reference: {self.book_name}, {chapter_ref}
- explanation: [3-4 sentences explaining importance]
- final_thoughts: [2-3 sentences]

Return the result as a JSON array of objects.
"""
        
        agent = create_gold_nugget_agent(self.model)
        
        try:
            result = Runner.run_sync(agent, prompt)
            response = result.final_output
            
            # Parse the response - it might be a string or a list
            if isinstance(response, str):
                # Try to parse as JSON first
                try:
                    nuggets = json.loads(response)
                    if isinstance(nuggets, list):
                        return nuggets
                    elif isinstance(nuggets, dict):
                        return [nuggets]
                except json.JSONDecodeError:
                    # If not JSON, try to parse multiple nuggets from text
                    # Look for patterns like "[Quote]" followed by "Reference:" and "Explanation:"
                    import re
                    nuggets = []
                    
                    # Split by the quote pattern (lines starting with [ or containing quotes)
                    # Look for nugget patterns: "[Quote]" followed by Reference, Explanation, Final thoughts
                    # Improved pattern to handle various formats
                    # Pattern: "Quote" followed by Reference, Explanation, Final thoughts
                    # Handle format: # "Quote" > "Quote" *Reference: ... ## Explanation Explanation: ... Final thoughts: ...
                    pattern = r'#\s*"([^"]+)"\s*\n\s*>\s*"([^"]+)"\s*\n\s*\*Reference:\s*(.+?)\s*\n\s*##\s*Explanation\s*\n\s*Explanation:\s*(.+?)\s*\n\s*Final thoughts:\s*(.+?)(?=\n\s*---\s*\n\s*#|$)'
                    
                    matches = re.findall(pattern, response, re.DOTALL)
                    
                    # If no matches, try alternative pattern
                    if not matches:
                        alt_pattern = r'#\s*"([^"]+)"\s*\n\s*>\s*"([^"]+)"\s*\n\s*\*Reference:\s*(.+?)\s*\n\s*##\s*Explanation\s*\n\s*Explanation:\s*(.+?)\s*\n\s*Final thoughts:\s*(.+?)(?=\n\s*---\s*\n\s*#|$)'
                        matches = re.findall(alt_pattern, response, re.DOTALL)
                    
                    # If still no matches, try simpler pattern
                    if not matches:
                        simple_pattern = r'#\s*"([^"]+)"\s*\n\s*>\s*"([^"]+)"\s*\n\s*\*Reference:\s*(.+?)\s*\n\s*##\s*Explanation\s*\n\s*Explanation:\s*(.+?)\s*\n\s*Final thoughts:\s*(.+?)(?=\n\s*---\s*\n\s*#|$)'
                        matches = re.findall(simple_pattern, response, re.DOTALL)
                    if matches:
                        for match in matches:
                            nuggets.append({
                                "quote": match[0].strip(),
                                "reference": match[1].strip(),
                                "explanation": match[2].strip(),
                                "final_thoughts": match[3].strip()
                            })
                    
                    if nuggets:
                        return nuggets
                    
                    # If no pattern matches, return as a single nugget
                    return [{
                        "quote": response[:100],
                        "reference": f"{self.book_name}, {chapter_ref}",
                        "explanation": response,
                        "final_thoughts": "Parsed from text response."
                    }]
            elif isinstance(response, list):
                return response
            else:
                return [{
                    "quote": str(response),
                    "reference": f"{self.book_name}, {chapter_ref}",
                    "explanation": "Response format not recognized.",
                    "final_thoughts": "Please review manually."
                }]
        except Exception as e:
            print(f"Error extracting nuggets: {e}")
            return [{
                "quote": "Error extracting insight",
                "reference": f"{self.book_name}, {chapter_ref}",
                "explanation": f"Error: {str(e)}",
                "final_thoughts": "Please review manually."
            }]
    
    def process_chapter(self, chapter_ref: str) -> int:
        """Process a single chapter and extract nuggets."""
        print(f"Processing chapter: {chapter_ref}")
        
        # Get chapter content
        chapter_content = self.get_chapter_content(chapter_ref)
        
        # Extract nuggets
        nuggets = self.extract_nuggets(chapter_ref, chapter_content)
        
        # Save each nugget
        for i, nugget in enumerate(nuggets, start=1):
            content = self.format_nugget(nugget, chapter_ref)
            result = save_gold_nugget(self.book_name, chapter_ref, content, nugget_index=i)
            print(f"  {result}")
        
        # Update state
        mark_chapter_processed(self.book_name, chapter_ref, len(nuggets))
        print(f"  Marked {chapter_ref} as processed with {len(nuggets)} nuggets")
        
        return len(nuggets)
    
    def format_nugget(self, nugget: dict, chapter_ref: str) -> str:
        """Format a nugget as Markdown."""
        return f"""# {nugget.get('quote', 'Insight')}

> "{nugget.get('quote', 'Insight')}"

*Reference: {nugget.get('reference', chapter_ref)}*

## Explanation

{nugget.get('explanation', 'No explanation provided.')}

## Final Thoughts

{nugget.get('final_thoughts', 'No final thoughts provided.')}

---
"""
    
    def run(self) -> dict:
        """Run the gold nugget extraction process."""
        print(f"Starting gold nugget extraction for: {self.book_name}")
        
        # Get table of contents
        chapters = self.get_table_of_contents()
        print(f"Found {len(chapters)} chapters")
        
        # Process each chapter
        total_nuggets = 0
        for chapter in chapters:
            # Check if already processed
            processed = get_processed_chapters(self.book_name)
            if chapter in processed:
                print(f"Skipping already processed: {chapter}")
                continue
            
            # Process chapter
            nugget_count = self.process_chapter(chapter)
            total_nuggets += nugget_count
        
        # Generate index
        index_path = generate_book_index(self.book_name)
        print(f"Generated index: {index_path}")
        
        # Generate summary
        summary_path = generate_book_summary(self.book_name)
        print(f"Generated summary: {summary_path}")
        
        return {
            "book_name": self.book_name,
            "total_chapters": len(chapters),
            "total_nuggets": total_nuggets,
            "index_path": index_path,
            "summary_path": summary_path,
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gold Nugget Extractor")
    parser.add_argument("book_name", help="Name of the book to process")
    parser.add_argument("--model", default=None, help="LLM model to use (OpenRouter format)")
    parser.add_argument("--output-dir", default="output-folder", help="Output directory")
    args = parser.parse_args()
    
    # Set output directory
    import gold_nugget_extractor.state as state_module
    state_module.STATE_FILE = Path(args.output_dir) / "processing-state.json"
    
    # Create DB client
    db_client = VectorDBClient(VECTOR_DB_PATH)
    
    # Run extraction
    extractor = GoldNuggetExtractor(args.book_name, args.model, db_client)
    result = extractor.run()
    
    print(f"\nExtraction complete!")
    print(f"Book: {result['book_name']}")
    print(f"Total chapters: {result['total_chapters']}")
    print(f"Total nuggets: {result['total_nuggets']}")
    print(f"Index: {result['index_path']}")
    print(f"Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
