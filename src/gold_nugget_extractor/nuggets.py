"""Nugget save module with deduplication and retry logic."""
import os
import uuid
import hashlib
import time
from pathlib import Path


def save_gold_nugget(book_name: str, chapter_ref: str, content: str, nugget_index: int = 1, max_retries: int = 3) -> str:
    """
    Saves an extracted nugget to the structured output folder with deduplication and retry logic.
    
    Args:
        book_name: The title of the book (used for folder name).
        chapter_ref: The chapter identifier (e.g., 'Chapter 1').
        content: The formatted Markdown content of the nugget.
        nugget_index: The index of this nugget within the chapter (for ordering).
        max_retries: Maximum number of retry attempts for file operations.
    
    Returns:
        Success or failure message.
    """
    # Remove file extension from book name for folder name
    safe_name = Path(book_name).stem.lower().replace(" ", "-")
    directory = Path(f"output-folder/nuggets-of-knowledge/{safe_name}")
    
    directory.mkdir(parents=True, exist_ok=True)
    
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    unique_id = uuid.uuid4().hex[:8]
    # Include nugget index in filename for clarity
    filename = f"{chapter_ref.lower().replace(' ', '_')}_nugget_{nugget_index}_{unique_id}_{content_hash}.md"
    
    full_path = directory / filename
    
    for attempt in range(max_retries):
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully saved nugget to {full_path}"
        except IOError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return f"Failed to save nugget after {max_retries} attempts: {str(e)}"
    
    return f"Failed to save nugget after {max_retries} attempts"


def check_duplicate(book_name: str, content: str) -> bool:
    """
    Check if a nugget with the same content already exists.
    
    Args:
        book_name: The title of the book.
        content: The content to check for duplicates.
    
    Returns:
        True if duplicate exists, False otherwise.
    """
    # Remove file extension from book name for folder name
    safe_name = Path(book_name).stem.lower().replace(" ", "-")
    directory = Path(f"output-folder/nuggets-of-knowledge/{safe_name}")
    
    if not directory.exists():
        return False
    
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    
    for filename in directory.glob("*.md"):
        if content_hash in filename.name:
            return True
    
    return False
