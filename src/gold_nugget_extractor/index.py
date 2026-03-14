"""Summary index generator module."""
import os
from pathlib import Path


def generate_book_index(book_name: str) -> str:
    """
    Generates an index.md file that lists all nuggets for a book.
    
    Args:
        book_name: The title of the book.
    
    Returns:
        Path to the generated index file.
    """
    # Remove file extension from book name for folder name
    safe_name = Path(book_name).stem.lower().replace(" ", "-")
    directory = Path(f"output-folder/nuggets-of-knowledge/{safe_name}")
    index_path = directory / "index.md"
    
    if not directory.exists():
        return f"Directory not found: {directory}"
    
    nuggets = []
    for filename in sorted(directory.glob("*.md")):
        if filename.name != "index.md":
            # Parse filename to extract chapter info
            # Format: chapter_6_nugget_1_*.md
            parts = filename.stem.split("_")
            if len(parts) >= 2 and parts[1] == "nugget":
                # New format: chapter_X_nugget_Y_*
                if len(parts) >= 3:
                    chapter_num = parts[0].replace("chapter", "").strip()
                    chapter = f"Chapter {chapter_num}"
                else:
                    chapter = filename.stem.replace("_", " ").title()
            elif len(parts) >= 2:
                # Old format: chapter_X_*
                chapter = " ".join(parts[0:2]).replace("_", " ").title()
            else:
                chapter = filename.stem.replace("_", " ").title()
            nuggets.append((chapter, filename.name))
    
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(f"# Gold Nuggets: {book_name}\n\n")
        f.write(f"Total nuggets: {len(nuggets)}\n\n")
        f.write("## Table of Contents\n\n")
        for chapter, filename in nuggets:
            f.write(f"- [{chapter}](./{filename})\n")
    
    return str(index_path)


def generate_book_summary(book_name: str) -> str:
    """
    Generates a consolidated summary document with all nuggets for a book.
    
    Args:
        book_name: The title of the book.
    
    Returns:
        Path to the generated summary file.
    """
    # Remove file extension from book name for folder name
    safe_name = Path(book_name).stem.lower().replace(" ", "-")
    directory = Path(f"output-folder/nuggets-of-knowledge/{safe_name}")
    summary_path = directory / f"{safe_name}_summary.md"
    
    if not directory.exists():
        return f"Directory not found: {directory}"
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Gold Nuggets Summary: {book_name}\n\n")
        
        for filename in sorted(directory.glob("*.md")):
            if filename.name == "index.md":
                continue
            
            with open(filename, "r", encoding="utf-8") as nugget_file:
                content = nugget_file.read()
            
            f.write(f"## {filename.stem}\n\n")
            f.write(content)
            f.write("\n\n---\n\n")
    
    return str(summary_path)


def generate_statistics(book_name: str) -> dict:
    """
    Generates statistics for a processed book.
    
    Args:
        book_name: The title of the book.
    
    Returns:
        Dictionary with statistics.
    """
    # Remove file extension from book name for folder name
    safe_name = Path(book_name).stem.lower().replace(" ", "-")
    directory = Path(f"output-folder/nuggets-of-knowledge/{safe_name}")
    
    if not directory.exists():
        return {"error": f"Directory not found: {directory}"}
    
    nugget_files = list(directory.glob("*.md"))
    index_count = sum(1 for f in nugget_files if f.name == "index.md")
    summary_count = sum(1 for f in nugget_files if f.name.endswith("_summary.md"))
    nugget_count = len(nugget_files) - index_count - summary_count
    
    return {
        "book_name": book_name,
        "total_nuggets": nugget_count,
        "directory": str(directory),
    }
