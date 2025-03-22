"""
File utility functions for JFKReveal.

This module provides standardized file operations used throughout the JFKReveal project.
It centralizes common file handling code to reduce duplication, improve consistency,
and make maintenance easier.
"""
import os
import re
import glob
import logging
from typing import List, Optional, Tuple, Dict, Any, Set
from pathlib import Path

logger = logging.getLogger(__name__)

def ensure_directory_exists(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
        
    Returns:
        The directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Created directory: {directory}")
    return directory

def list_files(
    directory: str, 
    pattern: str = "*", 
    recursive: bool = False
) -> List[str]:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match (e.g., "*.pdf", "**/*.json")
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    full_pattern = os.path.join(directory, pattern)
    
    if recursive:
        # Add ** to pattern if it doesn't already have it and recursive is True
        if "**" not in pattern:
            full_pattern = os.path.join(directory, "**", pattern)
    
    matching_files = glob.glob(full_pattern, recursive=recursive)
    matching_files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time
    
    logger.debug(f"Found {len(matching_files)} files matching {pattern} in {directory}")
    return matching_files

def list_pdf_files(directory: str, recursive: bool = True) -> List[str]:
    """
    List all PDF files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of PDF file paths
    """
    return list_files(directory, "*.pdf", recursive)

def list_json_files(directory: str, recursive: bool = True) -> List[str]:
    """
    List all JSON files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of JSON file paths
    """
    return list_files(directory, "*.json", recursive)

def get_file_extension(file_path: str) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (lowercase, without the dot)
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower().lstrip(".")

def is_pdf_file(file_path: str) -> bool:
    """
    Check if a file is a PDF.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a PDF, False otherwise
    """
    return get_file_extension(file_path) == "pdf"

def is_json_file(file_path: str) -> bool:
    """
    Check if a file is a JSON file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a JSON file, False otherwise
    """
    return get_file_extension(file_path) == "json"

def get_output_path(
    input_path: str, 
    output_dir: str, 
    extension: str = None,
    create_dirs: bool = True
) -> str:
    """
    Get the output path for a file, preserving its name but changing its location and extension.
    
    Args:
        input_path: Path to the input file
        output_dir: Directory for the output file
        extension: New extension (without the dot) or None to keep the original
        create_dirs: Whether to create the output directory if it doesn't exist
        
    Returns:
        Path to the output file
    """
    if create_dirs:
        ensure_directory_exists(output_dir)
    
    # Get the filename without extension
    filename = os.path.basename(input_path)
    basename, ext = os.path.splitext(filename)
    
    # Use the provided extension or keep the original
    if extension:
        new_ext = f".{extension.lstrip('.')}"
    else:
        new_ext = ext
    
    return os.path.join(output_dir, f"{basename}{new_ext}")

def clean_filename(filename: str) -> str:
    """
    Clean a filename to remove invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Replace invalid characters with underscores
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes or 0 if the file doesn't exist
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    return 0

def get_file_hash(file_path: str, algorithm: str = "md5") -> Optional[str]:
    """
    Calculate the hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (md5, sha1, sha256)
        
    Returns:
        File hash or None if the file doesn't exist
    """
    if not os.path.exists(file_path):
        return None
    
    import hashlib
    
    hash_algorithms = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256
    }
    
    if algorithm not in hash_algorithms:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    hasher = hash_algorithms[algorithm]()
    
    with open(file_path, "rb") as f:
        # Read in chunks to avoid loading large files into memory
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()

def get_document_id(file_path: str) -> str:
    """
    Generate a consistent document ID for a file based on its name.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Document ID (MD5 hash of the filename)
    """
    filename = os.path.basename(file_path)
    return hashlib.md5(filename.encode()).hexdigest()

def check_if_embedded(file_path: str) -> bool:
    """
    Check if a document has already been embedded in the vector store.
    
    Args:
        file_path: Path to the processed document file
        
    Returns:
        True if the document has been embedded, False otherwise
    """
    marker_path = f"{file_path}.embedded"
    return os.path.exists(marker_path)

def mark_as_embedded(file_path: str) -> None:
    """
    Mark a document as embedded in the vector store.
    
    Args:
        file_path: Path to the processed document file
    """
    marker_path = f"{file_path}.embedded"
    with open(marker_path, 'w') as f:
        f.write("1")

def find_files_by_content(
    directory: str, 
    pattern: str, 
    file_pattern: str = "*", 
    recursive: bool = True
) -> List[Tuple[str, List[str]]]:
    """
    Find files containing a specific pattern in their content.
    
    Args:
        directory: Directory to search
        pattern: Regular expression pattern to search for
        file_pattern: Glob pattern to filter files
        recursive: Whether to search recursively
        
    Returns:
        List of tuples (file_path, matching_lines)
    """
    results = []
    files = list_files(directory, file_pattern, recursive)
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                matches = re.findall(pattern, content)
                if matches:
                    results.append((file_path, matches))
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {str(e)}")
    
    return results

def read_file_content(file_path: str) -> Optional[str]:
    """
    Read the content of a text file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content or None if the file doesn't exist or can't be read
    """
    if not os.path.exists(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return None

def write_file_content(file_path: str, content: str, create_dirs: bool = True) -> bool:
    """
    Write content to a text file.
    
    Args:
        file_path: Path to the file
        content: Content to write
        create_dirs: Whether to create parent directories if they don't exist
        
    Returns:
        True if the file was written successfully, False otherwise
    """
    if create_dirs:
        directory = os.path.dirname(file_path)
        ensure_directory_exists(directory)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.debug(f"Wrote {len(content)} characters to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing to {file_path}: {str(e)}")
        return False

def is_directory_empty(directory: str) -> bool:
    """
    Check if a directory is empty.
    
    Args:
        directory: Directory to check
        
    Returns:
        True if the directory is empty or doesn't exist, False otherwise
    """
    if not os.path.exists(directory):
        return True
    return not os.listdir(directory)