"""
Helper functions for unit testing.
"""

# These are standalone helper functions, not pytest tests
def process_doc(doc_path):
    """Simple test processor function that can be serialized."""
    return f"processed_{doc_path}"

def identity_processor(doc_path):
    """Simple identity function that can be serialized."""
    return doc_path