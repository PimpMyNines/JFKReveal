import unittest
from jfkreveal.database.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor(
            input_dir="data/raw",
            output_dir="data/processed",
            batch_size=10
        )
    
    def test_init(self):
        """Test the initialization of DocumentProcessor."""
        self.assertEqual(self.processor.input_dir, "data/raw")
        self.assertEqual(self.processor.output_dir, "data/processed")
        self.assertEqual(self.processor.batch_size, 10)

if __name__ == '__main__':
    unittest.main()