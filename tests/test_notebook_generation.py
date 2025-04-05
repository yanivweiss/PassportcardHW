import os
import json
import unittest
import nbformat
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestNotebookGeneration(unittest.TestCase):
    """Test cases for notebook generation scripts"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.notebooks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks')
        self.expected_notebooks = [
            'PassportCard_Insurance_Claims_Prediction.ipynb',
            'PassportCard_Model_Development.ipynb',
            'PassportCard_Business_Applications.ipynb',
        ]
        self.min_notebook_size = 1000  # Expected minimum size in bytes
    
    def test_notebook_content_size(self):
        """Test that notebooks have sufficient content"""
        for notebook in self.expected_notebooks:
            notebook_path = os.path.join(self.notebooks_dir, notebook)
            if os.path.exists(notebook_path):
                self.assertGreater(os.path.getsize(notebook_path), self.min_notebook_size,
                                  f"Notebook {notebook} is too small (less than {self.min_notebook_size} bytes)")
    
    def test_notebook_json_structure(self):
        """Test that notebooks have valid JSON structure"""
        for notebook in self.expected_notebooks:
            notebook_path = os.path.join(self.notebooks_dir, notebook)
            if os.path.exists(notebook_path):
                try:
                    # Try to load with nbformat
                    nb = nbformat.read(notebook_path, as_version=4)
                    self.assertIsNotNone(nb, f"Notebook {notebook} could not be parsed by nbformat")
                    
                    # Check for minimum required keys
                    self.assertIn('cells', nb, f"Notebook {notebook} missing 'cells' key")
                    self.assertIn('metadata', nb, f"Notebook {notebook} missing 'metadata' key")
                    self.assertIn('nbformat', nb, f"Notebook {notebook} missing 'nbformat' key")
                    
                    # Check that cells are not empty
                    self.assertGreater(len(nb['cells']), 0, f"Notebook {notebook} has no cells")
                except Exception as e:
                    self.fail(f"Notebook {notebook} has invalid JSON structure: {str(e)}")
    
    def test_notebook_cell_content(self):
        """Test that notebooks have cells with content"""
        for notebook in self.expected_notebooks:
            notebook_path = os.path.join(self.notebooks_dir, notebook)
            if os.path.exists(notebook_path):
                try:
                    nb = nbformat.read(notebook_path, as_version=4)
                    
                    # Check for markdown cells
                    markdown_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'markdown']
                    self.assertGreater(len(markdown_cells), 0, f"Notebook {notebook} has no markdown cells")
                    
                    # Check for code cells
                    code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
                    self.assertGreater(len(code_cells), 0, f"Notebook {notebook} has no code cells")
                    
                    # Check that cells have content
                    for i, cell in enumerate(nb['cells']):
                        if cell['cell_type'] == 'markdown':
                            self.assertTrue(cell['source'], f"Markdown cell {i} in {notebook} is empty")
                        elif cell['cell_type'] == 'code':
                            # Code cells can be empty, but we want to ensure most aren't
                            pass
                except Exception as e:
                    self.fail(f"Error checking cell content in {notebook}: {str(e)}")

if __name__ == '__main__':
    unittest.main() 