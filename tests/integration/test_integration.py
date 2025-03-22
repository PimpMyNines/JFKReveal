"""
Integration tests for the JFKReveal application
"""
import os
import json
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

from jfkreveal.main import JFKReveal
from jfkreveal.database.document_processor import DocumentProcessor
from jfkreveal.database.text_cleaner import TextCleaner
from jfkreveal.database.vector_store import VectorStore
from jfkreveal.analysis.document_analyzer import DocumentAnalyzer
from jfkreveal.search.semantic_search import SemanticSearchEngine
from jfkreveal.summarization.findings_report import FindingsReport


class TestIntegration:
    """Integration tests for the JFKReveal application"""

    def test_document_processor_to_vector_store(self, temp_data_dir, sample_text):
        """Test integration between DocumentProcessor and VectorStore"""
        # Create PDF file
        pdf_path = os.path.join(temp_data_dir["raw"], "test_integration.pdf")
        with open(pdf_path, "w") as f:
            f.write("PDF content - will be mocked")
        
        # Initialize components
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            chunk_size=100,
            chunk_overlap=10
        )
        
        vector_store = VectorStore(
            persist_directory=temp_data_dir["vector"],
            embedding_provider="fake"  # Use fake embeddings for testing
        )
        
        # Mock PDF extraction to return our sample text
        with patch.object(processor, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = (sample_text, {"filename": "test_integration.pdf"})
            
            # Process document
            processed_file = processor.process_document(pdf_path)
            
            # Verify file was processed
            assert processed_file is not None
            assert os.path.exists(processed_file)
            
            # Add to vector store
            num_chunks = vector_store.add_documents_from_file(processed_file)
            
            # Verify chunks were added
            assert num_chunks > 0
            
            # Test search functionality
            results = vector_store.similarity_search("Oswald")
            
            # Verify search returned results
            assert len(results) > 0
            assert any("Oswald" in result["text"] for result in results)

    def test_processor_with_immediate_embedding(self, temp_data_dir, sample_text):
        """Test document processor with immediate embedding into vector store"""
        # Create PDF file
        pdf_path = os.path.join(temp_data_dir["raw"], "test_immediate.pdf")
        with open(pdf_path, "w") as f:
            f.write("PDF content - will be mocked")
        
        # Initialize vector store
        vector_store = VectorStore(
            persist_directory=temp_data_dir["vector"],
            embedding_provider="fake"  # Use fake embeddings for testing
        )
        
        # Initialize processor with vector store
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            chunk_size=100,
            chunk_overlap=10,
            vector_store=vector_store
        )
        
        # Mock vector store add_documents_from_file
        with patch.object(vector_store, 'add_documents_from_file') as mock_add:
            mock_add.return_value = 3  # Pretend 3 chunks were added
            
            # Mock PDF extraction
            with patch.object(processor, 'extract_text_from_pdf') as mock_extract:
                mock_extract.return_value = (sample_text, {"filename": "test_immediate.pdf"})
                
                # Process document
                processed_file = processor.process_document(pdf_path)
                
                # Verify file was processed
                assert processed_file is not None
                
                # Verify vector store was called immediately
                mock_add.assert_called_once_with(processed_file)
                
                # Verify document was marked as embedded
                assert processor.check_if_embedded(processed_file)

    def test_pipeline_integration(self, temp_data_dir, sample_text, sample_document_chunks):
        """Test the complete pipeline integration"""
        # Create JFKReveal instance
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        
        # Mock the scraper to return a list of PDF files
        with patch.object(jfk, 'scrape_documents') as mock_scrape:
            mock_scrape.return_value = [
                os.path.join(temp_data_dir["raw"], "doc1.pdf"),
                os.path.join(temp_data_dir["raw"], "doc2.pdf")
            ]
            
            # Create test PDF files
            for pdf_path in mock_scrape.return_value:
                with open(pdf_path, "w") as f:
                    f.write("Test PDF content")
            
            # Mock document processor to return processed files
            with patch.object(DocumentProcessor, 'process_document') as mock_process:
                processed_files = [
                    pdf_path.replace(".pdf", ".json").replace(temp_data_dir["raw"], temp_data_dir["processed"])
                    for pdf_path in mock_scrape.return_value
                ]
                mock_process.side_effect = processed_files
                
                # Create processed files with test content
                for file_path in processed_files:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(sample_document_chunks, f)
                
                # Mock vector store to avoid actual embedding
                with patch.object(VectorStore, 'similarity_search') as mock_search:
                    mock_search.return_value = [
                        {
                            "text": chunk["text"],
                            "metadata": chunk["metadata"],
                            "score": 0.95 - i * 0.1
                        }
                        for i, chunk in enumerate(sample_document_chunks)
                    ]
                    
                    # Mock document analyzer to avoid OpenAI API calls
                    with patch.object(DocumentAnalyzer, 'analyze_key_topics') as mock_analyze:
                        mock_analyze.return_value = ["topic1", "topic2"]
                        
                        # Run the pipeline
                        result = jfk.run_pipeline(skip_scraping=False)
                        
                        # Verify pipeline completed
                        assert result is not None
                        assert os.path.exists(result) or "dummy_report" in result
                        
                        # Verify key methods were called
                        mock_scrape.assert_called_once()
                        assert mock_process.call_count == 2  # Two documents
                        mock_analyze.assert_called_once()

    @patch('langchain_openai.ChatOpenAI')
    def test_document_analysis_chain(self, mock_chat_openai, temp_data_dir, vector_store, sample_document_chunks):
        """Test the document analysis chain from vector store to analyzer"""
        # Mock LLM to avoid API calls
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock structured output method
        mock_chain = MagicMock()
        mock_llm.with_structured_output.return_value = mock_chain
        
        # Mock vector store search to return our sample chunks
        vector_store.similarity_search = MagicMock(return_value=[
            {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": 0.95
            }
            for chunk in sample_document_chunks
        ])
        
        # Create analyzer
        analyzer = DocumentAnalyzer(
            vector_store=vector_store,
            output_dir=temp_data_dir["root"] + "/analysis"
        )
        
        # Use a real DocumentAnalyzer but mock the LLM responses
        with patch.object(analyzer, 'analyze_document_chunk') as mock_analyze_chunk:
            # Return fake analyzed documents
            mock_analyze_chunk.return_value = MagicMock(error=None)
            
            # Search and analyze a topic
            result = analyzer.search_and_analyze_topic("Test Topic", num_results=3)
            
            # Verify vector store was queried
            vector_store.similarity_search.assert_called_once_with("Test Topic", k=3)
            
            # Verify all chunks were analyzed
            assert mock_analyze_chunk.call_count == len(sample_document_chunks)
            
            # Verify result structure
            assert result.topic == "Test Topic"
            assert result.num_documents == len(sample_document_chunks)
            assert result.error is None
            
    def test_text_cleaner_integration_with_document_processor(self, temp_data_dir):
        """Test TextCleaner integration with DocumentProcessor"""
        # Create a text cleaner instance
        text_cleaner = TextCleaner()
        
        # Create a sample document with OCR artifacts
        text_with_artifacts = """
        CONF  I  D  ENTI  AL
        
        Doc. #   123-456-789
        
        Tbe man was seen at  tbe  book depository
        at approx1mately 12.3O PM. W1tness report
        tbat subject had a r1fle and was act1ng 
        susp1ciously. Subject ident1fied as L.H. 0swald.
        
        The date was l1/22/l963 when the events transpired.
        
        C L A S S I F I E D
        """
        
        # Create a PDF file with OCR artifacts
        pdf_path = os.path.join(temp_data_dir["raw"], "ocr_test.pdf")
        with open(pdf_path, "w") as f:
            f.write("PDF content - will be mocked")
        
        # Create a document processor with our text cleaner
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            chunk_size=100,
            chunk_overlap=10,
            text_cleaner=text_cleaner
        )
        
        # Mock PDF extraction to return text with artifacts
        with patch.object(processor, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = (text_with_artifacts, {"filename": "ocr_test.pdf"})
            
            # Process the document
            processed_file = processor.process_document(pdf_path)
            
            # Read the processed file to check cleaned content
            with open(processed_file, 'r') as f:
                chunks = json.load(f)
                
            # Verify OCR artifacts were cleaned
            for chunk in chunks:
                # Check common OCR error corrections
                assert "tbe" not in chunk["text"]  # 'tbe' should be corrected to 'the'
                assert "0swald" not in chunk["text"]  # '0swald' should be corrected to 'Oswald'
                assert "1" not in chunk["text"].split()  # Single digit '1' often mistaken for 'l'
                assert "l2" not in chunk["text"]  # 'l2' should be corrected to '12'
                
                # Verify spacing is normalized
                assert "C O N F I D E N T I A L" not in chunk["text"]
                assert "C L A S S I F I E D" not in chunk["text"]
                
                # Check that some corrections were properly made
                assert "CONFIDENTIAL" in chunks[0]["text"] or "Confidential" in chunks[0]["text"]
                assert "Oswald" in " ".join([c["text"] for c in chunks])
                assert "12:30" in " ".join([c["text"] for c in chunks]) or "12.30" in " ".join([c["text"] for c in chunks])
                
    def test_semantic_search_integration(self, temp_data_dir, sample_document_chunks):
        """Test integration of SemanticSearch with VectorStore"""
        # Set up vector store with sample data
        vector_store = VectorStore(
            persist_directory=temp_data_dir["vector"],
            embedding_provider="fake"  # Use fake embeddings for testing
        )
        
        # Create a sample processed file
        processed_file = os.path.join(temp_data_dir["processed"], "search_test.json")
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
        with open(processed_file, "w", encoding="utf-8") as f:
            json.dump(sample_document_chunks, f)
        
        # Add sample document to vector store
        vector_store.add_documents_from_file(processed_file)
        
        # Create SemanticSearchEngine instance using our vector store
        search = SemanticSearchEngine(
            vector_db=vector_store,
            results_dir=os.path.join(temp_data_dir["root"], "search_results")
        )
        
        # Define test queries
        queries = [
            "Oswald Texas School Book Depository",
            "witness reports grassy knoll",
            "ballistic evidence multiple shooters",
            "conspiracy theories JFK assassination"
        ]
        
        # Test semantic search functionality
        with patch.object(vector_store, 'similarity_search') as mock_search:
            # Set up mock to return different subsets of our sample chunks for different queries
            mock_search.side_effect = lambda query, k=5: [
                {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": 0.95 - i * 0.05
                }
                # Return different chunks based on query keywords
                for i, chunk in enumerate(
                    [c for c in sample_document_chunks 
                     if any(kw.lower() in c["text"].lower() 
                            for kw in query.split())]
                )
            ]
            
            # Run searches for each query
            results = []
            for query in queries:
                # Mock the hybrid search method
                with patch.object(search, 'hybrid_search') as mock_hybrid_search:
                    mock_result = {
                        "query": query,
                        "results": [
                            {
                                "text": chunk["text"],
                                "metadata": chunk["metadata"],
                                "score": 0.95 - i * 0.05
                            }
                            for i, chunk in enumerate(sample_document_chunks[:3])
                        ]
                    }
                    mock_hybrid_search.return_value = mock_result["results"]
                    
                    # Run search
                    query_results = search.hybrid_search(query, k=3)
                    results.append({"query": query, "results": query_results})
            
            # Check that search was executed for each query
            assert mock_search.call_count == len(queries)
            
            # Verify results structure
            assert len(results) == len(queries)
            for query_result in results:
                assert "query" in query_result
                assert "results" in query_result
                assert isinstance(query_result["results"], list)
                
            # Verify results were saved
            assert os.path.exists(os.path.join(temp_data_dir["root"], "search_results"))
                
    def test_findings_report_integration(self, temp_data_dir, sample_document_chunks):
        """Test FindingsReport integration with document analysis"""
        # Set up directory structure
        analysis_dir = os.path.join(temp_data_dir["root"], "analysis")
        reports_dir = os.path.join(temp_data_dir["root"], "reports")
        os.makedirs(analysis_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        
        # Create sample analysis files
        analysis_files = [
            "oswald_analysis.json",
            "ballistics_analysis.json",
            "witnesses_analysis.json"
        ]
        
        # Create analysis with topics and summaries
        for i, filename in enumerate(analysis_files):
            analysis = {
                "topic": f"Topic {i+1}",
                "documents": sample_document_chunks,
                "summary": {
                    "key_findings": [f"Finding {j+1} for topic {i+1}" for j in range(3)],
                    "potential_evidence": [f"Evidence {j+1} for topic {i+1}" for j in range(2)],
                    "credibility": "High" if i % 2 == 0 else "Medium"
                },
                "entities": [
                    {"name": "Lee Harvey Oswald", "type": "PERSON"},
                    {"name": "Texas School Book Depository", "type": "LOCATION"},
                    {"name": "Grassy Knoll", "type": "LOCATION"}
                ],
                "additional_evidence": [
                    {"document_id": f"doc_{j+1}", "text": f"Additional evidence {j+1}"} 
                    for j in range(2)
                ]
            }
            
            file_path = os.path.join(analysis_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f)
        
        # Create FindingsReport instance
        report = FindingsReport(
            analysis_dir=analysis_dir,
            output_dir=reports_dir
        )
        
        # Mock LangChain calls to avoid API usage
        with patch('langchain_openai.ChatOpenAI', autospec=True) as mock_llm_class:
            # Mock the LLM
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            
            # Mock the structured output
            mock_structured_output = MagicMock()
            mock_llm.with_structured_output.return_value = mock_structured_output
            
            # Create a mock response for unstructured calls
            mock_response = MagicMock()
            mock_response.content = "Mock report content for testing"
            mock_llm.invoke.return_value = mock_response
            
            # Generate an executive summary
            summary = report.generate_executive_summary(report.load_analyses())
            
            # Check that the summary was generated
            assert summary is not None
            assert len(summary) > 0
            
            # Verify the output files were created
            assert os.path.exists(os.path.join(reports_dir, "executive_summary.md"))
            
            # Generate detailed findings
            findings = report.generate_detailed_findings(report.load_analyses())
            
            # Check that findings were generated
            assert findings is not None
            assert len(findings) > 0
            
            # Verify the output files were created
            assert os.path.exists(os.path.join(reports_dir, "detailed_findings.md"))
            
    def test_document_analyzer_to_findings_report_integration(self, temp_data_dir, sample_document_chunks):
        """Test integration between DocumentAnalyzer and FindingsReport"""
        # Set up directory structure
        analysis_dir = os.path.join(temp_data_dir["root"], "analysis")
        reports_dir = os.path.join(temp_data_dir["root"], "reports")
        os.makedirs(analysis_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        
        # Set up vector store with sample data
        vector_store = VectorStore(
            persist_directory=temp_data_dir["vector"],
            embedding_provider="fake"  # Use fake embeddings for testing
        )
        
        # Create a sample processed file
        processed_file = os.path.join(temp_data_dir["processed"], "analyzer_test.json")
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
        with open(processed_file, "w", encoding="utf-8") as f:
            json.dump(sample_document_chunks, f)
        
        # Add sample document to vector store
        vector_store.add_documents_from_file(processed_file)
        
        # Create DocumentAnalyzer instance
        analyzer = DocumentAnalyzer(
            vector_store=vector_store,
            output_dir=analysis_dir,
            model_name="fake-model"  # Use fake model name for testing
        )
        
        # Mock the LLM to avoid OpenAI API calls
        with patch('langchain_openai.ChatOpenAI', autospec=True) as mock_llm_class:
            # Mock LLM instance
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            
            # Mock structured output
            mock_structured_output = MagicMock()
            mock_llm.with_structured_output.return_value = mock_structured_output
            
            # Mock LLM responses
            mock_response = MagicMock()
            mock_response.error = None
            mock_response.document_id = "test_doc"
            mock_response.summary = "Test summary"
            mock_response.key_findings = ["Finding 1", "Finding 2"]
            mock_response.key_individuals = ["Oswald", "Ruby"]
            mock_structured_output.invoke.return_value = mock_response
            
            # Mock vector store search to return our sample chunks
            vector_store.similarity_search = MagicMock(return_value=[
                {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": 0.95
                }
                for chunk in sample_document_chunks
            ])
            
            # Analyze a test topic
            result = analyzer.search_and_analyze_topic("Oswald's movements")
            
            # Check that analysis was completed
            assert result is not None
            assert result.topic == "Oswald's movements"
            
            # Verify analysis file was saved
            assert len(os.listdir(analysis_dir)) > 0
            
            # Create FindingsReport instance
            report = FindingsReport(
                analysis_dir=analysis_dir,
                output_dir=reports_dir
            )
            
            # Generate a report from the analyzed topic
            report.llm = mock_llm  # Use our already mocked LLM
            
            # Mock _save_report_file to avoid file system issues in testing
            with patch.object(report, '_save_report_file') as mock_save:
                mock_save.return_value = "test_report_path.md"
                
                # Generate the executive summary
                exec_summary = report.generate_executive_summary(report.load_analyses())
                
                # Verify executive summary was generated
                assert exec_summary is not None
                mock_save.assert_called_with(exec_summary, "executive_summary.md")
                
    def test_end_to_end_pipeline_with_mocked_components(self, temp_data_dir, sample_document_chunks):
        """Test end-to-end pipeline with mocked components"""
        # Create basic file structure
        os.makedirs(os.path.join(temp_data_dir["root"], "raw"), exist_ok=True)
        os.makedirs(os.path.join(temp_data_dir["root"], "processed"), exist_ok=True)
        os.makedirs(os.path.join(temp_data_dir["root"], "vectordb"), exist_ok=True)
        os.makedirs(os.path.join(temp_data_dir["root"], "analysis"), exist_ok=True)
        os.makedirs(os.path.join(temp_data_dir["root"], "reports"), exist_ok=True)
        
        # Create sample PDF files
        for i in range(3):
            pdf_path = os.path.join(temp_data_dir["root"], "raw", f"document_{i}.pdf")
            with open(pdf_path, "w") as f:
                f.write(f"Sample content for document {i}")
        
        # Create JFKReveal instance with our temp directory
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        
        # Mock scraper to return our sample files
        with patch.object(jfk, 'scrape_documents') as mock_scrape:
            mock_scrape.return_value = [
                os.path.join(temp_data_dir["root"], "raw", f"document_{i}.pdf")
                for i in range(3)
            ]
            
            # Mock document processor to avoid actual PDF parsing
            with patch.object(DocumentProcessor, 'process_document') as mock_process:
                # Define processed files that would be created
                processed_files = [
                    os.path.join(temp_data_dir["root"], "processed", f"document_{i}.json")
                    for i in range(3)
                ]
                mock_process.side_effect = processed_files
                
                # Create the processed files with our sample chunks
                for file_path in processed_files:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(sample_document_chunks, f)
                
                # Mock the vectorization to avoid actual embeddings
                with patch.object(VectorStore, 'add_documents_from_file') as mock_add:
                    mock_add.return_value = len(sample_document_chunks)
                    
                    # Mock the analysis to avoid API calls
                    with patch.object(DocumentAnalyzer, 'analyze_key_topics') as mock_analyze:
                        # Return a list of topics that would be analyzed
                        mock_analyze.return_value = ["Topic 1", "Topic 2", "Topic 3"]
                        
                        # Mock the search_and_analyze_topic method
                        with patch.object(DocumentAnalyzer, 'search_and_analyze_topic') as mock_search_analyze:
                            # Create a mock result
                            analysis_result = MagicMock()
                            analysis_result.topic = "Mock Topic"
                            analysis_result.num_documents = 3
                            analysis_result.documents = sample_document_chunks
                            analysis_result.error = None
                            mock_search_analyze.return_value = analysis_result
                            
                            # Mock the report generation
                            with patch.object(FindingsReport, 'generate_full_report') as mock_report:
                                mock_report.return_value = {
                                    "executive_summary": "Mock executive summary",
                                    "detailed_findings": "Mock detailed findings"
                                }
                                
                                # Run the pipeline
                                result = jfk.run_pipeline(
                                    skip_scraping=False,
                                    skip_processing=False, 
                                    skip_analysis=False
                                )
                                
                                # Verify the pipeline completed successfully
                                assert result is not None
                                
                                # Verify each component was called appropriately
                                mock_scrape.assert_called_once()
                                assert mock_process.call_count == 3  # One for each document
                                assert mock_add.call_count == 3  # One for each processed document
                                mock_analyze.assert_called_once()
                                assert mock_search_analyze.call_count == 3  # One for each topic
                                mock_report.assert_called_once() 