"""
Scraper for downloading PDF documents from the National Archives JFK Release 2025 collection.
"""
import os
import re
import time
import logging
import random
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm
import backoff
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class PDFDocument(BaseModel):
    """Pydantic model for a PDF document."""
    url: str
    filename: str
    local_path: Optional[str] = None
    downloaded: bool = False
    error: Optional[str] = None

class ScraperConfig(BaseModel):
    """Configuration for the scraper."""
    delay: float = Field(default=1.0, description="Base delay between requests in seconds")
    max_retries: int = Field(default=5, description="Maximum number of retries for failed requests")
    backoff_factor: float = Field(default=0.5, description="Backoff factor for retries")
    jitter: float = Field(default=0.25, description="Jitter factor to randomize delays")
    timeout: int = Field(default=30, description="Request timeout in seconds")

class ArchivesGovScraper:
    """Scraper for the National Archives JFK Release 2025 collection."""
    
    BASE_URL = "https://www.archives.gov/research/jfk/release-2025"
    
    def __init__(self, output_dir="data/raw", config: Optional[ScraperConfig] = None):
        """
        Initialize the scraper.
        
        Args:
            output_dir (str): Directory to save downloaded PDFs
            config (ScraperConfig, optional): Configuration for the scraper
        """
        self.output_dir = output_dir
        self.config = config or ScraperConfig()
        self.use_cache = True  # Default to using cache
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure session with retry, backoff, and jitter
        self.session = self._create_session()
        
    def _create_session(self):
        """
        Create a requests session with retry capability.
        
        Returns:
            requests.Session: Configured session with retry functionality
        """
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _sleep_with_jitter(self):
        """Sleep with jitter to avoid rate limiting"""
        jitter_amount = random.uniform(-self.config.jitter, self.config.jitter) * self.config.delay
        sleep_time = max(0.1, self.config.delay + jitter_amount)
        time.sleep(sleep_time)
    
    @backoff.on_exception(
        backoff.expo,
        (RequestException, HTTPError, ConnectionError, Timeout),
        max_tries=5,
        jitter=backoff.full_jitter,
        factor=2
    )
    def _fetch_page(self, url):
        """
        Fetch a page with retry capabilities.
        
        Args:
            url: URL to fetch
            
        Returns:
            Response object
        """
        logger.info(f"Fetching page: {url}")
        response = self.session.get(url, timeout=self.config.timeout)
        response.raise_for_status()
        return response
    
    def extract_links(self, url=None):
        """
        Extract all PDF links from the National Archives page.
        
        Args:
            url (str, optional): URL to scrape. Defaults to BASE_URL.
            
        Returns:
            list: List of PDF URLs
        """
        if url is None:
            url = self.BASE_URL
            
        logger.info(f"Extracting links from {url}")
        try:
            response = self._fetch_page(url)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            pdf_links = []
            
            # Find all links in the page
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and href.endswith('.pdf'):
                    full_url = urljoin(url, href)
                    pdf_links.append(full_url)
            
            # Also find links to additional pages that might contain PDFs
            sub_pages = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and 'jfk/release-2025' in href and href != url:
                    sub_page_url = urljoin(url, href)
                    sub_pages.append(sub_page_url)
                    
            # Process sub-pages with delay to avoid rate limiting
            for sub_page_url in sub_pages:
                self._sleep_with_jitter()
                # Recursively extract links from sub-pages
                sub_links = self.extract_links(sub_page_url)
                pdf_links.extend(sub_links)
                
            return list(set(pdf_links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {e}")
            return []
    
    def _sanitize_filename(self, url):
        """
        Create a safe filename from URL.
        
        Args:
            url (str): URL of the PDF
            
        Returns:
            str: Sanitized filename
        """
        # Extract the filename from the URL
        filename = url.split('/')[-1]
        
        # Remove any invalid characters
        filename = re.sub(r'[^\w\-\.]', '_', filename)
        
        return filename
    
    @backoff.on_exception(
        backoff.expo,
        (RequestException, HTTPError, ConnectionError, Timeout, IOError),
        max_tries=10,
        jitter=backoff.full_jitter,
        factor=2
    )
    def _download_file(self, url, output_path):
        """
        Download a file with retry capabilities.
        
        Args:
            url: URL to download
            output_path: Path to save the file
            
        Returns:
            bool: True if download was successful
        """
        logger.info(f"Downloading {url} to {output_path}")
        
        # Stream the download for large files
        response = self.session.get(url, stream=True, timeout=self.config.timeout)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=os.path.basename(output_path)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify file is not empty
        if os.path.getsize(output_path) > 0:
            logger.info(f"Successfully downloaded {url}")
            return True
        else:
            logger.warning(f"Downloaded file is empty: {output_path}")
            os.remove(output_path)
            return False
            
    def set_retry_config(self, max_retries: int, retry_delay: int, backoff_factor: float):
        """
        Update retry configuration.
        
        Args:
            max_retries: Maximum number of retries for failed operations
            retry_delay: Base delay between retries in seconds
            backoff_factor: Exponential backoff factor for retries
        """
        # Update config
        self.config.max_retries = max_retries
        self.config.delay = retry_delay
        self.config.backoff_factor = backoff_factor
        
        # Recreate session with new settings
        self.session = self._create_session()
        logger.info(f"Updated retry configuration: max_retries={max_retries}, delay={retry_delay}s, backoff_factor={backoff_factor}")
    
    def download_pdf(self, url):
        """
        Download a PDF from the given URL with retry capabilities.
        
        Args:
            url (str): URL of the PDF to download
            
        Returns:
            str: Path to the downloaded file
        """
        document = PDFDocument(url=url, filename=self._sanitize_filename(url))
        output_path = os.path.join(self.output_dir, document.filename)
        document.local_path = output_path
        
        # Skip if file already exists and using cache
        if os.path.exists(output_path) and self.use_cache:
            logger.info(f"File already exists (cache enabled): {output_path}")
            document.downloaded = True
            return output_path
        elif os.path.exists(output_path) and not self.use_cache:
            logger.info(f"File exists but cache disabled, re-downloading: {output_path}")
            os.remove(output_path)  # Remove existing file to force re-download
        
        try:
            success = self._download_file(url, output_path)
            if success:
                document.downloaded = True
                return output_path
            else:
                document.error = "Empty file"
                return None
        except Exception as e:
            error_msg = f"Error downloading {url}: {str(e)}"
            logger.error(error_msg)
            document.error = error_msg
            # Remove partially downloaded file
            if os.path.exists(output_path) and os.path.isfile(output_path):
                os.remove(output_path)
            return None
    
    def scrape_all(self):
        """
        Scrape all PDF documents from the National Archives JFK Release site.
        
        Returns:
            list: List of downloaded file paths
        """
        pdf_links = self.extract_links()
        logger.info(f"Found {len(pdf_links)} PDF documents")
        
        # Filter out URLs where the file already exists if using cache
        filtered_links = []
        existing_files = []
        
        for url in pdf_links:
            filename = self._sanitize_filename(url)
            output_path = os.path.join(self.output_dir, filename)
            if os.path.exists(output_path) and self.use_cache:
                logger.info(f"Skipping {url} - file already exists at {output_path} (cache enabled)")
                existing_files.append(output_path)
            else:
                if os.path.exists(output_path) and not self.use_cache:
                    logger.info(f"File exists but cache disabled, queuing for re-download: {output_path}")
                filtered_links.append(url)
        
        if self.use_cache:
            logger.info(f"After filtering existing files: {len(filtered_links)} PDFs need to be downloaded, {len(existing_files)} already exist")
        else:
            logger.info(f"Cache disabled: downloading all {len(filtered_links)} PDFs, ignoring {len(existing_files)} existing files")
        
        downloaded_files = []
        documents = []
        
        # Add existing files to documents list if using cache
        if self.use_cache:
            for output_path in existing_files:
                filename = os.path.basename(output_path)
                # Use BASE_URL for more accurate URL reconstruction
                url = f"{self.BASE_URL}/{filename}"
                document = PDFDocument(
                    url=url,
                    filename=filename,
                    local_path=output_path,
                    downloaded=True,
                    error=None
                )
                documents.append(document)
                downloaded_files.append(output_path)
        
        for url in tqdm(filtered_links, desc="Downloading PDFs"):
            file_path = self.download_pdf(url)
            
            # Create document object
            document = PDFDocument(
                url=url, 
                filename=self._sanitize_filename(url),
                local_path=file_path,
                downloaded=file_path is not None,
                error=None if file_path else "Download failed"
            )
            documents.append(document)
            
            if file_path:
                downloaded_files.append(file_path)
            
            # Respect the site by adding a delay with jitter between requests
            self._sleep_with_jitter()
            
        if self.use_cache:
            logger.info(f"Downloaded {len(downloaded_files) - len(existing_files)} new files, {len(existing_files)} already existed, total: {len(downloaded_files)} files in {self.output_dir}")
        else:
            logger.info(f"Downloaded {len(downloaded_files)} files to {self.output_dir} (cache disabled)")
            
        return downloaded_files, documents