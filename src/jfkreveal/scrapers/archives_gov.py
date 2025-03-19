"""
Scraper for downloading PDF documents from the National Archives JFK Release 2025 collection.
"""
import os
import re
import time
import logging
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ArchivesGovScraper:
    """Scraper for the National Archives JFK Release 2025 collection."""
    
    BASE_URL = "https://www.archives.gov/research/jfk/release-2025"
    
    def __init__(self, output_dir="data/raw", delay=1.0):
        """
        Initialize the scraper.
        
        Args:
            output_dir (str): Directory to save downloaded PDFs
            delay (float): Delay between requests in seconds
        """
        self.output_dir = output_dir
        self.delay = delay
        os.makedirs(output_dir, exist_ok=True)
        
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
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_links = []
        
        # Find all links in the page
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href.endswith('.pdf'):
                full_url = urljoin(url, href)
                pdf_links.append(full_url)
                
        # Also find links to additional pages that might contain PDFs
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and 'jfk/release-2025' in href and href != url:
                sub_page_url = urljoin(url, href)
                # Recursively extract links from sub-pages
                sub_links = self.extract_links(sub_page_url)
                pdf_links.extend(sub_links)
                
        return list(set(pdf_links))  # Remove duplicates
    
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
    
    def download_pdf(self, url):
        """
        Download a PDF from the given URL.
        
        Args:
            url (str): URL of the PDF to download
            
        Returns:
            str: Path to the downloaded file
        """
        filename = self._sanitize_filename(url)
        output_path = os.path.join(self.output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            return output_path
        
        try:
            logger.info(f"Downloading {url} to {output_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            # Remove partially downloaded file
            if os.path.exists(output_path):
                os.remove(output_path)
            return None
    
    def scrape_all(self):
        """
        Scrape all PDF documents from the National Archives JFK Release site.
        
        Returns:
            list: Paths to all downloaded files
        """
        pdf_links = self.extract_links()
        logger.info(f"Found {len(pdf_links)} PDF documents")
        
        downloaded_files = []
        
        for url in tqdm(pdf_links, desc="Downloading PDFs"):
            file_path = self.download_pdf(url)
            if file_path:
                downloaded_files.append(file_path)
            
            # Respect the site by adding a delay between requests
            time.sleep(self.delay)
            
        logger.info(f"Downloaded {len(downloaded_files)} files to {self.output_dir}")
        return downloaded_files