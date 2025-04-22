import trafilatura
import requests
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_website_text_content(url: str) -> Optional[str]:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.

    Args:
        url: The URL of the website to scrape
    
    Returns:
        str: The extracted text content or None if extraction failed
    """
    try:
        # Send a request to the website
        logger.info(f"Fetching content from: {url}")
        downloaded = trafilatura.fetch_url(url)
        
        if downloaded:
            text = trafilatura.extract(downloaded)
            
            if text and len(text.strip()) > 0:
                logger.info(f"Successfully extracted {len(text)} characters from {url}")
                return text
            else:
                logger.warning(f"No text content extracted from {url}")
                return None
        else:
            logger.warning(f"Failed to download content from {url}")
            return None
    
    except Exception as e:
        logger.error(f"Error scraping website {url}: {str(e)}")
        return None