from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List


class Parser:
    def __init__(self) -> None:
        pass

    def extract_links(self, html: bytes, base_url: str) -> List[str]:
        """Parse HTML bytes and return list of absolute URLs."""
        
        soup = BeautifulSoup(html, 'html.parser')
        
        links = []
        for tag in soup.find_all('a', href=True):
            href = tag['href']
            url = urljoin(base_url, href)
            links.append(url)
            
        return links
