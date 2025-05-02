from lxml import html
from urllib.parse import urljoin
from typing import List


class Parser:
    def __init__(self) -> None:
        pass

    def extract_links(self, html_content: bytes, base_url: str) -> List[str]:
        """Parse HTML bytes with lxml and return list of absolute URLs."""
        
        try:
            doc = html.fromstring(html_content)
        except Exception:
            return []
        
        links: List[str] = []
        for element in doc.xpath('//a[@href]'):
            href = element.get('href')
            if not href:
                continue
            url = urljoin(base_url, href)
            links.append(url)
            
        return links
