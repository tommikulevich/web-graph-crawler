import requests
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin
from typing import Optional


class Fetcher:
    def __init__(self, user_agent: str, timeout_s: float) -> None:
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        
        self.robots_parser: RobotFileParser = RobotFileParser()
        
        self.session: Optional[requests.Session] = None
        self.base_url: Optional[str] = None

    def start(self, base_url: str) -> None:
        """Initialize requests session and fetch robots.txt for the base URL."""
        
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        self.base_url = base_url.rstrip('/')
        robots_url = urljoin(self.base_url, '/robots.txt')
        
        try:
            resp = self.session.get(robots_url, timeout=self.timeout_s)
            if resp.status_code == 200:
                text = resp.text
            else:
                text = ''
        except Exception:
            text = ''
            
        self.robots_parser.parse(text.splitlines())

    def fetch(self, url: str) -> bytes:
        """Fetch content from URL synchronously."""
        
        if not self.session:
            raise RuntimeError('Session not initialized; call start() first')
        
        resp = self.session.get(url, timeout=self.timeout_s)
        resp.raise_for_status()
        
        return resp.content

    def close(self) -> None:
        """Close requests session."""
        
        if self.session:
            self.session.close()
            
    def is_allowed(self, url: str) -> bool:
        """Check if fetching the URL is allowed by the parsed robots.txt."""
        
        try:
            return self.robots_parser.can_fetch(self.user_agent, url)
        except Exception:
            return True
