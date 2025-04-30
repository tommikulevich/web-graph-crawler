from aiohttp import ClientTimeout, ClientSession
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin
from typing import Optional


class Fetcher:
    def __init__(self, user_agent: str, timeout_s: float) -> None:
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        
        self.robots_parser: RobotFileParser = RobotFileParser()
        
        self.session: Optional[ClientSession] = None
        self.base_url: Optional[str] = None

    async def start(self, base_url: str) -> None:
        """Initialize aiohttp session with timeout and fetch robots.txt for the base URL."""
        
        client_timeout = ClientTimeout(total=self.timeout_s)
        self.session = ClientSession(
            headers={'User-Agent': self.user_agent},
            timeout=client_timeout
        )
        
        self.base_url = base_url.rstrip('/')
        robots_url = urljoin(self.base_url, '/robots.txt')
        try:
            async with self.session.get(robots_url) as resp:
                if resp.status == 200:
                    text = await resp.text()
                else:
                    text = ''
        except Exception:
            text = ''
            
        self.robots_parser.parse(text.splitlines())

    async def fetch(self, url: str) -> bytes:
        """Fetch content from URL asynchronously."""
        
        if not self.session:
            await self.start()
            
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.read()

    async def close(self) -> None:
        """Close aiohttp session."""
        
        if self.session:
            await self.session.close()
            
    def is_allowed(self, url: str) -> bool:
        """Check if fetching the URL is allowed by the parsed robots.txt."""
        
        try:
            return self.robots_parser.can_fetch(self.user_agent, url)
        except Exception:
            return True
