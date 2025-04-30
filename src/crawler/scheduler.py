import os
import csv
import asyncio
from typing import Set, Optional

from common.logger import get_logger
from .fetcher import Fetcher
from .parser import Parser
from .storage import Storage


class Scheduler:
    def __init__(self, base_url: str, max_pages: int, threads_num: int,
                 user_agent: str, storage_path: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip('/')
        self.max_pages = max_pages
        self.threads_num = threads_num
        self.user_agent = user_agent
        self.storage_path = storage_path
        self.timeout_s = timeout_s

        self.logger = get_logger(self.__class__.__name__)
        self.storage = Storage(self.storage_path)
        
        self.visited: Set[str] = set()
        self.seen: Set[str] = set()
        self.edges: Set[tuple[str, str]] = set()
        
        self._visited_file = os.path.join(self.storage_path, 'visited.txt')
        self._edges_file = os.path.join(self.storage_path, 'edges.partial.csv')

    def start(self) -> None:
        """Entry point: run the asynchronous crawl loop."""
        
        asyncio.run(self._crawl())
    
    def _init_state(self) -> None:
        """Initialize or load persistent visited URLs and partial edges."""
        
        self.edges = set()
        self.visited = set()
        
        os.makedirs(self.storage_path, exist_ok=True)
        if os.path.exists(self._visited_file):
            with open(self._visited_file, 'r', encoding='utf-8') as f:
                for line in f:
                    url = line.strip()
                    if url:
                        self.visited.add(url)
        self.seen = set(self.visited)

        if os.path.exists(self._edges_file):
            with open(self._edges_file, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        self.edges.add((row[0], row[1]))
        else:
            with open(self._edges_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['source', 'target'])

    def _record_visited(self, url: str) -> None:
        """Append a visited URL to persistent store and update in-memory set."""
        
        if url not in self.visited:
            self.visited.add(url)
            with open(self._visited_file, 'a', encoding='utf-8') as f:
                f.write(url + '\n')

    def _record_edge(self, source: str, target: str) -> None:
        """Append a discovered edge to persistent store if new."""
        
        pair = (source, target)
        if pair not in self.edges:
            with open(self._edges_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([source, target])
            self.edges.add(pair)

    # [MS] 2.2
    
    async def _crawl(self) -> None:
        self._init_state()
        
        queue: asyncio.Queue = asyncio.Queue()
        
        initial = set()
        if self.base_url not in self.visited:
            initial.add(self.base_url)
            
        frontier = {t for (_, t) in self.edges if t not in self.visited}
        initial |= frontier

        self.seen.update(initial)
        for url in initial:
            queue.put_nowait(url)
        
        fetcher = Fetcher(self.user_agent, self.timeout_s)
        parser = Parser()
        storage = self.storage

        await fetcher.start(self.base_url)
        self.logger.info(f"Initialized fetcher and loaded robots.txt for {self.base_url}")

        workers = [Worker(fetcher, parser, storage, self, idx) for idx in range(self.threads_num)]
        tasks = [asyncio.create_task(w.run(queue)) for w in workers]
        await asyncio.gather(*tasks)
        
        edges_file = os.path.join(self.storage_path, 'edges.csv')
        try:
            with open(edges_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['source', 'target'])
                
                valid_edges = [(s, t) for (s, t) in self.edges if s in self.visited and t in self.visited]
                for source, target in sorted(valid_edges):
                    writer.writerow([source, target])
            self.logger.info(f"Saved filtered edges to {edges_file}")
        except Exception as e:
            self.logger.error(f"Failed to save edges: {e}")
            
        await fetcher.close()
        self.logger.info(f"Crawling finished, visited {len(self.visited)} pages.")

class Worker:
    def __init__(self, fetcher: Fetcher, parser: Parser, storage: Storage, 
                 scheduler: Scheduler, worker_id: int) -> None:
        self.fetcher = fetcher
        self.parser = parser
        self.storage = storage
        self.scheduler = scheduler
        self.worker_id = worker_id

    async def run(self, queue: asyncio.Queue) -> None:
        """Worker loop: fetch and process URLs until the crawl completes or queue is empty."""
        
        logger = self.scheduler.logger
        while len(self.scheduler.visited) < self.scheduler.max_pages:
            url = await self._get_next_url(queue)
            if url is None:
                return
            
            if not self._should_visit(url):
                continue
            
            # [MS] 2.1
            
            if not self.fetcher.is_allowed(url):
                logger.info(f"[{self.worker_id}] Blocked by robots.txt: {url}")
                continue
            
            await self._process_url(url, queue)

    async def _get_next_url(self, queue: asyncio.Queue) -> Optional[str]:
        """Retrieve the next URL from the queue with a timeout."""
        
        try:
            return await asyncio.wait_for(queue.get(), timeout=10.0)
        except asyncio.TimeoutError:
            return None

    def _should_visit(self, url: str) -> bool:
        """Determine if the URL should be visited (not visited and within domain)."""
        
        return url.startswith(self.scheduler.base_url) and url not in self.scheduler.visited

    async def _process_url(self, url: str, queue: asyncio.Queue) -> None:
        """Fetch, store, parse URL content and enqueue discovered links."""
        
        logger = self.scheduler.logger
        
        try:
            content = await self.fetcher.fetch(url)
        except Exception as exc:
            logger.error(f"[{self.worker_id}] Failed to fetch {url}: {exc}")
            return

        self.scheduler._record_visited(url)
        try:
            path = self.storage.save(url, content)
            logger.info(f"[{self.worker_id}] Fetched {url} -> {path}")
        except Exception as exc:
            logger.error(f"[{self.worker_id}] Failed to save {url}: {exc}")
            return

        try:
            links = self.parser.extract_links(content, url)
        except Exception as exc:
            logger.error(f"[{self.worker_id}] Failed to parse {url}: {exc}")
            return

        for link in links:
            if link.startswith(self.scheduler.base_url) and link not in self.scheduler.seen:
                self.scheduler._record_edge(url, link)
                self.scheduler.seen.add(link)
                queue.put_nowait(link)