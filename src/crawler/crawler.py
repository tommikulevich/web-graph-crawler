import os
import csv
import time
import queue
import threading
from typing import Set, Tuple

from common.logger import get_logger
from .fetcher import Fetcher
from .parser import Parser
from .storage import Storage


class Crawler:
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
        self.edges: Set[Tuple[str, str]] = set()

        os.makedirs(self.storage_path, exist_ok=True)
        self._visited_file = os.path.join(self.storage_path, 'visited.txt')
        self._edges_file = os.path.join(self.storage_path, 'edges.partial.csv')

        self.visited_lock = threading.RLock()
        self.edges_lock = threading.RLock()
        self._stop_flush = threading.Event()

    def start(self) -> None:
        """Entry point: initialize state, start threads, and finalize crawl."""
        
        self._init_state()

        task_queue = queue.Queue()

        initial = set()
        if self.base_url not in self.visited:
            initial.add(self.base_url)
        for _, tgt in self.edges:
            if tgt not in self.visited:
                initial.add(tgt)
        with self.visited_lock:
            self.seen.update(initial)
        for url in initial:
            task_queue.put(url)

        flush_thread = threading.Thread(
            target=self._flush_loop,
            name='FlushThread',
            daemon=True,
        )
        flush_thread.start()

        workers = []
        for wid in range(self.threads_num):
            w = Worker(self, wid)
            t = threading.Thread(
                target=w.run,
                args=(task_queue,),
                name=f'Worker-{wid}',
            )
            t.start()
            workers.append(t)

        for t in workers:
            t.join()

        self._stop_flush.set()
        flush_thread.join()
        with self.visited_lock:
            try:
                self._visited_fh.flush()
            except Exception:
                pass
            try:
                self._visited_fh.close()
            except Exception:
                pass
        with self.edges_lock:
            try:
                self._edges_fh.flush()
            except Exception:
                pass
            try:
                self._edges_fh.close()
            except Exception:
                pass

        edges_csv = os.path.join(self.storage_path, 'edges.csv')
        try:
            with open(edges_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['source', 'target'])
                valid = [(s, t) for (s, t) in self.edges if s in self.visited and t in self.visited]
                for src, tgt in sorted(valid):
                    writer.writerow([src, tgt])
            self.logger.info(f'Final edges saved to {edges_csv}')
        except Exception as e:
            self.logger.error(f'Error writing final edges.csv: {e}')

    def _init_state(self) -> None:
        """Load or initialize persistent visited URLs and partial edges."""

        self.visited.clear()
        if os.path.exists(self._visited_file):
            with open(self._visited_file, 'r', encoding='utf-8') as f:
                for line in f:
                    u = line.strip()
                    if u:
                        self.visited.add(u)
        self.seen = set(self.visited)

        self._visited_fh = open(self._visited_file, 'a', encoding='utf-8')

        self.edges.clear()
        is_new = not os.path.exists(self._edges_file)
        if not is_new:
            with open(self._edges_file, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        self.edges.add((row[0], row[1]))

        self._edges_fh = open(self._edges_file, 'a', newline='', encoding='utf-8')
        self._edges_writer = csv.writer(self._edges_fh)
        if is_new:
            self._edges_writer.writerow(['source', 'target'])

    def _record_visited(self, url: str) -> None:
        """Record visited URL to buffer and memory."""
        
        with self.visited_lock:
            if url not in self.visited:
                self.visited.add(url)
                self._visited_fh.write(url + '\n')

    def _record_edge(self, src: str, tgt: str) -> None:
        """Record directed edge to buffer and memory."""
        
        with self.edges_lock:
            pair = (src, tgt)
            if pair not in self.edges:
                self._edges_writer.writerow([src, tgt])
                self.edges.add(pair)

    def _flush_loop(self) -> None:
        """Periodically flush open file buffers to disk."""
        
        while not self._stop_flush.is_set():
            with self.visited_lock:
                try:
                    self._visited_fh.flush()
                except Exception:
                    pass

            with self.edges_lock:
                try:
                    self._edges_fh.flush()
                except Exception:
                    pass
            time.sleep(5)


class Worker:
    def __init__(self, crawler: Crawler, worker_id: int) -> None:
        self.crawler = crawler
        self.worker_id = worker_id
        
        self.fetcher = Fetcher(crawler.user_agent, crawler.timeout_s)
        self.parser = Parser()

    def run(self, task_queue: queue.Queue) -> None:
        logger = self.crawler.logger

        try:
            self.fetcher.start(self.crawler.base_url)
        except Exception as e:
            logger.error(f'[{self.worker_id}] Fetcher init error: {e}')
            return

        while True:
            with self.crawler.visited_lock:
                if len(self.crawler.visited) >= self.crawler.max_pages:
                    break
            try:
                url = task_queue.get(timeout=5)
            except queue.Empty:
                break

            with self.crawler.visited_lock:
                if url in self.crawler.visited:
                    continue

            if not self.fetcher.is_allowed(url):
                logger.info(f'[{self.worker_id}] Blocked by robots.txt: {url}')
                continue

            try:
                content = self.fetcher.fetch(url)
            except Exception as e:
                logger.error(f'[{self.worker_id}] Fetch error {url}: {e}')
                continue

            self.crawler._record_visited(url)

            try:
                path = self.crawler.storage.save(url, content)
                logger.info(f'[{self.worker_id}] Fetched {url} -> {path}')
            except Exception as e:
                logger.error(f'[{self.worker_id}] Save error {url}: {e}')
                continue

            try:
                links = self.parser.extract_links(content, url)
            except Exception as e:
                logger.error(f'[{self.worker_id}] Parse error {url}: {e}')
                continue

            for link in links:
                self.crawler._record_edge(url, link)

                with self.crawler.visited_lock:
                    if link not in self.crawler.seen:
                        self.crawler.seen.add(link)
                        task_queue.put(link)

        try:
            self.fetcher.close()
        except Exception:
            pass