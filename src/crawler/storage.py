import os
import json
from hashlib import md5


class Storage:
    def __init__(self, storage_path: str) -> None:
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

        self.pages_dir = os.path.join(self.storage_path, 'pages')
        os.makedirs(self.pages_dir, exist_ok=True)

        self.mapping_path = os.path.join(self.storage_path, 'mapping.json')
        if os.path.exists(self.mapping_path):
            try:
                with open(self.mapping_path, 'r', encoding='utf-8') as f:
                    self.mapping = json.load(f)
            except Exception:
                self.mapping = {}
        else:
            self.mapping = {}
            with open(self.mapping_path, 'w', encoding='utf-8') as f:
                json.dump(self.mapping, f, indent=2)

    def save(self, url: str, content: bytes) -> str:
        """Save content to disk under a path derived from the URL (hash). Return file path."""
        
        url_hash = md5(url.encode('utf-8')).hexdigest()
        filename = f"{url_hash}.html"
        file_path = os.path.join(self.pages_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(content)

        self.mapping[url_hash] = url
        try:
            with open(self.mapping_path, 'w', encoding='utf-8') as f:
                json.dump(self.mapping, f, indent=2)
        except Exception:
            pass
        
        return file_path
