"""Fetch only the pinned kitchen engine and recipe required by submitted examples."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'task_support/kitchen_resources.json').read_text(encoding='utf-8'))
    destination = ROOT/'task_support/sources/collab/upstream'
    for entry in manifest['files']:
        path = destination/entry['path']
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']:
            continue
        with urllib.request.urlopen(entry['url'], timeout=30) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError('Upstream content changed: ' + entry['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    print(json.dumps({'status':'ready','commit':manifest['commit'],'files':len(manifest['files'])}))


if __name__ == '__main__':
    main()
