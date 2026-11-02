#!/usr/bin/env python3
"""Download an immutable Hugging Face snapshot with resume and hash checks.

Uses the Python standard library, so preparation does not install a
model framework on the transfer computer. Files are published only after their
upstream LFS SHA256 or Git blob SHA1 has been checked.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def digest(path: Path, algorithm: str, *, git_blob: bool = False) -> str:
    """Hash a file without loading model shards into memory."""
    value = hashlib.new(algorithm)
    if git_blob:
        value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """Resolve a revision once, download its files, and write transfer hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--authenticated", action="store_true", help="use the token saved by hf auth login")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    api = f"https://huggingface.co/api/models/{args.repo}/revision/{args.revision}?blobs=true"
    token = None
    if args.authenticated:
        from huggingface_hub import get_token

        token = get_token()
        if not token:
            raise RuntimeError("run hf auth login before using --authenticated")
    api_request = urllib.request.Request(api)
    if token:
        # Do not forward credentials to redirected storage/CDN hosts.
        api_request.add_unredirected_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(api_request, timeout=30) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    files = metadata["siblings"]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "download-manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if (previous["repo"], previous["revision"]) != (args.repo, revision):
            raise RuntimeError("destination already belongs to a different model revision")
    manifest.write_text(json.dumps({"repo": args.repo, "revision": revision, "files": files}, indent=2) + "\n")
    print(
        json.dumps(
            {"event": "start", "repo": args.repo, "revision": revision, "bytes": sum(item["size"] for item in files)}
        ),
        flush=True,
    )

    def download(item: dict) -> tuple[str, str]:
        name = item["rfilename"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe snapshot path: {name!r}")
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        candidate = path if path.exists() else partial
        if not candidate.exists() or candidate.stat().st_size != item["size"]:
            if path.exists():
                raise RuntimeError(f"existing file has incorrect size: {path}")
            url = f"https://huggingface.co/{args.repo}/resolve/{revision}/{urllib.parse.quote(name)}"
            for attempt in range(9):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == item["size"]:
                    break
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
                if token:
                    request.add_unredirected_header("Authorization", f"Bearer {token}")
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        if offset and response.status != 206:
                            raise RuntimeError("server did not honor the resume range")
                        if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise RuntimeError("server returned an unexpected resume range")
                        with partial.open("ab" if offset else "wb") as output:
                            while block := response.read(1024 * 1024):
                                output.write(block)
                    if partial.stat().st_size == item["size"]:
                        break
                except (OSError, TimeoutError, RuntimeError) as error:
                    if attempt == 8:
                        raise
                    print(
                        json.dumps({"event": "retry", "file": name, "attempt": attempt + 1, "error": str(error)}),
                        flush=True,
                    )
                    time.sleep(min(2**attempt, 30))
            candidate = partial
        if candidate.stat().st_size != item["size"]:
            raise RuntimeError(f"incomplete download: {name}")
        sha256 = digest(candidate, "sha256")
        lfs = item.get("lfs")
        valid = sha256 == lfs["sha256"] if lfs else digest(candidate, "sha1", git_blob=True) == item["blobId"]
        if not valid:
            raise RuntimeError(f"upstream hash mismatch: {name}")
        if candidate != path:
            candidate.replace(path)
        print(json.dumps({"event": "verified", "file": name, "bytes": item["size"]}), flush=True)
        return name, sha256

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(download, files))
    (args.output / "SHA256SUMS").write_text("".join(f"{sha}  {name}\n" for name, sha in sorted(results)))
    print(json.dumps({"event": "complete", "repo": args.repo, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
