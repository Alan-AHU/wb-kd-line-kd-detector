from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import ssl
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EXCLUDE_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
}

EXCLUDE_SUFFIX = {
    ".pyc",
    ".pyo",
    ".pyd",
}


def _api_request(method: str, url: str, token: str, data: dict | None = None) -> dict:
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")

    max_retries = 6
    base_sleep = 1.0

    for attempt in range(1, max_retries + 1):
        req = Request(url=url, method=method, data=body)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "kd-line-kd-detector-uploader")
        if body is not None:
            req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req, timeout=45) as resp:
                payload = resp.read().decode("utf-8")
                return json.loads(payload) if payload else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            retryable_http = exc.code in {429, 500, 502, 503, 504}
            if retryable_http and attempt < max_retries:
                sleep_s = base_sleep * (2 ** (attempt - 1))
                print(f"Transient HTTP {exc.code}, retrying in {sleep_s:.1f}s ...")
                time.sleep(sleep_s)
                continue
            raise RuntimeError(f"GitHub API error {exc.code}: {detail}") from exc
        except URLError as exc:
            reason_text = str(exc.reason).lower()
            transient_network = (
                isinstance(exc.reason, (socket.timeout, TimeoutError, ssl.SSLError, OSError))
                or "timed out" in reason_text
                or "temporary failure" in reason_text
                or "connection reset" in reason_text
                or "unexpected eof" in reason_text
                or "eof occurred" in reason_text
            )
            if transient_network and attempt < max_retries:
                sleep_s = base_sleep * (2 ** (attempt - 1))
                print(f"Transient network error, retrying in {sleep_s:.1f}s ...")
                time.sleep(sleep_s)
                continue
            raise RuntimeError(f"Network error: {exc}") from exc

    raise RuntimeError("Unexpected request failure after retries")


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_NAMES for part in path.parts):
            continue
        if path.suffix.lower() in EXCLUDE_SUFFIX:
            continue
        yield path


def create_repo(owner: str, repo: str, token: str, private: bool) -> None:
    url = "https://api.github.com/user/repos"
    data = {
        "name": repo,
        "private": private,
        "auto_init": False,
        "description": "Detect gray/black blot bands and convert y to kD",
    }
    try:
        _api_request("POST", url, token, data)
        print(f"Created repository: {owner}/{repo}")
    except RuntimeError as exc:
        msg = str(exc)
        if "422" in msg and "name already exists" in msg:
            print(f"Repository already exists: {owner}/{repo}")
            return
        raise


def repo_exists(owner: str, repo: str, token: str) -> bool:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        _api_request("GET", url, token)
        return True
    except RuntimeError as exc:
        if "404" in str(exc):
            return False
        raise


def get_current_file_sha(owner: str, repo: str, branch: str, rel_path: str, token: str) -> str | None:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{rel_path}?ref={branch}"
    try:
        data = _api_request("GET", url, token)
        return data.get("sha")
    except RuntimeError as exc:
        if "404" in str(exc):
            return None
        raise


def upload_file(
    owner: str,
    repo: str,
    branch: str,
    rel_path: str,
    file_path: Path,
    token: str,
    message: str,
) -> None:
    content = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    sha = get_current_file_sha(owner, repo, branch, rel_path, token)

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{rel_path}"
    data = {
        "message": message,
        "content": content,
        "branch": branch,
    }
    if sha is not None:
        data["sha"] = sha

    _api_request("PUT", url, token, data)


def get_default_branch(owner: str, repo: str, token: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    data = _api_request("GET", url, token)
    return data.get("default_branch", "main")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and upload this project to GitHub (without git).")
    parser.add_argument("--owner", required=True, help="GitHub username/owner")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root directory")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument(
        "--skip-create",
        action="store_true",
        help="Skip repository creation (use when token has no create-repo permission)",
    )
    parser.add_argument(
        "--message",
        default="Upload kd-line-kd-detector project",
        help="Commit message used by GitHub Contents API",
    )
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing GITHUB_TOKEN environment variable.")

    root = args.root.resolve()
    if not args.skip_create:
        try:
            create_repo(args.owner, args.repo, token, private=args.private)
        except RuntimeError as exc:
            msg = str(exc)
            if "403" in msg and "Resource not accessible by personal access token" in msg:
                print(
                    "No permission to create repository with current token. "
                    "Create the repo manually, then rerun with --skip-create."
                )
            raise
    else:
        if not repo_exists(args.owner, args.repo, token):
            raise RuntimeError(
                f"Repository not found: {args.owner}/{args.repo}. "
                "Create it first, or rerun without --skip-create."
            )
        print(f"Using existing repository: {args.owner}/{args.repo}")

    branch = get_default_branch(args.owner, args.repo, token)

    files = list(iter_files(root))
    files.sort(key=lambda p: str(p))
    total = len(files)
    if total == 0:
        raise RuntimeError(f"No files to upload under: {root}")

    print(f"Uploading {total} files to {args.owner}/{args.repo}@{branch} ...")
    for i, fp in enumerate(files, start=1):
        rel_path = fp.relative_to(root).as_posix()
        try:
            upload_file(
                owner=args.owner,
                repo=args.repo,
                branch=branch,
                rel_path=rel_path,
                file_path=fp,
                token=token,
                message=args.message,
            )
            print(f"[{i}/{total}] {rel_path}")
        except RuntimeError as exc:
            msg = str(exc)
            print(f"Failed at file: {rel_path}")
            if "403" in msg and "Resource not accessible by personal access token" in msg:
                print(
                    "Token can read the repository but cannot write file contents. "
                    "Grant 'Contents: Read and write' on this repository."
                )
            raise

    print(f"Done: https://github.com/{args.owner}/{args.repo}")


if __name__ == "__main__":
    main()
