#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded read-only fixture transport; explicit SHA-pinned reads, never a cache."""
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import stat
import time
from urllib.parse import urlsplit
import artifact_d35 as artifact

CHUNK = 1024**2
MAX_MANIFEST = 256 * 1024
MAX_BLOB = 128 * 1024**2


def items(manifest):
    result = {}
    for entry in manifest["tensors"]:
        for key in ["initial", "gradient"]:
            item = entry[key]
            artifact.safe_path(Path("."), item["file"])
            if (
                item["file"] in result
                or type(item["bytes"]) is not int
                or not 0 < item["bytes"] <= MAX_BLOB
                or len(item["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in item["sha256"])
            ):
                raise ValueError("Invalid or oversized reference blob")
            result[item["file"]] = item
    return result


def receive(endpoint, item, *, retain):
    url = urlsplit(endpoint)
    if (
        url.scheme != "http"
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ["", "/"]
        or not url.hostname
    ):
        raise ValueError("Expected explicit direct HTTP fixture endpoint")
    limit = MAX_MANIFEST if item["file"] == "manifest.json" else MAX_BLOB
    if type(item["bytes"]) is not int or not 0 < item["bytes"] <= limit:
        raise ValueError("Unbounded reference response refused")
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=20)
    try:
        conn.request("GET", "/" + item["file"], headers={"Connection": "close"})
        response = conn.getresponse()
        if (
            response.status != 200
            or response.getheader("Content-Length") != str(item["bytes"])
            or response.getheader("X-Content-SHA256") != item["sha256"]
            or response.getheader("Transfer-Encoding") is not None
            or response.getheader("Content-Encoding") is not None
        ):
            raise ValueError("Reference status/length/digest/encoding refused")
        data = bytearray(item["bytes"] if retain else min(CHUNK, item["bytes"]))
        view = memoryview(data)
        checksum, consumed = hashlib.sha256(), 0
        deadline = time.monotonic() + 60
        while consumed < item["bytes"]:
            if time.monotonic() > deadline:
                raise TimeoutError("Reference blob deadline exceeded")
            start = consumed if retain else 0
            part = view[start : start + min(CHUNK, item["bytes"] - consumed)]
            count = response.readinto(part)
            if not count:
                raise ValueError("Truncated reference response")
            checksum.update(part[:count])
            consumed += count
        if response.read(1) or checksum.hexdigest() != item["sha256"]:
            raise ValueError("Reference content digest changed")
        return data if retain else None
    finally:
        conn.close()


class RemoteStore:
    def __init__(self, endpoint, manifest_path, pin):
        self.endpoint, self.root = endpoint, Path(manifest_path).parent
        if artifact.digest(manifest_path) != pin:
            raise ValueError("Reference manifest digest changed")
        self.table = items(json.loads(Path(manifest_path).read_text()))
        self.stats = dict(
            endpoint=endpoint,
            manifest_sha256=pin,
            verified_blobs=0,
            read_blobs=0,
            verified_bytes=0,
            read_bytes=0,
            max_retained_blob_bytes=0,
            verify_chunk_bytes=CHUNK,
            disk_cache_bytes=0,
            local_fallback_used=False,
        )

    def request(self, root, item, *, retain):
        if Path(root) != self.root or self.table.get(item["file"]) != item:
            raise ValueError("Reference outside pinned manifest refused")
        data = receive(self.endpoint, item, retain=retain)
        key = "read" if retain else "verified"
        self.stats[key + "_blobs"] += 1
        self.stats[key + "_bytes"] += item["bytes"]
        if retain:
            self.stats["max_retained_blob_bytes"] = max(
                self.stats["max_retained_blob_bytes"], item["bytes"]
            )
        return data

    def verify(self, root, item):
        self.request(root, item, retain=False)

    def read(self, root, item):
        return self.request(root, item, retain=True)

    def validate(self, path, pin, config, parameters):
        original = artifact.verify_blob
        artifact.verify_blob = self.verify
        try:
            return artifact.load_manifest(path, pin, config, parameters)
        finally:
            artifact.verify_blob = original


def fetch_manifest(endpoint, destination, pin, size):
    # Its length and SHA come from the qualified original CPU producer, not HTTP.
    item = dict(file="manifest.json", bytes=size, sha256=pin)
    data = receive(endpoint, item, retain=True)
    Path(destination).parent.mkdir()
    with Path(destination).open("xb") as stream:
        stream.write(data)


class ReferenceServer(HTTPServer):
    def __init__(self, address, directory, manifest_data, allowed):
        self.directory = Path(directory)
        self.manifest_data = manifest_data
        self.table = items(json.loads(manifest_data))
        self.allowed = set(allowed)
        self.stats = dict(
            completed_requests=0,
            completed_bytes=0,
            rejected_requests=0,
            failed_requests=0,
            max_chunk_bytes=CHUNK,
            blob_counts={},
        )
        super().__init__(address, Handler)
        self.timeout = 0.5


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(20)

    def do_GET(self):
        server = self.server
        name = self.path[1:]
        if self.client_address[0] not in server.allowed or (
            name != "manifest.json" and name not in server.table
        ):
            server.stats["rejected_requests"] += 1
            self.send_error(404)
            return
        descriptor = None
        try:
            if name == "manifest.json":
                data = server.manifest_data
                size, sha = len(data), hashlib.sha256(data).hexdigest()
            else:
                item = server.table[name]
                descriptor = os.open(
                    server.directory / name, os.O_RDONLY | os.O_NOFOLLOW
                )
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_size != item["bytes"]:
                    raise ValueError("Source file type/size changed")
                size, sha = item["bytes"], item["sha256"]
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.send_header("X-Content-SHA256", sha)
            self.end_headers()
            if descriptor is None:
                self.wfile.write(data)
            else:
                remaining = size
                while remaining:
                    part = os.read(descriptor, min(CHUNK, remaining))
                    if not part:
                        raise ValueError("Source truncated during response")
                    self.wfile.write(part)
                    remaining -= len(part)
                    del part
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            server.stats["completed_requests"] += 1
            server.stats["completed_bytes"] += size
            counts = server.stats["blob_counts"]
            counts[name] = counts.get(name, 0) + 1
        except (OSError, ValueError):
            server.stats["failed_requests"] += 1
            self.close_connection = True
        finally:
            if descriptor is not None:
                os.close(descriptor)
