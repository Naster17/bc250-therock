# SPDX-License-Identifier: MIT
"""Exercise actual sockets and source files, including corrupt/truncated replies."""
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
from reference_transport_d40 import (
    CHUNK,
    MAX_BLOB,
    RemoteStore,
    ReferenceServer,
    fetch_manifest,
    items,
    receive,
)


class TransportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.data = b"pinned-reference" * 80000
        self.item = dict(
            file="000-initial.raw",
            bytes=len(self.data),
            sha256=hashlib.sha256(self.data).hexdigest(),
        )
        self.other = dict(self.item, file="001-gradient.raw")
        for item in [self.item, self.other]:
            (self.source / item["file"]).write_bytes(self.data)
        self.manifest = dict(tensors=[dict(initial=self.item, gradient=self.other)])
        self.metadata = json.dumps(self.manifest).encode()
        self.pin = hashlib.sha256(self.metadata).hexdigest()
        self.server = ReferenceServer(
            ("127.0.0.1", 0), self.source, self.metadata, ["127.0.0.1"]
        )
        self.start(self.server)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.path = self.root / "descriptor/manifest.json"
        fetch_manifest(self.endpoint, self.path, self.pin, len(self.metadata))
        self.store = RemoteStore(self.endpoint, self.path, self.pin)

    def start(self, server):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 3)
        self.addCleanup(server.shutdown)

    def test_manifest_only_roundtrip_and_bounded_verification(self):
        for item in [self.item, self.other]:
            self.store.verify(self.path.parent, item)
            self.assertEqual(self.store.read(self.path.parent, item), self.data)
        self.assertEqual(self.store.stats["verified_bytes"], len(self.data) * 2)
        self.assertEqual(self.store.stats["read_bytes"], len(self.data) * 2)
        self.assertEqual(self.store.stats["max_retained_blob_bytes"], len(self.data))
        self.assertEqual(self.store.stats["verify_chunk_bytes"], CHUNK)
        self.assertEqual(self.store.stats["disk_cache_bytes"], 0)
        self.assertFalse(self.store.stats["local_fallback_used"])
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.server.shutdown()  # Drain the final handler before reading its counters.
        self.assertEqual(self.server.stats["completed_requests"], 5)

    def test_source_bytes_corrupted_after_server_start(self):
        (self.source / self.item["file"]).write_bytes(b"x" * len(self.data))
        with self.assertRaises(ValueError):
            self.store.read(self.path.parent, self.item)
        self.assertEqual(self.store.stats["read_blobs"], 0)

    def test_source_truncated_after_server_start(self):
        (self.source / self.item["file"]).write_bytes(self.data[:-1])
        with self.assertRaises(http.client.HTTPException):
            self.store.read(self.path.parent, self.item)

    def test_source_symlink_refused(self):
        path = self.source / self.item["file"]
        path.unlink()
        path.symlink_to(self.source / self.other["file"])
        with self.assertRaises(http.client.HTTPException):
            self.store.read(self.path.parent, self.item)

    def test_paths_methods_and_peer_refused(self):
        for method, path, status in [
            ("GET", "/../001-gradient.raw", 404),
            ("GET", "/%30%30%30-initial.raw", 404),
            ("GET", "/unknown.raw", 404),
            ("POST", "/000-initial.raw", 501),
        ]:
            connection = http.client.HTTPConnection(
                "127.0.0.1", self.server.server_port
            )
            connection.request(method, path)
            self.assertEqual(connection.getresponse().status, status)
            connection.close()
        self.server.allowed.clear()
        with self.assertRaises(ValueError):
            self.store.read(self.path.parent, self.item)

    def test_untrusted_response_protocol(self):
        item = dict(file="x.raw", bytes=4, sha256=hashlib.sha256(b"abcd").hexdigest())
        for case in [
            "truncated",
            "content",
            "length",
            "redirect",
            "header",
            "encoding",
        ]:
            with self.subTest(case=case):

                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, *args):
                        pass

                    def do_GET(self):
                        self.send_response(302 if case == "redirect" else 200)
                        self.send_header(
                            "Content-Length", "5" if case == "length" else "4"
                        )
                        self.send_header(
                            "X-Content-SHA256",
                            "0" * 64 if case == "header" else item["sha256"],
                        )
                        if case == "encoding":
                            self.send_header("Content-Encoding", "gzip")
                        self.end_headers()
                        self.wfile.write(
                            b"abc"
                            if case == "truncated"
                            else b"efgh" if case == "content" else b"abcd"
                        )

                server = HTTPServer(("127.0.0.1", 0), Handler)
                self.start(server)
                with self.assertRaises((ValueError, http.client.HTTPException)):
                    receive(f"http://127.0.0.1:{server.server_port}", item, retain=True)

    def test_pins_scope_and_oversized_allocations_refused(self):
        with self.assertRaises(ValueError):
            RemoteStore(self.endpoint, self.path, "0" * 64)
        with self.assertRaises(ValueError):
            self.store.read(self.source, self.item)
        with self.assertRaises(ValueError):
            self.store.read(self.path.parent, dict(self.item, sha256="0" * 64))
        for malformed in [
            dict(self.item, bytes=MAX_BLOB + 1),
            dict(self.item, file="../000-initial.raw"),
        ]:
            with self.assertRaises(ValueError):
                items(dict(tensors=[dict(initial=malformed, gradient=self.other)]))
        with self.assertRaises(ValueError):
            items(dict(tensors=[dict(initial=self.item, gradient=self.item)]))


if __name__ == "__main__":
    unittest.main()
