#!/usr/bin/env python3
"""Local application-level HTTP capture proxy for the Pi A/B qualification."""

from __future__ import annotations

import http.client
import json
import queue
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class CaptureProxy(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], backend_port: int, events: queue.Queue):
        super().__init__(address, CaptureHandler)
        self.backend_port = backend_port
        self.events = events
        self.exchange_id = 0


class CaptureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: CaptureProxy

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def do_OPTIONS(self) -> None:
        self._forward()

    def _forward(self) -> None:
        started = time.monotonic_ns()
        length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(length) if length else b""
        self.server.exchange_id += 1
        exchange_id = self.server.exchange_id
        request_headers = {
            key.lower(): value
            for key, value in self.headers.items()
            if key.lower() not in {"authorization", "connection", "host"}
        }
        request_json: Any = None
        if request_body:
            try:
                request_json = json.loads(request_body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                request_json = {"base64": __import__("base64").b64encode(request_body).decode("ascii")}

        upstream = http.client.HTTPConnection("127.0.0.1", self.server.backend_port, timeout=900)
        response_body = bytearray()
        try:
            upstream_headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"authorization", "connection", "host", "transfer-encoding"}
            }
            upstream.request(self.command, self.path, body=request_body or None, headers=upstream_headers)
            response = upstream.getresponse()
            response_headers = {
                key.lower(): value
                for key, value in response.getheaders()
                if key.lower() in {"content-type", "cache-control", "x-request-id"}
            }
            self.send_response(response.status, response.reason)
            for key, value in response_headers.items():
                self.send_header(key, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                response_body.extend(chunk)
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

            try:
                response_content: Any = bytes(response_body).decode("utf-8")
            except UnicodeDecodeError:
                response_content = {"base64": __import__("base64").b64encode(response_body).decode("ascii")}
            self.server.events.put(("http_exchange", {
                "kind": "http_exchange",
                "id": exchange_id,
                "method": self.command,
                "path": self.path,
                "request_headers": request_headers,
                "request_body": request_json,
                "response_status": response.status,
                "response_headers": response_headers,
                "response_body": response_content,
                "elapsed_ms": (time.monotonic_ns() - started) / 1_000_000,
            }))
        except Exception as error:
            try:
                body = json.dumps({"error": str(error)}).encode("utf-8")
                self.send_response(502, "Bad Gateway")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.server.events.put(("http_proxy_error", {
                "kind": "http_proxy_error",
                "id": exchange_id,
                "method": self.command,
                "path": self.path,
                "error": repr(error),
            }))
        finally:
            upstream.close()


def start_proxy(backend_port: int, events: queue.Queue) -> CaptureProxy:
    return CaptureProxy(("127.0.0.1", 18180), backend_port, events)
