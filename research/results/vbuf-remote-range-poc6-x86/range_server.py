#!/usr/bin/env python3
import argparse
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(BaseHTTPRequestHandler):
    server_version = "vbuf-range-server/1"

    def do_GET(self):
        mode = self.server.mode
        if mode == "status500":
            self.send_response(500)
            self.end_headers()
            return
        with open(self.server.path, "rb") as source:
            size = os.fstat(source.fileno()).st_size
            requested = self.headers.get("Range", "")
            if mode == "ignore-range" or not requested.startswith("bytes="):
                self.send_response(200)
                self.send_header("Content-Length", str(size))
                self.end_headers()
                return
            start, end = requested[6:].split("-", 1)
            start = int(start)
            end = int(end)
            if start < 0 or end < start or end >= size:
                self.send_response(416)
                self.end_headers()
                return
            source.seek(start)
            body = source.read(end - start + 1)
            if mode == "truncate":
                body = body[:-1]
            advertised_start = start + 1 if mode == "wrong-range" else start
            self.send_response(206)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {advertised_start}-{end}/{size}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, _format, *args):
        pass


class RangeServer(ThreadingHTTPServer):
    path: str
    mode: str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", default="range",
        choices=["range", "status500", "ignore-range", "truncate", "wrong-range"])
    args = parser.parse_args()
    server = RangeServer(("127.0.0.1", args.port), RangeHandler)
    server.path = args.file
    server.mode = args.mode
    server.serve_forever()


if __name__ == "__main__":
    main()
