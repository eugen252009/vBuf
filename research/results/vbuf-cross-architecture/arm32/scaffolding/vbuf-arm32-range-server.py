import argparse
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class RangeHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        size = os.stat(self.server.path).st_size
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        with open(self.server.path, "rb") as source:
            size = os.fstat(source.fileno()).st_size
            start, end = (int(part) for part in self.headers["Range"][6:].split("-", 1))
            if start < 0 or end < start or end >= size:
                self.send_response(416)
                self.end_headers()
                return
            source.seek(start)
            body = source.read(end - start + 1)
            self.send_response(206)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True)
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()
server = ThreadingHTTPServer(("0.0.0.0", args.port), RangeHandler)
server.path = args.file
server.serve_forever()
