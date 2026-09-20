"""Expose an SSH-forwarded Unix socket on remote loopback only, for HTTP clients."""
import argparse
import select
import socket
import socketserver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--port", type=int, default=17891)
    args = parser.parse_args()

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
                    upstream.connect(args.socket)
                    peers = {self.request: upstream, upstream: self.request}
                    while True:
                        ready, _, _ = select.select(list(peers), [], [], 120)
                        if not ready:
                            return
                        for source in ready:
                            data = source.recv(65536)
                            if not data:
                                return
                            peers[source].sendall(data)
            except (OSError, ConnectionError):
                return

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server(("127.0.0.1", args.port), Handler) as server:
        print(f"HTTP proxy bridge listening on 127.0.0.1:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
