import multiprocessing
import signal
import sys
import os
import time
import urllib.request
import urllib.error

import uvicorn


ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

SERVICES = [
    {
        "app": "main:app",
        "port": 3003,
        "name": "embedder",
        "pythonpath": [
            os.path.join(ROOT, "packages", "embedder", "src"),
            os.path.join(ROOT, "packages"),
        ],
    },
    {
        "app": "main:app",
        "port": 3002,
        "name": "indexer",
        "pythonpath": [
            os.path.join(ROOT, "packages", "indexer", "src"),
            os.path.join(ROOT, "packages"),
        ],
        "wait_for": "http://localhost:3003/status",
    },
    {
        "app": "main:app",
        "port": 3001,
        "name": "chat",
        "pythonpath": [
            os.path.join(ROOT, "packages", "chat", "src"),
            os.path.join(ROOT, "packages"),
        ],
        "wait_for": "http://localhost:3003/status",
    },
]


def wait_for_url(url: str, name: str, timeout: int = 120, interval: int = 2):
    print(f"[{name}] Waiting for {url} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                print(f"[{name}] {url} is ready")
                return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(interval)
    print(f"[{name}] Timed out waiting for {url}")
    sys.exit(1)


def run_service(service: dict):
    log_file = open(os.path.join(LOGS_DIR, f"{service['name']}.log"), "w")
    sys.stdout = log_file
    sys.stderr = log_file

    if wait_url := service.get("wait_for"):
        wait_for_url(wait_url, service["name"])

    for p in service["pythonpath"]:
        if p not in sys.path:
            sys.path.insert(0, p)

    for key, value in service.get("env", {}).items():
        os.environ[key] = value

    uvicorn.run(
        service["app"],
        host="0.0.0.0",
        port=service["port"],
        log_level="info",
    )


def main():
    processes: list[multiprocessing.Process] = []

    for svc in SERVICES:
        p = multiprocessing.Process(target=run_service, args=(svc,), name=svc["name"])
        p.start()
        print(f"Started {svc['name']} (pid={p.pid}) on port {svc['port']}")
        processes.append(p)

    def shutdown(sig, frame):
        print(f"\nReceived {signal.Signals(sig).name}, shutting down...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
