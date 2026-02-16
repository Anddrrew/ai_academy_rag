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

DOCKER_SERVICES = [
    {"name": "WebUI", "port": 3000, "url": "http://localhost:3000"},
    {"name": "Qdrant", "port": 6333, "url": "http://localhost:6333/dashboard"},
]


def wait_for_url(url: str, name: str, statuses: dict, timeout: int = 120, interval: int = 2):
    statuses[name] = f"Waiting for {url.split('/')[2]}..."
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(interval)
    statuses[name] = "Timed out"
    sys.exit(1)


def run_service(service: dict, statuses: dict):
    log_file = open(os.path.join(LOGS_DIR, f"{service['name']}.log"), "w")
    sys.stdout = log_file
    sys.stderr = log_file

    if wait_url := service.get("wait_for"):
        wait_for_url(wait_url, service["name"], statuses)

    statuses[service["name"]] = "Starting..."

    for p in service["pythonpath"]:
        if p not in sys.path:
            sys.path.insert(0, p)

    for key, value in service.get("env", {}).items():
        os.environ[key] = value

    statuses[service["name"]] = "Running"

    uvicorn.run(
        service["app"],
        host="0.0.0.0",
        port=service["port"],
        log_level="info",
    )

    statuses[service["name"]] = "Stopped"


def print_table(statuses: dict):
    rows = []
    for svc in DOCKER_SERVICES:
        rows.append((svc["name"], svc["port"], "Docker", svc["url"], ""))
    for svc in SERVICES:
        url = f"http://localhost:{svc['port']}/docs"
        status = statuses.get(svc["name"], "Pending")
        rows.append((svc["name"].capitalize(), svc["port"], "Python", url, status))

    header = f"{'Name':<12} {'Port':<6} {'Runtime':<10} {'Status':<30} {'URL'}"
    lines = [header, "-" * len(header)]
    for name, port, runtime, url, status in rows:
        lines.append(f"{name:<12} {port:<6} {runtime:<10} {status:<30} {url}")

    return lines


def refresh_table(statuses: dict, total_lines: int):
    sys.stdout.write(f"\033[{total_lines}A")
    for line in print_table(statuses):
        sys.stdout.write(f"\033[2K{line}\n")
    sys.stdout.flush()


def main():
    manager = multiprocessing.Manager()
    statuses = manager.dict()

    for svc in SERVICES:
        statuses[svc["name"]] = "Pending"

    lines = print_table(statuses)
    for line in lines:
        print(line)
    total_lines = len(lines)

    processes: list[multiprocessing.Process] = []
    for svc in SERVICES:
        p = multiprocessing.Process(target=run_service, args=(svc, statuses), name=svc["name"])
        p.start()
        processes.append(p)

    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
        for svc in SERVICES:
            statuses[svc["name"]] = "Stopped"
        refresh_table(statuses, total_lines)
        manager.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while running:
        refresh_table(statuses, total_lines)

        if all(not p.is_alive() for p in processes):
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
