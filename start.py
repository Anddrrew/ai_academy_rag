import json
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

SPINNER = ["[=     ]", "[ =    ]", "[  =   ]", "[   =  ]", "[    = ]", "[     =]", "[    = ]", "[   =  ]", "[  =   ]", "[ =    ]"]
RENDER_INTERVAL = 0.15
HEALTH_INTERVAL = 1.0
START_TIME = time.time()

SERVICES = [
    {
        "app": "main:app",
        "port": 3003,
        "name": "embedder",
        "pythonpath": [
            os.path.join(ROOT, "packages", "embedder", "src"),
            os.path.join(ROOT, "packages"),
        ],
        "health_url": "http://localhost:3003/status",
    },
    {
        "app": "main:app",
        "port": 3002,
        "name": "indexer",
        "pythonpath": [
            os.path.join(ROOT, "packages", "indexer", "src"),
            os.path.join(ROOT, "packages"),
        ],
        "wait_for": "embedder",
        "health_url": "http://localhost:3002/status",
    },
    {
        "app": "main:app",
        "port": 3001,
        "name": "chat",
        "pythonpath": [
            os.path.join(ROOT, "packages", "chat", "src"),
            os.path.join(ROOT, "packages"),
        ],
        "wait_for": "embedder",
        "health_url": "http://localhost:3001/status",
    },
]

DOCKER_SERVICES = [
    {"name": "WebUI", "port": 3000, "url": "http://localhost:3000", "health_url": "http://localhost:3000/health"},
    {"name": "Qdrant", "port": 6333, "url": "http://localhost:6333/dashboard", "health_url": "http://localhost:6333/healthz"},
]


def _resolve_health_url(service_name: str) -> str:
    for svc in SERVICES:
        if svc["name"] == service_name:
            return svc["health_url"]
    raise ValueError(f"Unknown service: {service_name}")


def wait_for_service(dependency: str, name: str, statuses: dict, timeout: int = 120, interval: int = 2):
    url = _resolve_health_url(dependency)
    statuses[name] = f"Waiting [{dependency.capitalize()}]"
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
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())
    sys.stdout = log_file
    sys.stderr = log_file

    if dependency := service.get("wait_for"):
        wait_for_service(dependency, service["name"], statuses)

    for p in service["pythonpath"]:
        if p not in sys.path:
            sys.path.insert(0, p)

    for key, value in service.get("env", {}).items():
        os.environ[key] = value

    statuses[service["name"]] = "Starting..."

    uvicorn.run(
        service["app"],
        host="0.0.0.0",
        port=service["port"],
        log_level="info",
    )

    statuses[service["name"]] = "Stopped"


def check_health(url: str) -> str | None:
    """Fetch the health URL and return the status field from the JSON response, or None on failure."""
    try:
        resp = urllib.request.urlopen(url, timeout=2)
        if resp.status == 200:
            body = json.loads(resp.read().decode())
            return body.get("status", "ok")
    except (urllib.error.URLError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        pass
    return None


def check_docker_health(url: str) -> bool:
    try:
        resp = urllib.request.urlopen(url, timeout=2)
        return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def update_health_statuses(statuses: dict):
    for svc in DOCKER_SERVICES:
        health_url = svc.get("health_url")
        if not health_url:
            continue
        name = svc["name"]
        if check_docker_health(health_url):
            statuses[name] = "Running"
        else:
            statuses[name] = "Unavailable"

    for svc in SERVICES:
        health_url = svc.get("health_url")
        if not health_url:
            continue
        name = svc["name"]
        current = statuses.get(name, "")
        if current in ("Pending", "Stopped", "Timed out"):
            continue
        health_status = check_health(health_url)
        if health_status is not None:
            statuses[name] = f"Running [{health_status}]"
        elif current.startswith("Running"):
            statuses[name] = "Unhealthy"


def format_uptime(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def print_table(statuses: dict, tick: int = 0):
    elapsed = time.time() - START_TIME
    spin = SPINNER[tick % len(SPINNER)]

    # Column widths
    W_NAME, W_PORT, W_RT, W_STATUS, W_URL = 14, 8, 10, 22, 36

    rows = []
    for svc in DOCKER_SERVICES:
        status = statuses.get(svc["name"], "Checking...")
        rows.append((svc["name"], str(svc["port"]), "Docker", status, svc["url"]))
    for svc in SERVICES:
        url = f"http://localhost:{svc['port']}/docs"
        status = statuses.get(svc["name"], "Pending")
        rows.append((svc["name"].capitalize(), str(svc["port"]), "Python", status, url))

    def row_line(n, p, r, s, u):
        return f"| {n:<{W_NAME}}| {p:<{W_PORT}}| {r:<{W_RT}}| {s:<{W_STATUS}}| {u:<{W_URL}}|"

    sep = f"+{'':-<{W_NAME + 1}}+{'':-<{W_PORT + 1}}+{'':-<{W_RT + 1}}+{'':-<{W_STATUS + 1}}+{'':-<{W_URL + 1}}+"

    lines = [sep]
    lines.append(row_line("Name", "Port", "Runtime", "Status", "URL"))
    lines.append(sep)
    for name, port, runtime, status, url in rows:
        lines.append(row_line(name, port, runtime, status, url))
    lines.append(sep)
    lines.append("")
    lines.append(f"  RAG Services {spin}  (uptime: {format_uptime(elapsed)})")
    lines.append("  Press Ctrl+C to stop all services. Logs: ./logs/<service>.log")

    return lines


def save_cursor():
    sys.stdout.write("\033[s")
    sys.stdout.flush()


def refresh_table(statuses: dict, total_lines: int, tick: int = 0):
    sys.stdout.write("\033[u")
    for line in print_table(statuses, tick):
        sys.stdout.write(f"\033[2K{line}\n")
    sys.stdout.flush()


def main():
    manager = multiprocessing.Manager()
    statuses = manager.dict()

    for svc in DOCKER_SERVICES:
        statuses[svc["name"]] = "Checking..."
    for svc in SERVICES:
        statuses[svc["name"]] = "Pending"

    save_cursor()
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
    tick = 0
    last_health_check = 0.0

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
        refresh_table(statuses, total_lines, tick)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while running:
        now = time.time()
        if now - last_health_check >= HEALTH_INTERVAL:
            update_health_statuses(statuses)
            last_health_check = now

        refresh_table(statuses, total_lines, tick)
        tick += 1

        if all(not p.is_alive() for p in processes):
            break
        time.sleep(RENDER_INTERVAL)

    manager.shutdown()


if __name__ == "__main__":
    main()
