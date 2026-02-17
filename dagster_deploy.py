"""
Dagster Pipeline: Docker + Cloudflare Tunnel Auto-Deploy
=========================================================
Automates the full lifecycle of a containerized web application:
  build image → run container → publish via Cloudflare Tunnel → register DNS

Each deployment step is modelled as a Dagster asset, giving full observability
over the pipeline in the Dagster UI (logs, metadata, dependency graph).

Asset Groups
------------
cloudflare_deploy   — builds, runs, and publishes the application:
    docker_create   : Creates project directory with app.py and Dockerfile.
    docker_build    : Builds the Docker image from the project directory.
    find_port       : Queries live tunnel config to find an unused host port.
    docker_run      : Runs the container on the assigned port.
    tunnel_route    : Adds/updates the Cloudflare Tunnel ingress rule.
    dns_record      : Creates a proxied CNAME DNS record for the hostname.

cloudflare_teardown — reverses all changes made during deployment:
    remove_dns_record     : Deletes the CNAME DNS record.
    remove_tunnel_route   : Removes the tunnel ingress rule.
    docker_stop           : Stops the container and cleans up the port file.

Dependency Graph (deploy)
-------------------------
    docker_create
        └── docker_build
                ├── find_port
                │       └── docker_run
                │               └── tunnel_route
                │                       └── dns_record
                └── docker_run  (also depends on find_port)

Dependency Graph (teardown)
---------------------------
    remove_dns_record
        └── remove_tunnel_route
                └── docker_stop

Inter-asset Port Communication
-------------------------------
The find_port asset writes the selected port to ~/.cf_deploy_port.
Downstream assets (docker_run, tunnel_route) read this file via
_read_assigned_port(). The file is deleted during docker_stop teardown.

Usage
-----
1. Set environment variables (or configure via Dagster UI):

   Required:
       CF_ACCOUNT_ID   — Cloudflare account ID
       CF_TUNNEL_ID    — ID of a pre-existing Cloudflare Tunnel
       CF_API_TOKEN    — API token with Tunnel:Edit + DNS:Edit permissions

   Optional:
       CF_ZONE_ID          — Zone ID; if omitted, the dns_record step is skipped
       APP_NAME            — Container/image name        (default: streamlit-hello)
       APP_SUBDOMAIN       — Subdomain prefix            (default: hello)
       APP_DOMAIN          — Root domain                 (default: ricardo.expert)
       APP_HOST_PORT       — Preferred host port         (default: 8501)
       APP_CONTAINER_PORT  — Internal container port     (default: 8501)

2. Start the Dagster dev server:
       dagster dev -f dagster_deploy.py

3. Open http://localhost:3000, then:
       - Materialize the 'cloudflare_deploy' group for a full deploy.
       - Materialize the 'cloudflare_teardown' group for a full teardown.

Notes
-----
- The Cloudflare Tunnel daemon (cloudflared) must already be running on the host.
  This pipeline manages routing rules only; it does not start/stop the tunnel process.
- docker_run uses --rm, so the container is auto-removed when stopped.
  Do not rely on restarting a stopped container.
- find_port has no upper bound; avoid an extremely fragmented port space.
- For parallel deployments, consider appending APP_NAME to the port file path
  to prevent concurrent runs from overwriting each other's port selection.
"""

import json
import os
import subprocess
import time

import requests
from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

# =============================================================================
# Configuration
# =============================================================================
# All values are read from environment variables at module load time.
# Derived values (IMAGE_NAME, HOSTNAME) are computed once from those env vars.

APP_NAME = os.getenv("APP_NAME", "streamlit-hello")            # Docker container/image name
SUBDOMAIN = os.getenv("APP_SUBDOMAIN", "hello")                # Public subdomain prefix
DOMAIN = os.getenv("APP_DOMAIN", "ricardo.expert")             # Root domain
HOST_PORT = int(os.getenv("APP_HOST_PORT", "8501"))            # Preferred host-side port
CONTAINER_PORT = int(os.getenv("APP_CONTAINER_PORT", "8501"))  # Port the app listens on inside the container
IMAGE_NAME = f"{APP_NAME}:latest"                              # Full Docker image tag
HOSTNAME = f"{SUBDOMAIN}.{DOMAIN}"                            # Fully qualified public hostname

# Cloudflare API base URL (v4)
CF_API = "https://api.cloudflare.com/client/v4"

# Cloudflare credentials — all sourced from environment variables
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")  # Required
CF_TUNNEL_ID = os.getenv("CF_TUNNEL_ID", "")    # Required — must be a pre-existing tunnel
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")    # Required — needs Tunnel:Edit + DNS:Edit
CF_ZONE_ID = os.getenv("CF_ZONE_ID", "")        # Optional — DNS step is skipped if empty

# Dagster asset group name for deploy assets
GROUP_NAME = "cloudflare_deploy"


# =============================================================================
# Cloudflare API Helpers
# =============================================================================

def cf_headers() -> dict:
    """
    Build the HTTP headers required for all Cloudflare API requests.

    Returns a dict with:
      - Authorization: Bearer token from CF_API_TOKEN
      - Content-Type: application/json
    """
    return {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }


def cf_get_config() -> dict:
    """
    Fetch the full configuration of the Cloudflare Tunnel (CF_TUNNEL_ID).

    Sends a GET request to:
        /accounts/{CF_ACCOUNT_ID}/cfd_tunnel/{CF_TUNNEL_ID}/configurations

    Returns the parsed JSON response, which includes the 'ingress' rules
    and 'warp-routing' settings under result.config.

    Raises:
        requests.HTTPError: if the API returns a non-2xx status code.
    """
    resp = requests.get(
        f"{CF_API}/accounts/{CF_ACCOUNT_ID}/cfd_tunnel/{CF_TUNNEL_ID}/configurations",
        headers=cf_headers(),
    )
    resp.raise_for_status()
    return resp.json()


def cf_put_config(config: dict) -> dict:
    """
    Fully replace the Cloudflare Tunnel configuration with the provided dict.

    Sends a PUT request to:
        /accounts/{CF_ACCOUNT_ID}/cfd_tunnel/{CF_TUNNEL_ID}/configurations

    The 'config' argument should follow the structure:
        {
            "config": {
                "ingress": [...],         # list of ingress rules; catch-all must be last
                "warp-routing": {...}     # warp routing settings
            }
        }

    Returns the parsed JSON response from Cloudflare.

    Raises:
        requests.HTTPError: if the API returns a non-2xx status code.
    """
    resp = requests.put(
        f"{CF_API}/accounts/{CF_ACCOUNT_ID}/cfd_tunnel/{CF_TUNNEL_ID}/configurations",
        headers=cf_headers(),
        json=config,
    )
    resp.raise_for_status()
    return resp.json()


def get_used_ports() -> set[int]:
    """
    Return the set of localhost ports currently occupied in the tunnel ingress config.

    Fetches the live tunnel configuration and parses every 'service' field that
    matches the pattern 'http://localhost:{port}' or 'https://localhost:{port}'.

    Returns:
        A set of integers representing ports already routed by the tunnel.
        Returns an empty set if no localhost services are configured.
    """
    config = cf_get_config()
    ports = set()
    for entry in config.get("result", {}).get("config", {}).get("ingress", []):
        service = entry.get("service", "")
        if "localhost:" in service:
            try:
                # Extract the port number from "http://localhost:8501" → 8501
                port = int(service.split("localhost:")[-1].split("/")[0])
                ports.add(port)
            except ValueError:
                pass
    return ports


def find_available_port(preferred: int) -> int:
    """
    Find the lowest available host port, starting from 'preferred'.

    Queries the live tunnel config via get_used_ports() and increments
    the candidate port until one is found that is not already in use.

    Args:
        preferred: The port to try first (typically HOST_PORT).

    Returns:
        The first port >= preferred that is not already in the tunnel config.

    Note:
        This function has no upper bound. If many ports are occupied it will
        keep searching indefinitely. In practice this is not a concern for
        small deployments.
    """
    used = get_used_ports()
    port = preferred
    while port in used:
        port += 1
    return port


# =============================================================================
# DEPLOY ASSETS
# =============================================================================

# Root directory where app.py and Dockerfile are written before the build.
PROJECT_DIR = os.path.expanduser(f"~/main-docker/streamlit")


@asset(group_name=GROUP_NAME)
def docker_create(context: AssetExecutionContext) -> MaterializeResult:
    """
    Create the project directory and write the application source files.

    This is the root asset of the deploy group — it has no upstream dependencies.

    Creates two files inside PROJECT_DIR (~/main-docker/streamlit/):
      - app.py      : A minimal "Hello World" Streamlit application.
      - Dockerfile  : Builds a python:3.11-slim image that runs app.py on port 8501.

    Dockerfile details:
      - Base image  : python:3.11-slim
      - Installs    : build-essential (for native deps), streamlit
      - Security    : runs as non-root 'appuser'
      - Exposes     : port 8501
      - Usage stats : disabled via --browser.gatherUsageStats=false

    Metadata returned:
      - project_dir : absolute path to the created directory
      - files       : comma-separated list of created files
    """
    os.makedirs(PROJECT_DIR, exist_ok=True)

    app_py = """\
import streamlit as st

st.set_page_config(page_title="Hello Streamlit", page_icon="👋")

st.title("👋 Hello, Streamlit!")
st.write("This is a minimal Streamlit app running inside a Docker container.")
st.write("If you can read this in your browser, your container is working!")
st.write("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
"""

    dockerfile = """\
FROM python:3.11-slim
WORKDIR /
RUN useradd -m appuser
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip \\
 && pip install --no-cache-dir streamlit
COPY app.py /app.py
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
"""

    app_path = os.path.join(PROJECT_DIR, "app.py")
    dockerfile_path = os.path.join(PROJECT_DIR, "Dockerfile")

    with open(app_path, "w") as f:
        f.write(app_py)

    with open(dockerfile_path, "w") as f:
        f.write(dockerfile)

    context.log.info(f"Project created at {PROJECT_DIR}")

    return MaterializeResult(
        metadata={
            "project_dir": MetadataValue.text(PROJECT_DIR),
            "files": MetadataValue.text("app.py, Dockerfile"),
        }
    )


@asset(group_name=GROUP_NAME, deps=[docker_create])
def docker_build(context: AssetExecutionContext) -> MaterializeResult:
    """
    Build the Docker image from the project directory.

    Runs: docker build -t {IMAGE_NAME} .
    inside PROJECT_DIR, using the Dockerfile created by docker_create.

    The resulting image is tagged as '{APP_NAME}:latest' and stored
    in the local Docker daemon image cache.

    Raises:
        Exception: if docker build exits with a non-zero return code.
                   The full stderr output is included in the error message.

    Metadata returned:
      - image       : full image tag (e.g. "streamlit-hello:latest")
      - project_dir : path used as the build context
      - build_log   : last 2000 characters of stdout (truncated to avoid
                      exceeding Dagster metadata size limits)
    """
    context.log.info(f"Building image {IMAGE_NAME} from {PROJECT_DIR}")

    result = subprocess.run(
        ["docker", "build", "-t", IMAGE_NAME, "."],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        context.log.error(result.stderr)
        raise Exception(f"Docker build failed:\n{result.stderr}")

    context.log.info("Docker build succeeded.")

    return MaterializeResult(
        metadata={
            "image": MetadataValue.text(IMAGE_NAME),
            "project_dir": MetadataValue.text(PROJECT_DIR),
            # Truncate to last 2000 chars to stay within Dagster metadata limits
            "build_log": MetadataValue.text(result.stdout[-2000:] if result.stdout else ""),
        }
    )


@asset(group_name=GROUP_NAME, deps=[docker_build])
def find_port(context: AssetExecutionContext) -> MaterializeResult:
    """
    Determine an available host port by inspecting the live Cloudflare Tunnel config.

    Calls find_available_port(HOST_PORT), which queries the Cloudflare API for
    all ports currently used by tunnel ingress rules and finds the first free
    port starting from HOST_PORT.

    The selected port is written to ~/.cf_deploy_port so that downstream assets
    (docker_run and tunnel_route) can read it via _read_assigned_port().

    Logs a warning if the preferred port (HOST_PORT) was already occupied and a
    different port was auto-assigned.

    Metadata returned:
      - preferred_port : the configured APP_HOST_PORT value
      - assigned_port  : the port actually selected (may differ from preferred)
      - used_ports     : sorted list of all ports currently in use by the tunnel
    """
    available_port = find_available_port(HOST_PORT)
    used_ports = get_used_ports()

    if available_port != HOST_PORT:
        context.log.warning(
            f"Port {HOST_PORT} is in use. Auto-assigned port: {available_port}"
        )
    else:
        context.log.info(f"Port {available_port} is available.")

    # Persist the selected port to a temp file for downstream assets.
    # docker_run and tunnel_route both call _read_assigned_port() to consume it.
    port_file = os.path.expanduser("~/.cf_deploy_port")
    with open(port_file, "w") as f:
        f.write(str(available_port))

    return MaterializeResult(
        metadata={
            "preferred_port": MetadataValue.int(HOST_PORT),
            "assigned_port": MetadataValue.int(available_port),
            "used_ports": MetadataValue.text(str(sorted(used_ports))),
        }
    )


def _read_assigned_port() -> int:
    """
    Read the port selected by find_port from the temp file ~/.cf_deploy_port.

    This is a private helper used by docker_run and tunnel_route to retrieve
    the port that was chosen during the find_port asset execution.

    Returns:
        The integer port written by find_port, or HOST_PORT as a fallback
        if the file does not exist (e.g. when running assets in isolation).
    """
    port_file = os.path.expanduser("~/.cf_deploy_port")
    if os.path.exists(port_file):
        return int(open(port_file).read().strip())
    return HOST_PORT


@asset(group_name=GROUP_NAME, deps=[docker_build, find_port])
def docker_run(context: AssetExecutionContext) -> MaterializeResult:
    """
    Start the Docker container on the port assigned by find_port.

    Steps performed:
      1. Reads the assigned port from ~/.cf_deploy_port via _read_assigned_port().
      2. Force-removes any existing container with the same APP_NAME to ensure
         a clean start (idempotent; does not fail if no container exists).
      3. Runs a new detached container with the built image.
      4. Waits 2 seconds to allow the application to initialize before returning.

    Docker flags used:
      --rm   : container is automatically removed when stopped
      -d     : detached mode (runs in background)
      -p     : maps host_port → CONTAINER_PORT
      --name : assigns a fixed name for easy reference in other commands

    Raises:
        Exception: if docker run exits with a non-zero return code.
                   Full stderr is included in the error message.

    Metadata returned:
      - container_id   : first 12 characters of the container ID
      - container_name : APP_NAME
      - host_port      : the host-side port the container is bound to
      - local_url      : clickable URL for local access (http://localhost:{port})
    """
    port = _read_assigned_port()

    # Remove any previously running container with the same name.
    # capture_output=True suppresses output; non-zero exit is ignored (container may not exist).
    subprocess.run(
        ["docker", "rm", "-f", APP_NAME],
        capture_output=True,
    )

    context.log.info(f"Starting container {APP_NAME} on port {port}")

    result = subprocess.run(
        [
            "docker", "run", "--rm", "-d",
            "-p", f"{port}:{CONTAINER_PORT}",
            "--name", APP_NAME,
            IMAGE_NAME,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        context.log.error(result.stderr)
        raise Exception(f"Docker run failed:\n{result.stderr}")

    # Truncate to first 12 chars — standard Docker short ID format
    container_id = result.stdout.strip()[:12]
    context.log.info(f"Container started: {container_id}")

    # Brief pause to let the app finish its startup sequence before
    # tunnel_route tries to route traffic to it.
    time.sleep(2)

    return MaterializeResult(
        metadata={
            "container_id": MetadataValue.text(container_id),
            "container_name": MetadataValue.text(APP_NAME),
            "host_port": MetadataValue.int(port),
            "local_url": MetadataValue.url(f"http://localhost:{port}"),
        }
    )


@asset(group_name=GROUP_NAME, deps=[docker_run])
def tunnel_route(context: AssetExecutionContext) -> MaterializeResult:
    """
    Add or update the Cloudflare Tunnel ingress rule for this application.

    Fetches the current tunnel configuration from the Cloudflare API, then
    either inserts a new ingress entry or updates an existing one for HOSTNAME.

    Cloudflare Tunnel ingress rules structure:
        [
            { "hostname": "app.example.com", "service": "http://localhost:8501", "originRequest": {} },
            { "service": "http_status:404" }  ← catch-all, must always be last
        ]

    Logic:
      - If a rule for HOSTNAME already exists → update its 'service' URL in-place.
      - If no rule exists → insert a new entry at position len(ingress) - 1,
        which is directly before the mandatory catch-all rule.

    The 'originRequest' field is included as an empty dict to allow future
    extension with TLS, timeout, or HTTP/2 origin settings.

    Raises:
        Exception: if the Cloudflare API returns success: false on PUT.

    Metadata returned:
      - hostname     : the public hostname being routed
      - service      : the local service URL (e.g. "http://localhost:8501")
      - action       : "created" or "updated"
      - total_routes : number of active ingress rules (excluding catch-all);
                       only present when a new route is created
    """
    port = _read_assigned_port()
    service = f"http://localhost:{port}"

    context.log.info(f"Adding tunnel route: {HOSTNAME} → {service}")

    current = cf_get_config()
    ingress = current["result"]["config"]["ingress"]
    warp_routing = current["result"]["config"].get("warp-routing", {"enabled": False})

    # Check if a route for this hostname already exists and update it
    for entry in ingress:
        if entry.get("hostname") == HOSTNAME:
            context.log.warning(f"Route for {HOSTNAME} already exists, updating service.")
            entry["service"] = service
            new_config = {
                "config": {
                    "ingress": ingress,
                    "warp-routing": warp_routing,
                }
            }
            result = cf_put_config(new_config)
            if result.get("success"):
                return MaterializeResult(
                    metadata={
                        "hostname": MetadataValue.text(HOSTNAME),
                        "service": MetadataValue.text(service),
                        "action": MetadataValue.text("updated"),
                    }
                )
            raise Exception(f"Failed to update route: {result.get('errors')}")

    # No existing route found — insert a new entry before the catch-all rule.
    # ingress[:-1] = all rules except the catch-all
    # ingress[-1]  = the catch-all rule (must stay last)
    new_entry = {"hostname": HOSTNAME, "service": service, "originRequest": {}}
    new_ingress = ingress[:-1] + [new_entry] + [ingress[-1]]

    new_config = {
        "config": {
            "ingress": new_ingress,
            "warp-routing": warp_routing,
        }
    }

    result = cf_put_config(new_config)

    if not result.get("success"):
        raise Exception(f"Failed to add tunnel route: {result.get('errors')}")

    context.log.info(f"Tunnel route added: {HOSTNAME} → {service}")

    return MaterializeResult(
        metadata={
            "hostname": MetadataValue.text(HOSTNAME),
            "service": MetadataValue.text(service),
            "action": MetadataValue.text("created"),
            # Subtract 1 to exclude the catch-all from the count
            "total_routes": MetadataValue.int(len(new_ingress) - 1),
        }
    )


@asset(group_name=GROUP_NAME, deps=[tunnel_route])
def dns_record(context: AssetExecutionContext) -> MaterializeResult:
    """
    Create a proxied CNAME DNS record pointing HOSTNAME to the Cloudflare Tunnel.

    The record points:
        {HOSTNAME}  →  {CF_TUNNEL_ID}.cfargotunnel.com  (proxied)

    Proxied = traffic flows through Cloudflare's network, enabling DDoS protection,
    SSL termination, and hiding the origin IP.

    Behavior:
      - If CF_ZONE_ID is not set: skips execution and returns a metadata hint
        with the manual CNAME that the operator should add.
      - If a CNAME for HOSTNAME already exists: returns successfully without
        creating a duplicate (idempotent).
      - Otherwise: creates a new proxied CNAME record.

    Raises:
        Exception: if the Cloudflare DNS API returns success: false on POST.

    Metadata returned:
      - hostname      : the DNS name being created
      - target        : the tunnel target (only when created)
      - record_id     : the Cloudflare DNS record ID
      - public_url    : the live HTTPS URL (only when created)
      - status        : "created" | "already exists" | "skipped — CF_ZONE_ID not set"
      - manual_action : (only when skipped) the CNAME to add manually
    """
    if not CF_ZONE_ID:
        context.log.warning("CF_ZONE_ID not set — skipping DNS record creation.")
        return MaterializeResult(
            metadata={
                "status": MetadataValue.text("skipped — CF_ZONE_ID not set"),
                "manual_action": MetadataValue.text(
                    f"Add CNAME: {HOSTNAME} → {CF_TUNNEL_ID}.cfargotunnel.com"
                ),
            }
        )

    context.log.info(f"Creating DNS CNAME: {HOSTNAME} → {CF_TUNNEL_ID}.cfargotunnel.com")

    # Check for an existing CNAME record to avoid duplicates
    existing = requests.get(
        f"{CF_API}/zones/{CF_ZONE_ID}/dns_records",
        params={"type": "CNAME", "name": HOSTNAME},
        headers=cf_headers(),
    ).json()

    if existing.get("result") and len(existing["result"]) > 0:
        context.log.warning(f"DNS record for {HOSTNAME} already exists.")
        return MaterializeResult(
            metadata={
                "hostname": MetadataValue.text(HOSTNAME),
                "status": MetadataValue.text("already exists"),
                "record_id": MetadataValue.text(existing["result"][0]["id"]),
            }
        )

    result = requests.post(
        f"{CF_API}/zones/{CF_ZONE_ID}/dns_records",
        headers=cf_headers(),
        json={
            "type": "CNAME",
            "name": HOSTNAME,
            "content": f"{CF_TUNNEL_ID}.cfargotunnel.com",
            "proxied": True,  # Route through Cloudflare's proxy network
        },
    ).json()

    if not result.get("success"):
        raise Exception(f"Failed to create DNS record: {result.get('errors')}")

    context.log.info(f"DNS record created: {HOSTNAME}")

    return MaterializeResult(
        metadata={
            "hostname": MetadataValue.text(HOSTNAME),
            "target": MetadataValue.text(f"{CF_TUNNEL_ID}.cfargotunnel.com"),
            "record_id": MetadataValue.text(result["result"]["id"]),
            "public_url": MetadataValue.url(f"https://{HOSTNAME}"),
            "status": MetadataValue.text("created"),
        }
    )


# =============================================================================
# TEARDOWN ASSETS
# =============================================================================
# Mirror of the deploy group — removes all resources in reverse dependency order:
#   DNS record → tunnel route → container

TEARDOWN_GROUP = "cloudflare_teardown"


@asset(group_name=TEARDOWN_GROUP)
def remove_dns_record(context: AssetExecutionContext) -> MaterializeResult:
    """
    Delete the CNAME DNS record for HOSTNAME from Cloudflare DNS.

    This is the root asset of the teardown group — it has no upstream dependencies
    and is the first step executed during teardown.

    Steps:
      1. Skips if CF_ZONE_ID is not configured.
      2. Queries Cloudflare DNS for a CNAME matching HOSTNAME.
      3. If not found, returns with status "not found" (no error raised).
      4. If found, sends a DELETE request using the record's ID.

    Raises:
        Exception: if the DELETE API call returns success: false.

    Metadata returned:
      - hostname  : the DNS name that was targeted
      - record_id : the Cloudflare record ID that was deleted (if found)
      - status    : "deleted" | "not found" | "skipped"
    """
    if not CF_ZONE_ID:
        context.log.warning("CF_ZONE_ID not set — skipping.")
        return MaterializeResult(
            metadata={"status": MetadataValue.text("skipped")}
        )

    context.log.info(f"Removing DNS record: {HOSTNAME}")

    existing = requests.get(
        f"{CF_API}/zones/{CF_ZONE_ID}/dns_records",
        params={"type": "CNAME", "name": HOSTNAME},
        headers=cf_headers(),
    ).json()

    if not existing.get("result") or len(existing["result"]) == 0:
        context.log.warning(f"No DNS record found for {HOSTNAME}")
        return MaterializeResult(
            metadata={"status": MetadataValue.text("not found")}
        )

    record_id = existing["result"][0]["id"]
    result = requests.delete(
        f"{CF_API}/zones/{CF_ZONE_ID}/dns_records/{record_id}",
        headers=cf_headers(),
    ).json()

    if not result.get("success"):
        raise Exception(f"Failed to delete DNS record: {result.get('errors')}")

    context.log.info(f"DNS record removed: {HOSTNAME}")

    return MaterializeResult(
        metadata={
            "hostname": MetadataValue.text(HOSTNAME),
            "record_id": MetadataValue.text(record_id),
            "status": MetadataValue.text("deleted"),
        }
    )


@asset(group_name=TEARDOWN_GROUP, deps=[remove_dns_record])
def remove_tunnel_route(context: AssetExecutionContext) -> MaterializeResult:
    """
    Remove the ingress rule for HOSTNAME from the Cloudflare Tunnel configuration.

    Fetches the current tunnel ingress config and filters out any entry whose
    'hostname' field matches HOSTNAME. Uploads the modified config back via PUT.

    The catch-all rule (last entry, no 'hostname' key) is always preserved.

    If no matching route is found, logs a warning and returns gracefully
    without raising an error — making this step safe to run even if the
    route was never created or was already removed manually.

    Raises:
        Exception: if the PUT request to Cloudflare returns success: false.

    Metadata returned:
      - hostname         : the hostname whose route was targeted
      - status           : "deleted" | "not found"
      - remaining_routes : count of active ingress rules after removal
                           (excluding catch-all); only present when deleted
    """
    context.log.info(f"Removing tunnel route: {HOSTNAME}")

    current = cf_get_config()
    ingress = current["result"]["config"]["ingress"]
    warp_routing = current["result"]["config"].get("warp-routing", {"enabled": False})

    # Filter out the route for this hostname; keep everything else including catch-all
    original_count = len(ingress)
    new_ingress = [e for e in ingress if e.get("hostname") != HOSTNAME]

    if len(new_ingress) == original_count:
        # No entry was removed — the route didn't exist
        context.log.warning(f"No route found for {HOSTNAME}")
        return MaterializeResult(
            metadata={"status": MetadataValue.text("not found")}
        )

    new_config = {
        "config": {
            "ingress": new_ingress,
            "warp-routing": warp_routing,
        }
    }

    result = cf_put_config(new_config)

    if not result.get("success"):
        raise Exception(f"Failed to remove tunnel route: {result.get('errors')}")

    context.log.info(f"Tunnel route removed: {HOSTNAME}")

    return MaterializeResult(
        metadata={
            "hostname": MetadataValue.text(HOSTNAME),
            "status": MetadataValue.text("deleted"),
            # Subtract 1 to exclude the catch-all from the count
            "remaining_routes": MetadataValue.int(len(new_ingress) - 1),
        }
    )


@asset(group_name=TEARDOWN_GROUP, deps=[remove_tunnel_route])
def docker_stop(context: AssetExecutionContext) -> MaterializeResult:
    """
    Stop and remove the running Docker container, then clean up the port file.

    This is the final step of the teardown group.

    Steps:
      1. Runs docker stop {APP_NAME} to send SIGTERM and wait for graceful exit.
      2. Runs docker rm -f {APP_NAME} to force-remove the container record.
         (Note: since the container was started with --rm, it may already be
         gone after stop; the rm -f is a safety net.)
      3. Deletes ~/.cf_deploy_port if it exists, releasing the port reservation.

    Does not raise an error if the container was not running — logs a warning
    instead, allowing the teardown to complete cleanly even after a partial deploy.

    Metadata returned:
      - container   : the container name that was targeted (APP_NAME)
      - was_running : True if docker stop succeeded; False if container was absent
      - status      : "stopped" | "not found"
    """
    context.log.info(f"Stopping container: {APP_NAME}")

    stop_result = subprocess.run(
        ["docker", "stop", APP_NAME],
        capture_output=True,
        text=True,
    )

    # Force-remove the container record regardless of whether stop succeeded.
    # This handles edge cases where the container exists but is in a bad state.
    rm_result = subprocess.run(
        ["docker", "rm", "-f", APP_NAME],
        capture_output=True,
        text=True,
    )

    stopped = stop_result.returncode == 0

    if stopped:
        context.log.info(f"Container {APP_NAME} stopped.")
    else:
        context.log.warning(f"Container {APP_NAME} was not running.")

    # Remove the port file created by find_port to release the port reservation
    port_file = os.path.expanduser("~/.cf_deploy_port")
    if os.path.exists(port_file):
        os.remove(port_file)

    return MaterializeResult(
        metadata={
            "container": MetadataValue.text(APP_NAME),
            "was_running": MetadataValue.bool(stopped),
            "status": MetadataValue.text("stopped" if stopped else "not found"),
        }
    )