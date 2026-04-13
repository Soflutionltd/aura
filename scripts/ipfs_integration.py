#!/usr/bin/env python3
"""
AURA IPFS Integration: publish and retrieve LoRA deltas on IPFS.
Each AURA node runs an IPFS daemon alongside Ollama.
Deltas are published to IPFS, pinned by the network, and referenced on-chain.
"""

import json
import subprocess
import hashlib
import urllib.request
from pathlib import Path

IPFS_API = "http://localhost:5001/api/v0"


def is_ipfs_running() -> bool:
    """Check if the local IPFS daemon is running."""
    try:
        req = urllib.request.Request(f"{IPFS_API}/id", method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def start_ipfs():
    """Start IPFS daemon in background if not running."""
    if is_ipfs_running():
        return True
    try:
        subprocess.Popen(
            ["ipfs", "daemon", "--enable-gc"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import time
        for _ in range(10):
            time.sleep(1)
            if is_ipfs_running():
                return True
        return False
    except FileNotFoundError:
        print("IPFS not installed. Install: https://docs.ipfs.tech/install/")
        return False


def publish_delta(file_path: str) -> dict | None:
    """
    Publish a LoRA delta file to IPFS.
    Returns {"cid": "Qm...", "size": 12345, "cid_hash": bytes} or None.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return None

    # Add file to IPFS
    try:
        result = subprocess.run(
            ["ipfs", "add", "--quieter", str(path)],
            capture_output=True, text=True, timeout=120,
        )
        cid = result.stdout.strip()
        if not cid:
            print("IPFS add failed")
            return None

        # Pin it locally to ensure persistence
        subprocess.run(
            ["ipfs", "pin", "add", cid],
            capture_output=True, timeout=30,
        )

        # Generate a 32-byte hash for on-chain reference
        cid_hash = hashlib.sha256(cid.encode()).digest()

        return {
            "cid": cid,
            "size": path.stat().st_size,
            "cid_hash": cid_hash,
            "cid_hash_hex": cid_hash.hex(),
        }
    except Exception as e:
        print(f"IPFS publish failed: {e}")
        return None


def retrieve_delta(cid: str, output_path: str) -> bool:
    """Download a LoRA delta from IPFS by its CID."""
    try:
        result = subprocess.run(
            ["ipfs", "get", cid, "-o", output_path],
            capture_output=True, text=True, timeout=300,
        )
        if Path(output_path).exists():
            # Auto-pin after download (contribute to network)
            subprocess.run(
                ["ipfs", "pin", "add", cid],
                capture_output=True, timeout=30,
            )
            return True
        return False
    except Exception as e:
        print(f"IPFS retrieve failed: {e}")
        return False


def list_pinned() -> list[str]:
    """List all pinned CIDs on this node."""
    try:
        result = subprocess.run(
            ["ipfs", "pin", "ls", "--type=recursive", "-q"],
            capture_output=True, text=True, timeout=15,
        )
        return [cid.strip() for cid in result.stdout.strip().split("\n") if cid.strip()]
    except Exception:
        return []


def get_node_id() -> str | None:
    """Get this IPFS node's peer ID."""
    try:
        req = urllib.request.Request(f"{IPFS_API}/id", method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
            return data.get("ID")
    except Exception:
        return None


# ── CLI for testing ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if not is_ipfs_running():
        print("Starting IPFS daemon...")
        if not start_ipfs():
            print("Failed to start IPFS.")
            sys.exit(1)

    node_id = get_node_id()
    print(f"IPFS Node: {node_id}")
    print(f"Pinned files: {len(list_pinned())}")

    if len(sys.argv) > 1:
        action = sys.argv[1]
        if action == "publish" and len(sys.argv) > 2:
            result = publish_delta(sys.argv[2])
            if result:
                print(f"Published to IPFS:")
                print(f"  CID: {result['cid']}")
                print(f"  Size: {result['size']} bytes")
                print(f"  Hash: {result['cid_hash_hex']}")
        elif action == "get" and len(sys.argv) > 3:
            ok = retrieve_delta(sys.argv[2], sys.argv[3])
            print(f"Download: {'OK' if ok else 'FAILED'}")
        else:
            print("Usage:")
            print("  python3 ipfs_integration.py publish <file>")
            print("  python3 ipfs_integration.py get <cid> <output_path>")
