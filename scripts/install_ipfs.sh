#!/bin/bash
set -e
echo "=== AURA Node Setup ==="
echo "Installing IPFS for decentralized storage..."
echo ""

# Detect OS
OS=$(uname -s)
ARCH=$(uname -m)

# Check if IPFS already installed
if command -v ipfs &> /dev/null; then
    echo "IPFS already installed: $(ipfs --version)"
else
    echo "Downloading IPFS..."
    if [ "$OS" = "Darwin" ]; then
        # macOS
        if [ "$ARCH" = "arm64" ]; then
            curl -fsSL -o /tmp/ipfs.tar.gz https://dist.ipfs.tech/kubo/v0.33.2/kubo_v0.33.2_darwin-arm64.tar.gz
        else
            curl -fsSL -o /tmp/ipfs.tar.gz https://dist.ipfs.tech/kubo/v0.33.2/kubo_v0.33.2_darwin-amd64.tar.gz
        fi
    elif [ "$OS" = "Linux" ]; then
        if [ "$ARCH" = "aarch64" ]; then
            curl -fsSL -o /tmp/ipfs.tar.gz https://dist.ipfs.tech/kubo/v0.33.2/kubo_v0.33.2_linux-arm64.tar.gz
        else
            curl -fsSL -o /tmp/ipfs.tar.gz https://dist.ipfs.tech/kubo/v0.33.2/kubo_v0.33.2_linux-amd64.tar.gz
        fi
    fi
    tar -xzf /tmp/ipfs.tar.gz -C /tmp
    sudo mv /tmp/kubo/ipfs /usr/local/bin/
    rm -rf /tmp/ipfs.tar.gz /tmp/kubo
    echo "IPFS installed: $(ipfs --version)"
fi

# Initialize IPFS if not already done
if [ ! -d "$HOME/.ipfs" ]; then
    echo "Initializing IPFS node..."
    ipfs init --profile=lowpower
    # Configure for AURA network
    # Reduce resource usage for background operation
    ipfs config Datastore.StorageMax "10GB"
    ipfs config --json Swarm.ConnMgr.LowWater 50
    ipfs config --json Swarm.ConnMgr.HighWater 200
    ipfs config --json Reprovider.Interval '"12h"'
    echo "IPFS node initialized."
else
    echo "IPFS node already initialized."
fi

# Start IPFS daemon in background
echo "Starting IPFS daemon..."
ipfs daemon --enable-gc &
sleep 3

# Verify
PEER_ID=$(ipfs id -f='<id>')
echo ""
echo "=== AURA IPFS Node Ready ==="
echo "Peer ID: $PEER_ID"
echo "API: http://localhost:5001"
echo "Gateway: http://localhost:8080"
echo ""
echo "Your node will automatically store and share"
echo "AURA LoRA deltas with the network."
echo "The more nodes, the stronger the network."
