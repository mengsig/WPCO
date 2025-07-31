#!/bin/bash
# Launcher script for asynchronous enhanced training

# Default values
EPISODES=50000
WORKERS=$(nproc --ignore=1)  # Number of CPUs minus 1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--episodes N] [--workers N]"
            echo "  --episodes N: Number of training episodes (default: 50000)"
            echo "  --workers N:  Number of worker processes (default: CPU count - 1)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting asynchronous enhanced training..."
echo "Episodes: $EPISODES"
echo "Workers: $WORKERS"
echo ""

# Run the training
python3 src/scripts/enhanced_train_async.py --episodes $EPISODES --workers $WORKERS