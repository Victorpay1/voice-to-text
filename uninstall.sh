#!/bin/bash

# Voice to Text - Uninstaller
# Removes all Voice to Text files and preferences

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

INSTALL_DIR="$HOME/Applications/VoiceToText"
DESKTOP_LAUNCHER="$HOME/Desktop/Voice to Text.command"
CONFIG_FILE="$HOME/.voice_to_text_config.json"
PID_FILE="$HOME/.voice_to_text.pid"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║         Voice to Text - Uninstaller              ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "This will remove Voice to Text from your Mac."
echo ""
echo "The following will be deleted:"
echo "  • Application files (~500MB)"
echo "  • Desktop launcher"
echo "  • User preferences"
echo ""
echo -e "${YELLOW}Note: Python 3.12 and Homebrew will NOT be removed.${NC}"
echo ""

read -p "Are you sure you want to uninstall? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstall cancelled."
    exit 0
fi

echo ""
echo "Removing Voice to Text..."

# Kill any running instances
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ ! -z "$PID" ] && ps -p $PID > /dev/null 2>&1; then
        echo "Stopping running app..."
        kill $PID 2>/dev/null || true
        sleep 1
    fi
fi

# Remove installation directory
if [ -d "$INSTALL_DIR" ]; then
    echo "  Removing application files..."
    rm -rf "$INSTALL_DIR"
    echo -e "${GREEN}  ✓ Application files removed${NC}"
fi

# Remove Desktop launcher
if [ -f "$DESKTOP_LAUNCHER" ]; then
    echo "  Removing Desktop launcher..."
    rm -f "$DESKTOP_LAUNCHER"
    echo -e "${GREEN}  ✓ Desktop launcher removed${NC}"
fi

# Remove config files
if [ -f "$CONFIG_FILE" ]; then
    echo "  Removing preferences..."
    rm -f "$CONFIG_FILE"
    echo -e "${GREEN}  ✓ Preferences removed${NC}"
fi

# Remove PID file
if [ -f "$PID_FILE" ]; then
    rm -f "$PID_FILE"
fi

echo ""
echo -e "${GREEN}✓ Voice to Text has been completely removed.${NC}"
echo ""
echo "Thank you for using Voice to Text!"
echo ""
