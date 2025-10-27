#!/bin/bash

# Voice to Text - One-Command Installer for Mac
# This script installs everything needed to run the Voice to Text app

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Installation directory
INSTALL_DIR="$HOME/Applications/VoiceToText"
DESKTOP_LAUNCHER="$HOME/Desktop/Voice to Text.command"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Voice to Text - Enhanced Installation         ║"
echo "║   AI-Powered Voice Transcription for Mac        ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "This will install Voice to Text on your Mac."
echo "Installation takes ~5 minutes and requires ~500MB disk space."
echo ""

# Ask for confirmation
read -p "Continue with installation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo -e "${BLUE}[1/7]${NC} Checking system requirements..."

# Check macOS version
if [[ $(uname) != "Darwin" ]]; then
    echo -e "${RED}Error: This installer is for macOS only.${NC}"
    exit 1
fi

# Check for Homebrew
echo -e "${BLUE}[2/7]${NC} Checking for Homebrew..."
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing Homebrew...${NC}"
    echo "This may take a few minutes and will ask for your password."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH for Apple Silicon Macs
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    echo -e "${GREEN}✓ Homebrew installed${NC}"
else
    echo -e "${GREEN}✓ Homebrew found${NC}"
fi

# Install Python 3.12
echo -e "${BLUE}[3/7]${NC} Installing Python 3.12..."
if ! brew list python@3.12 &> /dev/null; then
    brew install python@3.12
    echo -e "${GREEN}✓ Python 3.12 installed${NC}"
else
    echo -e "${GREEN}✓ Python 3.12 already installed${NC}"
fi

# Create installation directory
echo -e "${BLUE}[4/7]${NC} Setting up application directory..."
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download application files from GitHub
echo -e "${BLUE}[5/7]${NC} Downloading application files..."

# Download main Python file
curl -fsSL "https://raw.githubusercontent.com/Victorpay1/voice-to-text/main/voice_to_text_menubar_enhanced.py" -o voice_to_text_menubar_enhanced.py

# If GitHub isn't set up yet, copy from local (for testing)
if [ ! -f "voice_to_text_menubar_enhanced.py" ]; then
    echo -e "${YELLOW}Note: Using local files for now (GitHub not configured)${NC}"
    # This section is temporary - will be replaced with GitHub download
    if [ -f "/Users/victorpaytuvi/Desktop/CLAUDE-PROJECTS/voice to text/voice_to_text_menubar_enhanced.py" ]; then
        cp "/Users/victorpaytuvi/Desktop/CLAUDE-PROJECTS/voice to text/voice_to_text_menubar_enhanced.py" .
    else
        echo -e "${RED}Error: Could not download or find application files.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Application files downloaded${NC}"

# Create virtual environment and install dependencies
echo -e "${BLUE}[6/7]${NC} Installing dependencies (this takes ~3-4 minutes)..."
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv venv

# Activate virtual environment and install packages
source venv/bin/activate

# Install packages quietly
pip install --quiet --upgrade pip setuptools wheel

echo "  Installing core packages..."
pip install --quiet rumps faster-whisper sounddevice numpy pynput torch psutil scipy language-tool-python certifi

echo "  Installing voice detection..."
pip install --quiet silero-vad torchaudio

echo "  Installing translation support..."
pip install --quiet argostranslate

deactivate

# Update the shebang in the Python file
sed -i '' '1s|.*|#!/usr/bin/env python3|' voice_to_text_menubar_enhanced.py
chmod +x voice_to_text_menubar_enhanced.py

echo -e "${GREEN}✓ All dependencies installed${NC}"

# Create Desktop launcher
echo -e "${BLUE}[7/7]${NC} Creating Desktop launcher..."
cat > "$DESKTOP_LAUNCHER" << 'EOF'
#!/bin/bash
cd "$HOME/Applications/VoiceToText"
./venv/bin/python3 voice_to_text_menubar_enhanced.py
EOF

chmod +x "$DESKTOP_LAUNCHER"
echo -e "${GREEN}✓ Desktop launcher created${NC}"

# Test installation
echo ""
echo -e "${BLUE}Testing installation...${NC}"
if "$INSTALL_DIR/venv/bin/python3" -c "import rumps, faster_whisper, sounddevice" 2>/dev/null; then
    echo -e "${GREEN}✓ Installation test passed${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Installation test failed, but app may still work${NC}"
fi

# Success message
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║            🎉 INSTALLATION COMPLETE! 🎉          ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Voice to Text is ready to use!${NC}"
echo ""
echo "HOW TO USE:"
echo "  1. Double-click 'Voice to Text.command' on your Desktop"
echo "  2. A microphone icon will appear in your menu bar"
echo "  3. Press Control + Space to start recording"
echo "  4. Speak your message"
echo "  5. Press Control + Space again to stop"
echo "  6. Your text appears where your cursor is!"
echo ""
echo "FEATURES:"
echo "  ✓ Ultra-fast AI transcription"
echo "  ✓ Smart voice detection"
echo "  ✓ English ↔ Spanish translation"
echo "  ✓ Multiple accuracy modes"
echo "  ✓ Works offline after first use"
echo ""
echo -e "${YELLOW}Note: First launch may take 1-2 minutes to download AI models.${NC}"
echo ""
echo "Try it now by double-clicking the Desktop icon!"
echo ""
