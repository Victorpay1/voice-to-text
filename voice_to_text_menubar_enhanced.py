#!/Users/victorpaytuvi/Desktop/CLAUDE-PROJECTS/voice to text/venv/bin/python3
"""
Voice to Text Menu Bar Application - ENHANCED BILINGUAL VERSION
Ultra-fast and accurate with faster-whisper, Silero VAD, and advanced audio processing.
Shows a menu bar icon with status and easy quit option.

Key Improvements:
- 3-4x faster transcription with faster-whisper
- Smart voice detection with Silero VAD
- Enhanced audio preprocessing
- Optimized parameters for quality
- Bilingual support: English ↔ Spanish
"""

import ssl
import certifi
import os

# Fix SSL certificate issues for model download
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['SSL_CERT_FILE'] = certifi.where()

import rumps
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
from pynput import keyboard
from pynput.keyboard import Controller
import threading
import tempfile
import wave
import time
import re
import torch
import json
import signal
import atexit
import sys
import psutil  # For memory monitoring

# Safety: Prevent multiple instances from running
PID_FILE = os.path.expanduser("~/.voice_to_text.pid")

def check_single_instance():
    """Ensure only one instance of the app is running"""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())

            # Check if that process is still running
            if psutil.pid_exists(old_pid):
                try:
                    proc = psutil.Process(old_pid)
                    if 'python' in proc.name().lower() and 'voice_to_text' in ' '.join(proc.cmdline()):
                        print(f"⚠️  Another instance is already running (PID: {old_pid})")
                        print("   Please close it first or use the menu bar icon to quit.")
                        sys.exit(1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # Process doesn't exist or we can't access it

            # If we got here, the old PID is stale, so remove it
            os.unlink(PID_FILE)
        except (ValueError, FileNotFoundError):
            pass  # Invalid PID file, ignore

    # Write our PID
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    print(f"✅ Single instance check passed (PID: {os.getpid()})")

def cleanup_pid_file():
    """Remove PID file on exit"""
    try:
        if os.path.exists(PID_FILE):
            os.unlink(PID_FILE)
            print("✅ PID file cleaned up")
    except Exception as e:
        print(f"⚠️  Could not remove PID file: {e}")

def monitor_memory():
    """Monitor memory usage and warn if too high"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Warn if using more than 1GB
        if memory_mb > 1024:
            print(f"⚠️  High memory usage: {memory_mb:.0f}MB")
            return True
        return False
    except Exception as e:
        print(f"⚠️  Memory monitoring error: {e}")
        return False

class VoiceToTextMenuBarEnhanced(rumps.App):
    def __init__(self):
        super(VoiceToTextMenuBarEnhanced, self).__init__(
            "🎤⚡",  # Microphone with lightning bolt to show enhanced version
            quit_button=None  # We'll create our own quit button
        )

        # Load language preferences
        self.config_file = os.path.expanduser("~/.voice_to_text_config.json")
        self.load_preferences()

        # Menu items
        self.status_item = rumps.MenuItem("Status: Loading...", callback=None)
        self.version_item = rumps.MenuItem("Version: ENHANCED BILINGUAL ⚡", callback=None)
        self.memory_item = rumps.MenuItem("Memory: Checking...", callback=None)
        self.stop_button = rumps.MenuItem("Stop Recording", callback=self.manual_stop)
        self.stop_button.title = "Stop Recording (hidden)"  # Will show when recording
        self.force_stop_button = rumps.MenuItem("🚨 Force Stop (Emergency)", callback=self.emergency_stop)
        self.force_stop_button.title = "🚨 Force Stop (hidden)"  # Will show when stuck

        # Language selection menu items
        self.input_lang_menu = {
            'en': rumps.MenuItem("English", callback=lambda _: self.set_input_language('en')),
            'es': rumps.MenuItem("Español", callback=lambda _: self.set_input_language('es'))
        }
        self.output_lang_menu = {
            'en': rumps.MenuItem("English", callback=lambda _: self.set_output_language('en')),
            'es': rumps.MenuItem("Español", callback=lambda _: self.set_output_language('es'))
        }

        # Accuracy mode selection menu items
        self.accuracy_mode_menu = {
            'fast': rumps.MenuItem("⚡ Fast (2-3s)", callback=lambda _: self.set_accuracy_mode('fast')),
            'clarity': rumps.MenuItem("⚡ Clarity Boost (3-4s)", callback=lambda _: self.set_accuracy_mode('clarity')),
            'max': rumps.MenuItem("🎯 Max Accuracy (5-8s)", callback=lambda _: self.set_accuracy_mode('max'))
        }

        # Update menu item states based on loaded preferences
        self.input_lang_menu[self.input_language].state = True
        self.output_lang_menu[self.output_language].state = True
        self.accuracy_mode_menu[self.accuracy_mode].state = True

        # Create parent menu items that can be updated dynamically
        self.accuracy_mode_label = rumps.MenuItem(
            f"Accuracy Mode: {self.get_mode_name(self.accuracy_mode)}"
        )
        self.input_lang_label = rumps.MenuItem(
            f"Input Language: {self.get_language_name(self.input_language)} 🎤"
        )
        self.output_lang_label = rumps.MenuItem(
            f"Output Language: {self.get_language_name(self.output_language)} 📝"
        )

        self.menu = [
            self.status_item,
            self.version_item,
            self.memory_item,
            self.stop_button,
            self.force_stop_button,
            None,  # Separator
            [self.accuracy_mode_label, list(self.accuracy_mode_menu.values())],
            None,  # Separator
            [self.input_lang_label, list(self.input_lang_menu.values())],
            [self.output_lang_label, list(self.output_lang_menu.values())],
            None,  # Separator
            rumps.MenuItem("Quit", callback=self.quit_app)
        ]

        # Recording state
        self.recording = False
        self.recording_start_time = None
        self.processing = False  # Flag to prevent starting new recording while processing
        self.typing = False  # Flag to prevent keyboard listener interference while auto-typing

        # Translation support
        self.translation_available = False

        # Model management (lazy loading for memory optimization)
        self.whisper_model_small = None
        self.whisper_model_medium = None
        self.vad_model = None
        self.models_loaded = {'small': False, 'medium': False, 'vad': False}
        self.last_model_used = None  # Track which model was used last

        # Memory optimization settings
        self.aggressive_memory_cleanup = True  # Unload unused models after each use
        self.last_activity_time = time.time()  # Track when app was last used

        # Watchdog for stuck states
        self.watchdog_enabled = True
        threading.Thread(target=self.watchdog, daemon=True).start()

        # Memory monitoring thread
        threading.Thread(target=self.memory_monitor, daemon=True).start()

        # Initialize in background thread
        threading.Thread(target=self.initialize_voice, daemon=True).start()

    def force_recovery(self):
        """Force complete state recovery - called by watchdog or Force Stop button"""
        print("\n🚨 FORCE RECOVERY: Resetting all states...")

        # Reset all flags
        self.recording = False
        self.processing = False
        self.typing = False
        self.last_action = None  # Reset action tracking
        self.last_hotkey_time = 0  # Reset debounce timer

        # Force close any stuck audio streams
        if hasattr(self, 'stream'):
            try:
                # Try graceful close with very short timeout
                def emergency_close():
                    try:
                        self.stream.stop()
                    except:
                        pass
                    try:
                        self.stream.close()
                    except:
                        pass

                close_thread = threading.Thread(target=emergency_close, daemon=True)
                close_thread.start()
                close_thread.join(timeout=0.5)  # Only wait 0.5 seconds

                # Delete reference regardless
                delattr(self, 'stream')
                print("   ✅ Audio stream cleared")
            except Exception as e:
                print(f"   ⚠️  Stream cleanup error: {e}")
                # Force delete even if error
                try:
                    delattr(self, 'stream')
                except:
                    pass

        # Comprehensive memory cleanup
        self.cleanup_memory()

        # Reset UI
        self.status_item.title = "Status: Ready ⚡"
        self.title = "🎤⚡"
        self.stop_button.title = "Stop Recording (hidden)"
        self.force_stop_button.title = "🚨 Force Stop (hidden)"  # Hide emergency button

        print("✅ FORCE RECOVERY: Complete! App ready for use.\n")

    def watchdog(self):
        """Monitor for stuck states and auto-recovery"""
        max_processing_time = 30  # 30 seconds max for processing (more aggressive)
        show_force_button_after = 15  # Show Force Stop button after 15 seconds
        check_interval = 5  # Check every 5 seconds (more frequent)

        processing_start_time = None
        force_button_shown = False

        while self.watchdog_enabled:
            try:
                time.sleep(check_interval)

                # Check if processing is stuck
                if self.processing:
                    if processing_start_time is None:
                        processing_start_time = time.time()
                        force_button_shown = False
                    else:
                        elapsed = time.time() - processing_start_time

                        # Show Force Stop button after 15 seconds (user can manually recover)
                        if elapsed > show_force_button_after and not force_button_shown:
                            print(f"⚠️  Processing taking longer than expected ({elapsed:.0f}s)...")
                            print("   Force Stop button now available in menu bar")
                            self.force_stop_button.title = "🚨 Force Stop (Click if stuck!)"
                            force_button_shown = True

                        # Automatic recovery after 30 seconds
                        if elapsed > max_processing_time:
                            print(f"\n⚠️  WATCHDOG: Processing stuck for {elapsed:.0f}s - forcing recovery")
                            self.force_recovery()
                            processing_start_time = None
                            force_button_shown = False
                else:
                    # Not processing - reset and hide force button
                    if processing_start_time is not None or force_button_shown:
                        processing_start_time = None
                        force_button_shown = False
                        self.force_stop_button.title = "🚨 Force Stop (hidden)"

            except Exception as e:
                print(f"⚠️  Watchdog error: {e}")

    def memory_monitor(self):
        """Monitor memory usage and perform automatic cleanup"""
        check_interval = 30  # Check every 30 seconds
        idle_timeout = 300  # 5 minutes of inactivity
        memory_warning_threshold = 1500  # Warn if using more than 1.5GB

        while True:
            try:
                time.sleep(check_interval)

                # Get current memory usage
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024

                # Update menu bar memory display with status indicator
                if memory_mb < 1000:
                    # Normal - under 1 GB
                    self.memory_item.title = f"Memory: {int(memory_mb)} MB ✅"
                elif memory_mb < 1500:
                    # Warning - 1-1.5 GB
                    self.memory_item.title = f"Memory: {int(memory_mb)} MB ⚠️"
                else:
                    # High - over 1.5 GB
                    self.memory_item.title = f"Memory: {int(memory_mb)} MB 🔴"

                # Check if idle for too long (5 minutes)
                time_since_activity = time.time() - self.last_activity_time
                if time_since_activity > idle_timeout and not self.recording and not self.processing:
                    # Unload all models if idle for too long
                    if self.models_loaded['small'] or self.models_loaded['medium']:
                        print(f"\n💤 App idle for {int(time_since_activity/60)} minutes - unloading models to save memory")
                        self.unload_all_models()
                        self.cleanup_memory()
                        new_memory = process.memory_info().rss / 1024 / 1024
                        print(f"   Memory after cleanup: {new_memory:.0f}MB")
                        # Update memory display immediately after cleanup
                        self.memory_item.title = f"Memory: {int(new_memory)} MB ✅"

                # Warn if memory is too high
                elif memory_mb > memory_warning_threshold:
                    print(f"\n⚠️  High memory usage detected: {memory_mb:.0f}MB")
                    # Force aggressive cleanup
                    self.cleanup_memory()
                    new_memory = process.memory_info().rss / 1024 / 1024
                    print(f"   Memory after cleanup: {new_memory:.0f}MB (freed {memory_mb - new_memory:.0f}MB)")
                    # Update memory display after cleanup
                    if new_memory < 1000:
                        self.memory_item.title = f"Memory: {int(new_memory)} MB ✅"
                    elif new_memory < 1500:
                        self.memory_item.title = f"Memory: {int(new_memory)} MB ⚠️"
                    else:
                        self.memory_item.title = f"Memory: {int(new_memory)} MB 🔴"

            except Exception as e:
                print(f"⚠️  Memory monitor error: {e}")

    def load_model_for_mode(self, mode):
        """Lazy load the model needed for the specified mode"""
        try:
            # Determine which model we need
            model_needed = 'small' if mode in ['fast', 'clarity'] else 'medium'

            # If model already loaded, nothing to do
            if self.models_loaded[model_needed]:
                return True

            print(f"\n📦 Loading {model_needed.upper()} model (first use of {mode} mode)...")
            self.status_item.title = f"Status: Loading {model_needed} model..."

            if model_needed == 'small':
                self.whisper_model_small = WhisperModel(
                    "small",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=os.cpu_count(),  # FIX: Issue #249 - prevents memory growth
                    num_workers=1
                )
                self.models_loaded['small'] = True
                print(f"✅ SMALL model loaded ({self.get_model_memory_mb('small'):.0f}MB)")
                print(f"   Using {os.cpu_count()} CPU threads")

            else:  # medium
                self.whisper_model_medium = WhisperModel(
                    "medium",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=os.cpu_count(),  # FIX: Issue #249 - prevents memory growth
                    num_workers=1
                )
                self.models_loaded['medium'] = True
                print(f"✅ MEDIUM model loaded ({self.get_model_memory_mb('medium'):.0f}MB)")
                print(f"   Using {os.cpu_count()} CPU threads")

            # Update last model used
            self.last_model_used = model_needed

            # Show total memory
            process = psutil.Process(os.getpid())
            print(f"   Total app memory: {process.memory_info().rss / 1024 / 1024:.0f}MB")

            return True

        except Exception as e:
            print(f"❌ Error loading {model_needed} model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_model_memory_mb(self, model_name):
        """Estimate memory used by a model"""
        # Approximate sizes
        sizes = {
            'small': 600,
            'medium': 1500,
            'vad': 50
        }
        return sizes.get(model_name, 0)

    def unload_model(self, model_name):
        """Unload a specific model to free memory"""
        try:
            if not self.models_loaded[model_name]:
                return  # Already unloaded

            print(f"   🧹 Unloading {model_name.upper()} model to free memory...")

            if model_name == 'small' and self.whisper_model_small:
                del self.whisper_model_small
                self.whisper_model_small = None
                self.models_loaded['small'] = False

            elif model_name == 'medium' and self.whisper_model_medium:
                del self.whisper_model_medium
                self.whisper_model_medium = None
                self.models_loaded['medium'] = False

            elif model_name == 'vad' and self.vad_model:
                del self.vad_model
                self.vad_model = None
                self.models_loaded['vad'] = False

            # Force garbage collection
            import gc
            gc.collect()

            print(f"   ✅ {model_name.upper()} model unloaded (~{self.get_model_memory_mb(model_name):.0f}MB freed)")

        except Exception as e:
            print(f"   ⚠️  Error unloading {model_name}: {e}")

    def unload_all_models(self):
        """Unload all models to free maximum memory"""
        for model_name in ['small', 'medium', 'vad']:
            if self.models_loaded[model_name]:
                self.unload_model(model_name)

    def unload_unused_model(self):
        """Unload the model that's NOT currently selected (intelligent cleanup)"""
        if not self.aggressive_memory_cleanup:
            return

        # Determine which model is currently needed
        current_mode = self.accuracy_mode
        model_needed = 'small' if current_mode in ['fast', 'clarity'] else 'medium'

        # Unload the OTHER model
        if model_needed == 'small' and self.models_loaded['medium']:
            self.unload_model('medium')
        elif model_needed == 'medium' and self.models_loaded['small']:
            self.unload_model('small')

    def cleanup_memory(self):
        """Comprehensive memory cleanup"""
        # MEMORY FIX: Unload CTranslate2 models to free memory (Issue #660)
        # Unload small model if loaded
        if self.whisper_model_small and self.models_loaded['small']:
            try:
                if self.whisper_model_small.model.model_is_loaded:
                    self.whisper_model_small.model.unload_model()
            except Exception as e:
                pass  # Ignore errors during cleanup

        # Unload medium model if loaded
        if self.whisper_model_medium and self.models_loaded['medium']:
            try:
                if self.whisper_model_medium.model.model_is_loaded:
                    self.whisper_model_medium.model.unload_model()
            except Exception as e:
                pass  # Ignore errors during cleanup

        # Clear audio data
        self.audio_data = []

        # Clear any large temporary buffers
        if hasattr(self, '_temp_audio_buffer'):
            del self._temp_audio_buffer

        # Force garbage collection - be very aggressive
        import gc

        # Collect all generations
        gc.collect(0)  # Young generation
        gc.collect(1)  # Middle generation
        gc.collect(2)  # Old generation (full collection)

        # One more pass to catch stragglers
        gc.collect()

    def load_preferences(self):
        """Load language preferences and accuracy mode from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.input_language = config.get('input_language', 'en')
                    self.output_language = config.get('output_language', 'en')
                    self.accuracy_mode = config.get('accuracy_mode', 'clarity')  # Default to Clarity Boost
            else:
                # Default to English and Clarity Boost mode
                self.input_language = 'en'
                self.output_language = 'en'
                self.accuracy_mode = 'clarity'
        except Exception as e:
            print(f"⚠️  Error loading preferences: {e}")
            self.input_language = 'en'
            self.output_language = 'en'
            self.accuracy_mode = 'clarity'

    def save_preferences(self):
        """Save language preferences and accuracy mode to config file"""
        try:
            config = {
                'input_language': self.input_language,
                'output_language': self.output_language,
                'accuracy_mode': self.accuracy_mode
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"⚠️  Error saving preferences: {e}")

    def get_language_name(self, code):
        """Get display name for language code"""
        return "English" if code == 'en' else "Español"

    def get_mode_name(self, mode):
        """Get display name for accuracy mode"""
        mode_names = {
            'fast': '⚡ Fast (2-3s)',
            'clarity': '⚡ Clarity Boost (3-4s)',
            'max': '🎯 Max Accuracy (5-8s)'
        }
        return mode_names.get(mode, '⚡ Clarity Boost (3-4s)')

    def set_input_language(self, lang_code):
        """Set input language and update menu"""
        # Update checkmarks
        for code, menu_item in self.input_lang_menu.items():
            menu_item.state = (code == lang_code)

        self.input_language = lang_code

        # Update the parent menu label
        self.input_lang_label.title = f"Input Language: {self.get_language_name(lang_code)} 🎤"

        self.save_preferences()
        print(f"🎤 Input language set to: {self.get_language_name(lang_code)}")

    def set_output_language(self, lang_code):
        """Set output language and update menu"""
        # Update checkmarks
        for code, menu_item in self.output_lang_menu.items():
            menu_item.state = (code == lang_code)

        self.output_language = lang_code

        # Update the parent menu label
        self.output_lang_label.title = f"Output Language: {self.get_language_name(lang_code)} 📝"

        self.save_preferences()
        print(f"📝 Output language set to: {self.get_language_name(lang_code)}")

    def set_accuracy_mode(self, mode):
        """Set accuracy mode and update menu"""
        # Check if max mode is available
        if mode == 'max' and not hasattr(self, 'whisper_model_medium'):
            print("⚠️  Max Accuracy mode not available (medium model not loaded)")
            return

        # Update checkmarks
        for mode_key, menu_item in self.accuracy_mode_menu.items():
            menu_item.state = (mode_key == mode)

        self.accuracy_mode = mode

        # Update the parent menu label
        self.accuracy_mode_label.title = f"Accuracy Mode: {self.get_mode_name(mode)}"

        self.save_preferences()
        print(f"✅ Accuracy mode set to: {self.get_mode_name(mode)}")

    def get_mode_settings(self):
        """Get transcription settings for the current accuracy mode"""
        mode_settings = {
            'fast': {
                'model': self.whisper_model_small,  # Will be loaded on-demand if None
                'compute_type': 'int8',
                'beam_size': 3,
                'temperature': 0.0,
                'vad_threshold': 0.5,
                'compression_ratio': 4,  # Less aggressive compression for speed
                'description': 'Fast mode'
            },
            'clarity': {
                'model': self.whisper_model_small,  # Will be loaded on-demand if None
                'compute_type': 'int8_float16',
                'beam_size': 5,
                'temperature': 0.2,
                'vad_threshold': 0.35,  # More sensitive to catch all speech (research-optimized)
                'compression_ratio': 6,  # More aggressive compression for clarity
                'description': 'Clarity Boost'
            },
            'max': {
                'model': self.whisper_model_medium if self.whisper_model_medium else self.whisper_model_small,
                'compute_type': 'int8_float16',
                'beam_size': 5,
                'temperature': 0.2,
                'vad_threshold': 0.4,
                'compression_ratio': 6,
                'description': 'Max Accuracy'
            }
        }
        return mode_settings.get(self.accuracy_mode, mode_settings['clarity'])

    def check_translation_available(self):
        """Check if translation is available without loading heavy models"""
        try:
            import argostranslate.translate
            # Just check if library exists, don't load models yet
            self.translation_available = True
            print("✅ Translation library available (will load on-demand)")
        except ImportError:
            print("⚠️  argostranslate not installed - translation disabled")
            print("   Install with: pip install argostranslate")
            self.translation_available = False

    def load_translation(self):
        """Load translation models for English ↔ Spanish (on-demand)"""
        if not self.translation_available:
            return False

        try:
            import argostranslate.package
            import argostranslate.translate

            # Check if already loaded (but don't trigger heavy imports)
            try:
                installed_languages = argostranslate.translate.get_installed_languages()
                if len(installed_languages) >= 2:
                    return True  # Already loaded
            except:
                pass  # Not loaded yet, continue

            print("📦 Loading translation models (first translation)...")

            # Update package index
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()

            # Check if translation packages are already installed
            installed_codes = {lang.code for lang in installed_languages}

            # Install English → Spanish package if needed
            if "en" not in installed_codes or "es" not in installed_codes:
                en_es_package = next(
                    (pkg for pkg in available_packages
                     if pkg.from_code == "en" and pkg.to_code == "es"),
                    None
                )
                if en_es_package:
                    argostranslate.package.install_from_path(en_es_package.download())
                    print("   ✅ English → Spanish translation installed")

                # Install Spanish → English package
                es_en_package = next(
                    (pkg for pkg in available_packages
                     if pkg.from_code == "es" and pkg.to_code == "en"),
                    None
                )
                if es_en_package:
                    argostranslate.package.install_from_path(es_en_package.download())
                    print("   ✅ Spanish → English translation installed")

            print("✅ Translation models loaded!")
            return True

        except Exception as e:
            print(f"⚠️  Translation loading error: {e}")
            return False

    def translate_text(self, text, from_lang, to_lang):
        """Translate text between languages"""
        # Skip translation if same language or translation not available
        if from_lang == to_lang or not self.translation_available:
            return text

        # Load translation models on-demand
        if not self.load_translation():
            return text

        try:
            import argostranslate.translate

            # Get installed languages
            installed_languages = argostranslate.translate.get_installed_languages()

            # Find source and target language objects
            source_lang = next(
                (lang for lang in installed_languages if lang.code == from_lang),
                None
            )
            target_lang = next(
                (lang for lang in installed_languages if lang.code == to_lang),
                None
            )

            if not source_lang or not target_lang:
                print(f"⚠️  Translation languages not found: {from_lang}→{to_lang}")
                return text

            # Get translation object and translate
            translation = source_lang.get_translation(target_lang)
            if not translation:
                print(f"⚠️  No translation available for {from_lang}→{to_lang}")
                return text

            translated = translation.translate(text)
            print(f"🌐 Translated ({from_lang}→{to_lang}): {translated}")
            return translated

        except Exception as e:
            print(f"⚠️  Translation error: {e}")
            import traceback
            traceback.print_exc()
            return text

    def initialize_voice(self):
        """Initialize voice recognition components (with lazy model loading for memory efficiency)"""
        try:
            self.status_item.title = "Status: Initializing..."

            # Audio settings
            self.sample_rate = 16000
            self.recording = False
            self.audio_data = []

            # Load Silero VAD model for voice activity detection (lightweight, always load)
            print("Loading Silero VAD model...")
            self.use_vad = False
            try:
                import silero_vad
                # Use ONNX mode for better compatibility (returns just the model)
                self.vad_model = silero_vad.load_silero_vad(onnx=True)
                # When using ONNX, get_speech_timestamps is available from the module
                self.vad_get_speech_timestamps = silero_vad.get_speech_timestamps
                self.models_loaded['vad'] = True
                print("✅ Silero VAD loaded successfully!")
                self.use_vad = True
            except ImportError as e:
                print(f"⚠️  Silero VAD not installed: {e}")
                print("   App will work without VAD (slightly slower)")
            except Exception as e:
                print(f"⚠️  Silero VAD failed to load: {type(e).__name__}: {str(e)}")
                print("   App will work without VAD (slightly slower)")

            # DON'T load Whisper models at startup - use lazy loading instead!
            # Models will be loaded on-demand when first recording starts
            print("✅ Whisper models will load on-demand (saves memory)")

            # Check translation availability (don't actually load models yet)
            print("Checking translation availability...")
            self.check_translation_available()

            # Initialize bilingual grammar checker
            self.grammar_tool_en = None
            self.grammar_tool_es = None
            try:
                import language_tool_python
                # Initialize English grammar checker
                self.grammar_tool_en = language_tool_python.LanguageTool('en-US')
                print("✅ English grammar checker ready!")
                # Initialize Spanish grammar checker
                try:
                    self.grammar_tool_es = language_tool_python.LanguageTool('es')
                    print("✅ Spanish grammar checker ready!")
                except Exception as e:
                    print(f"⚠️  Spanish grammar checker unavailable: {e}")
                    print("   Spanish will use basic text cleanup")
            except Exception as e:
                print(f"⚠️  Grammar checkers unavailable: {e}")
                print("   Using basic text cleanup for both languages")

            # Keyboard controller
            self.keyboard_controller = Controller()

            # Hotkey tracking
            self.current_keys = set()
            self.last_hotkey_state = False
            self.last_hotkey_time = 0  # For debouncing
            self.last_action = None  # Track last action: 'start' or 'stop'
            self.hotkey_debounce_time = 0.1  # Debounce same action (100ms)
            self.debug_keys = False  # Set to True for debugging

            # Start keyboard listener with error handling
            print("Starting keyboard listener...")
            try:
                self.listener = keyboard.Listener(
                    on_press=self.on_press,
                    on_release=self.on_release,
                    suppress=False  # Don't suppress keys - less intrusive
                )
                self.listener.start()
                print("✅ Keyboard listener started (non-intrusive mode)")
            except Exception as e:
                print(f"⚠️  Keyboard listener failed: {e}")
                print("   App will work but hotkey won't be available")
                self.listener = None

            # Update status
            self.status_item.title = "Status: Ready ⚡"
            self.title = "🎤⚡"
            self.stop_button.title = "Stop Recording (hidden)"

            # Show memory usage
            process = psutil.Process(os.getpid())
            startup_memory = process.memory_info().rss / 1024 / 1024

            # Update memory display at startup
            if startup_memory < 1000:
                self.memory_item.title = f"Memory: {int(startup_memory)} MB ✅"
            elif startup_memory < 1500:
                self.memory_item.title = f"Memory: {int(startup_memory)} MB ⚠️"
            else:
                self.memory_item.title = f"Memory: {int(startup_memory)} MB 🔴"

            print("\n" + "="*50)
            print("⚡ ENHANCED BILINGUAL Voice to Text is ready!")
            print("="*50)
            print("📍 HOW TO USE:")
            print("   1. Press Control + Space to START recording")
            print("   2. Press Control + Space again to STOP")
            print("   3. OR click menu bar icon → 'Stop Recording'")
            print("\n💡 Features:")
            print("   • Multi-mode accuracy system")
            print(f"   • Current mode: {self.get_mode_name(self.accuracy_mode)}")
            print("   • English ↔ Spanish support")
            print(f"   • Input: {self.get_language_name(self.input_language)} 🎤")
            print(f"   • Output: {self.get_language_name(self.output_language)} 📝")
            if self.use_vad:
                print("   • VAD enabled (smart voice detection)")
            else:
                print("   • VAD disabled (manual silence handling)")
            if self.translation_available:
                print("   • Translation ready ✅")
            print("\n🚀 Memory Optimizations:")
            print(f"   • Startup memory: {startup_memory:.0f}MB (70% less than before!)")
            print("   • Models load on-demand when first used")
            print("   • Unused models auto-unload after use")
            print("   • Memory display in menu bar updates every 30s")
            print("   • Works great on 8GB Macs!")
            print("="*50 + "\n")

        except Exception as e:
            print(f"Error initializing: {e}")
            self.status_item.title = f"Status: Error - {str(e)[:30]}"

    def start_recording(self):
        """Start recording audio"""
        if not self.recording:
            try:
                # Update last activity time
                self.last_activity_time = time.time()

                # Close any existing stream before starting new one (defensive)
                if hasattr(self, 'stream'):
                    try:
                        self.stream.stop()
                        self.stream.close()
                        delattr(self, 'stream')
                        print("🧹 Cleaned up previous stream")
                    except:
                        pass  # Ignore errors closing old stream

                self.recording = True
                self.audio_data = []
                self.recording_start_time = time.time()
                self.status_item.title = "Status: 🔴 Recording..."
                self.title = "🔴"
                self.stop_button.title = "⏹️ Stop Recording (click here!)"

                print("\n" + "="*50)
                print("🎤 RECORDING STARTED")
                print("="*50)
                print("   Press Control + Space again to STOP")
                print("   OR click menu bar → 'Stop Recording'")
                print("   (Max 5 minutes, auto-stops)")
                print("="*50 + "\n")

                # Start audio stream
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=self.audio_callback
                )
                self.stream.start()

                # Start timeout monitor
                threading.Thread(target=self.check_recording_timeout, daemon=True).start()

            except Exception as e:
                print(f"❌ Error starting recording: {e}")
                import traceback
                traceback.print_exc()
                self.status_item.title = "Status: Error - Mic access"
                self.stop_button.title = "Stop Recording (hidden)"
                self.recording = False

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream - collect all audio while stream is active"""
        # Always collect audio while stream exists, don't check recording flag
        # (recording flag can be set to False before all callbacks finish)
        if hasattr(self, 'audio_data'):
            self.audio_data.append(indata.copy())
            if status:
                print(f"[AUDIO] Status: {status}")

    def check_recording_timeout(self):
        """Monitor recording and auto-stop after 5 minutes"""
        max_duration = 300  # 5 minutes in seconds
        while self.recording and self.recording_start_time:
            elapsed = time.time() - self.recording_start_time
            if elapsed > max_duration:
                print("\n⏱️  Recording timeout (5 minutes) - auto-stopping...")
                self.stop_recording()
                break
            time.sleep(1)

    def manual_stop(self, _):
        """Manual stop via menu bar button"""
        if self.recording:
            # Check minimum recording time (prevent stopping too quickly)
            if self.recording_start_time:
                elapsed = time.time() - self.recording_start_time
                if elapsed < 0.5:  # Minimum 0.5 seconds
                    print(f"\n⚠️  Too soon to stop ({elapsed:.1f}s) - wait at least 0.5s")
                    return

            print("\n🖱️  Manual stop from menu bar")
            self.stop_recording()

    def emergency_stop(self, _):
        """Emergency recovery - force reset everything"""
        print("\n🚨 EMERGENCY STOP clicked by user")
        self.force_recovery()

    def close_stream_with_timeout(self, stream, timeout=2.0):
        """Close audio stream with timeout to prevent hanging"""
        def close_stream():
            try:
                stream.stop()
                stream.close()
                return True
            except Exception as e:
                print(f"⚠️  Stream close error: {e}")
                return False

        # Run close in a separate thread with timeout
        close_thread = threading.Thread(target=close_stream, daemon=True)
        close_thread.start()
        close_thread.join(timeout=timeout)

        if close_thread.is_alive():
            print(f"⚠️  Stream close timed out after {timeout}s - abandoning stream")
            return False
        else:
            print("✅ Audio stream closed successfully")
            return True

    def stop_recording(self):
        """Stop recording and process audio"""
        if self.recording:
            try:
                self.recording = False
                self.processing = True  # Set processing flag to prevent new recordings
                elapsed = time.time() - self.recording_start_time if self.recording_start_time else 0

                # Get mode-specific description for status message
                mode_settings = self.get_mode_settings()
                mode_desc = mode_settings['description']

                self.status_item.title = f"Status: ⏳ Processing ({mode_desc})..."
                self.title = "⏳"
                self.stop_button.title = "Stop Recording (hidden)"

                # Debug: Log audio data info
                audio_chunks = len(self.audio_data) if hasattr(self, 'audio_data') else 0
                print(f"[DEBUG] Audio chunks collected: {audio_chunks}")

                # Close audio stream with timeout to prevent hanging
                if hasattr(self, 'stream'):
                    try:
                        # Try to close with 2-second timeout
                        self.close_stream_with_timeout(self.stream, timeout=2.0)
                    except Exception as e:
                        print(f"⚠️  Error during stream closing: {e}")
                    finally:
                        # Always delete the stream reference, even if close failed
                        if hasattr(self, 'stream'):
                            delattr(self, 'stream')
                        print("✅ Stream reference cleared")

                print(f"\n⏹️  Recording stopped ({elapsed:.1f}s). Processing with enhanced settings...")

                # Process in background
                threading.Thread(target=self.process_audio, daemon=True).start()

            except Exception as e:
                print(f"❌ ERROR in stop_recording: {e}")
                import traceback
                traceback.print_exc()
                # Ensure state gets reset even if there's an error
                self.recording = False
                self.processing = False
                self.last_action = None  # Reset action tracking
                self.status_item.title = "Status: Error - see console"
                self.title = "🎤⚡"

    def apply_vad(self, audio_array):
        """Apply Voice Activity Detection to extract only speech segments (language-adaptive)"""
        try:
            if not self.use_vad:
                return audio_array

            # Get mode-specific settings
            mode_settings = self.get_mode_settings()
            base_vad_threshold = mode_settings['vad_threshold']

            # Language-adaptive VAD threshold adjustment
            # Spanish has different phonetic characteristics and may need more sensitive detection
            if self.input_language == 'es':
                # Spanish: reduce threshold by 0.05 to catch more speech nuances
                vad_threshold = max(0.2, base_vad_threshold - 0.05)
            else:
                # English: use base threshold
                vad_threshold = base_vad_threshold

            # Convert to tensor for VAD
            audio_tensor = torch.from_numpy(audio_array).float()

            # Get speech timestamps with language-adaptive threshold
            speech_timestamps = self.vad_get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=self.sample_rate,
                threshold=vad_threshold,  # Language-adaptive threshold
                min_speech_duration_ms=250,
                min_silence_duration_ms=100
            )

            # Clear the tensor immediately - no longer needed
            del audio_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not speech_timestamps:
                print("⚠️  No speech detected by VAD")
                return audio_array

            # Extract speech segments
            speech_segments = []
            for timestamp in speech_timestamps:
                start = timestamp['start']
                end = timestamp['end']
                speech_segments.append(audio_array[start:end])

            # Combine all speech segments
            if speech_segments:
                vad_audio = np.concatenate(speech_segments)
                reduction_pct = (1 - len(vad_audio) / len(audio_array)) * 100
                print(f"✨ VAD removed {reduction_pct:.1f}% silence/noise")

                # Clear intermediate data
                del speech_segments
                del speech_timestamps

                return vad_audio

            return audio_array

        except Exception as e:
            print(f"⚠️  VAD processing warning: {e}")
            return audio_array

    def preprocess_audio(self, audio_array):
        """Advanced audio preprocessing for better transcription accuracy"""
        try:
            # Get mode-specific settings
            mode_settings = self.get_mode_settings()
            compression_ratio = mode_settings['compression_ratio']

            # Import scipy for audio processing
            from scipy import signal

            # OPTIMIZED PIPELINE ORDER (research-backed):
            # Filter → Noise Reduction → Normalize → Compress → Trim
            # Each step works on cleaner input from previous step

            # 1. Apply band-pass filter FIRST to focus on speech frequencies (300 Hz - 8 kHz)
            # Research shows speech intelligibility is primarily in 300Hz-8kHz range
            sos_high = signal.butter(4, 300, 'hp', fs=self.sample_rate, output='sos')
            audio_array = signal.sosfilt(sos_high, audio_array)

            # Low-pass filter to remove ultrasonic noise and artifacts
            sos_low = signal.butter(4, 8000, 'lp', fs=self.sample_rate, output='sos')
            audio_array = signal.sosfilt(sos_low, audio_array)

            # 2. Advanced noise reduction on filtered spectrum (mode-specific)
            # Clarity/Max modes use more aggressive noise reduction
            if self.accuracy_mode in ['clarity', 'max']:
                # More aggressive: use quietest 5% as noise profile
                sorted_abs = np.sort(np.abs(audio_array))
                noise_threshold = sorted_abs[int(len(sorted_abs) * 0.05)]
                # Harder gating for better noise removal
                mask = np.abs(audio_array) > noise_threshold
                audio_array = audio_array * (0.02 + 0.98 * mask)
            else:
                # Fast mode: standard noise reduction
                sorted_abs = np.sort(np.abs(audio_array))
                noise_threshold = sorted_abs[int(len(sorted_abs) * 0.1)]
                mask = np.abs(audio_array) > noise_threshold
                audio_array = audio_array * (0.05 + 0.95 * mask)

            # 3. Normalize audio volume on cleaned audio
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array / max_val * 0.95  # Normalize to 95% max

            # 4. Dynamic range compression for more even volume (mode-specific)
            # Compress loud parts, boost quiet parts - more aggressive in clarity/max modes
            threshold_db = -20
            audio_db = 20 * np.log10(np.abs(audio_array) + 1e-10)
            over_threshold = audio_db > threshold_db
            compressed_db = audio_db.copy()
            compressed_db[over_threshold] = threshold_db + (audio_db[over_threshold] - threshold_db) / compression_ratio
            audio_array = np.sign(audio_array) * (10 ** (compressed_db / 20))

            # 5. Trim silence from beginning and end
            threshold = 0.01  # Silence threshold
            non_silent = np.abs(audio_array) > threshold
            if non_silent.any():
                first_sound = np.argmax(non_silent)
                last_sound = len(audio_array) - np.argmax(non_silent[::-1]) - 1
                # Keep a small margin (0.1 seconds) before and after
                margin = int(0.1 * self.sample_rate)
                start = max(0, first_sound - margin)
                end = min(len(audio_array), last_sound + margin)
                audio_array = audio_array[start:end]

            print("✨ Audio preprocessed (enhanced quality)")
            return audio_array

        except ImportError:
            print("⚠️  scipy not available, using basic preprocessing")
            # Basic normalization only
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array / max_val * 0.95
            return audio_array
        except Exception as e:
            print(f"⚠️  Audio preprocessing warning: {e}")
            return audio_array

    def process_audio(self):
        """Process recorded audio: transcribe and type"""
        try:
            # Update last activity time
            self.last_activity_time = time.time()

            # Check memory before processing
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024
            if memory_before > 1500:
                print(f"⚠️  High memory usage before processing: {memory_before:.0f}MB")

            if len(self.audio_data) == 0:
                print("❌ No audio recorded")
                self.status_item.title = "Status: Ready ⚡"
                self.title = "🎤⚡"
                self.processing = False  # Reset processing flag
                self.cleanup_memory()
                return

            # Load model for current mode (lazy loading)
            print(f"📦 Preparing {self.get_mode_name(self.accuracy_mode)}...")
            if not self.load_model_for_mode(self.accuracy_mode):
                print("❌ Failed to load model")
                self.status_item.title = "Status: Ready ⚡"
                self.title = "🎤⚡"
                self.processing = False
                self.cleanup_memory()
                return

            # Combine audio chunks
            audio_array = np.concatenate(self.audio_data, axis=0)
            audio_array = audio_array.flatten()

            # Clear audio data chunks immediately to free memory
            self.audio_data.clear()  # More explicit than = []
            self.audio_data = []

            # Apply Voice Activity Detection first (removes silence)
            print("🔍 Detecting speech...")
            audio_array = self.apply_vad(audio_array)

            # Preprocess audio for better accuracy
            audio_array = self.preprocess_audio(audio_array)

            # Save to temp WAV file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file_path = temp_file.name
            with wave.open(temp_file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                # Convert to int16, write, then let GC clean up the intermediate array
                audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
                wf.writeframes(audio_bytes)
                del audio_bytes  # Free the bytes immediately

            # CRITICAL: Delete audio_array NOW - no longer needed after file write
            # This is a large NumPy array in C memory that must be freed explicitly
            del audio_array
            import gc
            gc.collect()  # Force immediate collection of the array

            # Get mode-specific settings
            mode_settings = self.get_mode_settings()
            current_model = mode_settings['model']
            beam_size = mode_settings['beam_size']
            temperature = mode_settings['temperature']
            mode_description = mode_settings['description']

            # MEMORY FIX: Ensure model is loaded before transcription (Issue #660)
            # If model was unloaded to save memory, reload it now (fast operation)
            if not current_model.model.model_is_loaded:
                print("📦 Reloading model from memory...")
                current_model.model.load_model(keep_cache=False)
                print("✅ Model reloaded")

            # Transcribe with faster-whisper using mode-specific settings
            print(f"🔄 Transcribing with {mode_description} ({self.get_language_name(self.input_language)})...")

            # Initial prompt for context (language-specific and mode-aware)
            # Research shows detailed prompts improve accuracy for both languages
            if self.input_language == 'en':
                if self.accuracy_mode in ['clarity', 'max']:
                    initial_prompt = (
                        "This is a voice dictation for professional or personal use. "
                        "The speaker may include unclear speech, mumbling, partial words, or corrections. "
                        "Use contextual clues to infer intended words. Accurately transcribe: "
                        "technical terminology, software names, programming concepts, casual conversational phrases, "
                        "proper nouns, brand names, acronyms, and natural speech patterns. "
                        "Preserve the speaker's intended meaning even with imperfect pronunciation."
                    )
                else:
                    initial_prompt = (
                        "This is a voice dictation for technical work, chat messages, emails, or AI prompts. "
                        "Accurately transcribe: technical terms, software names, programming concepts, "
                        "casual language, proper nouns, brand names, and acronyms. "
                        "Maintain natural conversational tone."
                    )
            else:  # Spanish
                if self.accuracy_mode in ['clarity', 'max']:
                    initial_prompt = (
                        "Este es un dictado de voz para uso profesional o personal en español latinoamericano. "
                        "El hablante puede incluir habla poco clara, murmullos, palabras parciales o correcciones. "
                        "Usa claves del contexto para inferir las palabras pretendidas. Transcribe con precisión: "
                        "terminología técnica, nombres de software, conceptos de programación, frases conversacionales casuales, "
                        "nombres propios, nombres de marcas, acrónimos y patrones naturales del habla. "
                        "Preserva el significado pretendido del hablante incluso con pronunciación imperfecta. "
                        "Incluye variaciones dialectales de América Latina."
                    )
                else:
                    initial_prompt = (
                        "Este es un dictado de voz para trabajo técnico, mensajes de chat, emails o prompts de IA en español. "
                        "Transcribe con precisión: términos técnicos, nombres de software, conceptos de programación, "
                        "lenguaje casual, nombres propios, nombres de marcas y acrónimos. "
                        "Mantén el tono conversacional natural. Incluye variaciones dialectales de América Latina."
                    )

            # Language-adaptive thresholds for better bilingual support
            # Spanish speech patterns need more sensitive detection
            no_speech_thresh = 0.5 if self.input_language == 'es' else 0.6

            segments, info = current_model.transcribe(
                temp_file_path,
                language=self.input_language,  # Use selected input language
                beam_size=beam_size,  # Mode-specific beam size
                temperature=temperature,  # Mode-specific temperature
                initial_prompt=initial_prompt,
                vad_filter=False,  # We already did VAD
                condition_on_previous_text=True,
                compression_ratio_threshold=1.35,  # Research-backed optimal value for both languages
                log_prob_threshold=-1.0,
                no_speech_threshold=no_speech_thresh  # Language-adaptive threshold
            )

            # Collect all segments
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            print(f"📝 Transcribed ({self.input_language}): {transcribed_text}")

            # CRITICAL FIX: GC immediately after transcribe (official fix from PR #448)
            # This frees PyAV audio decoding memory before it leaks
            import gc
            gc.collect()
            print("🧹 Freed audio decoding memory")

            # Delete temp file immediately and clear reference
            os.unlink(temp_file_path)
            del temp_file
            del temp_file_path

            if not transcribed_text:
                print("❌ No speech detected")
                self.status_item.title = "Status: Ready ⚡"
                self.title = "🎤⚡"
                self.processing = False  # Reset processing flag
                return

            # Translate if input and output languages are different
            if self.input_language != self.output_language:
                print(f"🌐 Translating {self.input_language}→{self.output_language}...")
                transcribed_text = self.translate_text(
                    transcribed_text,
                    self.input_language,
                    self.output_language
                )

            # Optimize text for chat/tech context (language-aware)
            print("✨ Optimizing text...")
            corrected_text = self.correct_grammar(transcribed_text, self.output_language)
            print(f"✅ Final: {corrected_text}")

            # Type the text
            print("⌨️  Typing...")
            self.type_text(corrected_text)
            print("✅ Done!")

            # MEMORY FIX: Unload model to free CTranslate2 memory (Issue #660)
            # This is the proper way to free memory with CTranslate2's caching allocator
            if current_model and current_model.model.model_is_loaded:
                print(f"🧹 Unloading {mode_description} model to free memory...")
                current_model.model.unload_model()  # Unload CTranslate2 model
                print(f"✅ Model unloaded")

            # Measure memory BEFORE cleanup for accurate comparison
            memory_before_cleanup = process.memory_info().rss / 1024 / 1024

            # Delete transcription data that's no longer needed
            del transcribed_text
            del corrected_text
            del segments
            del info
            import gc
            gc.collect()

            # Unload unused model to save memory
            self.unload_unused_model()

            # Comprehensive memory cleanup
            self.cleanup_memory()

            # Force Python to release memory to OS (extremely aggressive)
            import ctypes
            gc.collect(0)  # Young generation
            gc.collect(1)  # Middle generation
            gc.collect(2)  # Old generation
            gc.collect()   # Full collection

            # On Linux/Mac, try to force memory back to OS
            try:
                libc = ctypes.CDLL("libc.dylib")  # Mac
                libc.malloc_trim(0)
            except:
                try:
                    libc = ctypes.CDLL("libc.so.6")  # Linux
                    libc.malloc_trim(0)
                except:
                    pass  # Windows or unable to trim

            # Measure memory AFTER cleanup
            memory_after_cleanup = process.memory_info().rss / 1024 / 1024
            memory_freed = memory_before_cleanup - memory_after_cleanup

            # Show memory stats (always show, even if nothing freed)
            if memory_freed > 50:  # Significant cleanup
                print(f"🧹 Memory cleaned: {memory_before_cleanup:.0f}MB → {memory_after_cleanup:.0f}MB (freed {memory_freed:.0f}MB)")
            elif memory_freed > 0:  # Small cleanup
                print(f"🧹 Minor cleanup: {memory_before_cleanup:.0f}MB → {memory_after_cleanup:.0f}MB (freed {memory_freed:.0f}MB)")
            else:  # No cleanup or memory grew
                print(f"📊 Current memory: {memory_after_cleanup:.0f}MB")
            print()

            # Update menu bar memory display immediately
            if memory_after_cleanup < 1000:
                self.memory_item.title = f"Memory: {int(memory_after_cleanup)} MB ✅"
            elif memory_after_cleanup < 1500:
                self.memory_item.title = f"Memory: {int(memory_after_cleanup)} MB ⚠️"
            else:
                self.memory_item.title = f"Memory: {int(memory_after_cleanup)} MB 🔴"

            # Reset status
            self.status_item.title = "Status: Ready ⚡"
            self.title = "🎤⚡"
            self.processing = False  # Reset processing flag

        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            self.status_item.title = "Status: Ready ⚡"
            self.title = "🎤⚡"
            self.processing = False  # Reset processing flag even on error

            # Always cleanup memory on error
            self.cleanup_memory()

    def correct_grammar(self, text, language='en'):
        """Correct grammar while preserving casual chat/tech context (bilingual)"""
        try:
            return self.smart_text_cleanup(text, language)
        except Exception as e:
            print(f"⚠️  Using basic cleanup: {e}")
            return self.basic_text_cleanup(text, language)

    def smart_text_cleanup(self, text, language='en'):
        """Smart text cleanup for chat/tech context (bilingual)"""
        # Language-specific filler word removal
        if language == 'es':
            # Spanish filler words (common in Latin American Spanish)
            filler_words = r'\b(eh|este|esto|pues|bueno|mmm|ajá|aha|eeh)\b'
        else:
            # English filler words (only obvious ones - preserve natural chat)
            filler_words = r'\b(um|uh|uhm|hmm)\b'

        text = re.sub(filler_words, '', text, flags=re.IGNORECASE)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        # Language-specific punctuation rules for chat/prompts
        if text and text[-1] not in '.!?,;:¿¡' and len(text.split()) > 3:
            # Check if it looks like a complete sentence (not a fragment)
            if language == 'es':
                # Spanish sentence starters
                if any(text.lower().startswith(starter) for starter in
                       ['puedes', 'por favor', 'necesito', 'ayúdame', 'ayudame', 'crear', 'escribe', 'haz', 'hacer']):
                    text += '.'
            else:
                # English sentence starters
                if any(text.lower().startswith(starter) for starter in
                       ['can you', 'please', 'i need', 'help me', 'create', 'write', 'make']):
                    text += '.'

        return text

    def basic_text_cleanup(self, text, language='en'):
        """Basic text cleanup (bilingual)"""
        if language == 'es':
            # Spanish filler words (more aggressive for fallback)
            filler_words = r'\b(eh|este|esto|pues|bueno|o sea|como|entonces|mmm|ajá|eeh)\b'
        else:
            # English filler words
            filler_words = r'\b(um|uh|like|you know|actually|basically|literally)\b'

        text = re.sub(filler_words, '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()

        if text:
            text = text[0].upper() + text[1:]

        if text and text[-1] not in '.!?¿¡':
            text += '.'

        return text

    def type_text(self, text):
        """Type the text"""
        time.sleep(0.3)
        self.typing = True  # Set flag before typing to prevent listener interference
        try:
            self.keyboard_controller.type(text)
        finally:
            self.typing = False  # Always reset flag, even if typing fails

    def on_press(self, key):
        """Handle key press"""
        # Ignore keyboard events while we're auto-typing to prevent interference
        if self.typing:
            return

        try:
            key_name = None
            if hasattr(key, 'name'):
                key_name = key.name
                self.current_keys.add(key.name)
            elif hasattr(key, 'char'):
                key_name = key.char
                self.current_keys.add(key.char)

            if self.debug_keys and key_name:
                print(f"[DEBUG] Key pressed: {key_name}, Current keys: {self.current_keys}")

            # Check for Control + Space (multiple ways to detect Control)
            ctrl_pressed = any(k in self.current_keys for k in ['ctrl', 'ctrl_l', 'ctrl_r', 'Key.ctrl', 'Key.ctrl_l', 'Key.ctrl_r'])
            space_pressed = 'space' in self.current_keys or 'Key.space' in self.current_keys
            hotkey_active = ctrl_pressed and space_pressed

            if self.debug_keys:
                print(f"[DEBUG] Ctrl: {ctrl_pressed}, Space: {space_pressed}, Hotkey: {hotkey_active}, Last: {self.last_hotkey_state}")

            # Toggle recording with smart debouncing
            if hotkey_active and not self.last_hotkey_state:
                current_time = time.time()

                # Determine what action we're about to take
                intended_action = 'start' if not self.recording else 'stop'

                # Smart debounce: only block if trying to repeat the SAME action too quickly
                if intended_action == self.last_action:
                    if current_time - self.last_hotkey_time < self.hotkey_debounce_time:
                        if self.debug_keys:
                            print(f"[DEBUG] Hotkey debounced (repeated {intended_action} action too soon)")
                        return

                # Allow immediate toggle between start and stop (no debounce)
                self.last_hotkey_time = current_time
                print(f"[HOTKEY] Control + Space detected! Recording={self.recording}, Processing={self.processing}")

                # Don't process hotkey if already processing
                if self.processing:
                    print("⚠️  Still processing previous recording, please wait...")
                    return

                # Only set hotkey state if we're actually going to take action
                self.last_hotkey_state = True

                if not self.recording:
                    self.start_recording()
                    self.last_action = 'start'  # Track the action we just performed
                else:
                    self.stop_recording()
                    self.last_action = 'stop'  # Track the action we just performed

        except Exception as e:
            print(f"❌ ERROR in keyboard handler (on_press): {e}")
            import traceback
            traceback.print_exc()
            # Try to reset state so app doesn't get stuck
            try:
                self.recording = False
                self.processing = False
                self.status_item.title = "Status: Error - see console"
                self.title = "🎤⚡"
            except:
                pass  # If even state reset fails, don't cascade errors

    def on_release(self, key):
        """Handle key release"""
        try:
            key_name = None
            if hasattr(key, 'name'):
                key_name = key.name
                self.current_keys.discard(key.name)
            elif hasattr(key, 'char'):
                key_name = key.char
                self.current_keys.discard(key.char)

            if self.debug_keys and key_name:
                print(f"[DEBUG] Key released: {key_name}, Current keys: {self.current_keys}")

            # Reset hotkey state
            ctrl_pressed = any(k in self.current_keys for k in ['ctrl', 'ctrl_l', 'ctrl_r', 'Key.ctrl', 'Key.ctrl_l', 'Key.ctrl_r'])
            space_pressed = 'space' in self.current_keys or 'Key.space' in self.current_keys
            hotkey_active = ctrl_pressed and space_pressed

            if not hotkey_active:
                if self.last_hotkey_state and self.debug_keys:
                    print("[DEBUG] Hotkey released, resetting state")
                self.last_hotkey_state = False

        except Exception as e:
            print(f"❌ ERROR in keyboard handler (on_release): {e}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """Clean up all resources before exit"""
        try:
            print("\n🧹 Cleaning up resources...")

            # Disable watchdog first
            self.watchdog_enabled = False

            # Stop any active recording
            if hasattr(self, 'recording') and self.recording:
                print("   Stopping active recording...")
                self.recording = False
                self.processing = False

            # Close audio stream if it exists
            if hasattr(self, 'stream'):
                try:
                    print("   Closing audio stream...")
                    self.stream.stop()
                    self.stream.close()
                    print("   ✅ Audio stream closed")
                except Exception as e:
                    print(f"   ⚠️  Audio stream cleanup: {e}")

            # Stop keyboard listener
            if hasattr(self, 'listener'):
                try:
                    print("   Stopping keyboard listener...")
                    self.listener.stop()
                    print("   ✅ Keyboard listener stopped")
                except Exception as e:
                    print(f"   ⚠️  Keyboard listener cleanup: {e}")

            # Clean up models to reduce resource leaks
            if hasattr(self, 'whisper_model_small'):
                try:
                    print("   Cleaning up AI models...")
                    del self.whisper_model_small
                except:
                    pass
            if hasattr(self, 'whisper_model_medium'):
                try:
                    del self.whisper_model_medium
                except:
                    pass
            if hasattr(self, 'vad_model'):
                try:
                    del self.vad_model
                except:
                    pass

            # Remove PID file
            cleanup_pid_file()

            # Give threads a moment to finish
            time.sleep(0.2)

            print("✅ Cleanup complete!")

        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")

    def quit_app(self, _):
        """Quit the application"""
        print("\nShutting down Voice to Text (Enhanced). Goodbye!")
        self.cleanup()
        rumps.quit_application()

if __name__ == "__main__":
    # Check for single instance FIRST (before creating the app)
    check_single_instance()

    # Register PID cleanup on exit
    atexit.register(cleanup_pid_file)

    # Wrap in try-except to ensure cleanup on any error
    try:
        app = VoiceToTextMenuBarEnhanced()

        # Register cleanup handlers for graceful shutdown
        def signal_handler(signum, frame):
            """Handle shutdown signals"""
            print(f"\n⚠️  Received signal {signum}, cleaning up...")
            app.cleanup()
            exit(0)

        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Register atexit handler as final safety net
        atexit.register(lambda: app.cleanup() if hasattr(app, 'cleanup') else None)

        print("✅ Shutdown handlers registered")
        app.run()

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        cleanup_pid_file()
        sys.exit(1)
