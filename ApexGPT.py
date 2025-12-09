import os
import sys
import importlib.util
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
NYX_CORE_ROOT = PROJECT_ROOT / ".nova"
PLUGINS_DIR = NYX_CORE_ROOT / "plugins"

def load_single_plugin(plugin_name: str):
    """Load a single plugin directory by name"""
    plugin_dir = PLUGINS_DIR / plugin_name
    if not plugin_dir.exists():
        return None
    
    plugin_init = plugin_dir / "__init__.py"
    if not plugin_init.exists():
        return None
        
    try:
        spec = importlib.util.spec_from_file_location(plugin_name, plugin_init)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, "Plugin"):
            plugin_class = getattr(module, "Plugin")
            return plugin_class()
        else:
            print(f"[ERROR] Plugin '{plugin_name}' missing 'Plugin' class")
            return None
    except Exception as e:
        print(f"[PLUGIN LOAD ERROR] {plugin_name}: {e}")
        traceback.print_exc()
        return None

def main():
    # === STEP 1: ONLY LOAD MEMORYPLUS ===
    memoryplus_plugin = load_single_plugin("MemoryPlus")
    if not memoryplus_plugin:
        print("[CRITICAL] MemoryPlus failed to load")
        sys.exit(1)
    
    # === STEP 2: Initialize PyGPT with ONLY this plugin ===
    from pygpt_net.app import Application
    app = Application()
    
    # Inject the single loaded plugin into the app
    app.plugin_manager.plugins = [memoryplus_plugin]
    app.plugin_manager.active = [memoryplus_plugin.id]  # Track only this ID as active

    # Start the app
    app.start()

if __name__ == "__main__":
    main()
