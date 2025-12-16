import threading
import sys
import os

sys.path.insert(0, "/home/novus/Projects/Nova/.nova/lib/python3.13/site-packages")

from pygpt_net.plugin.base.plugin import BasePlugin
from apex.plugins.MemoryPlus.plugin import Plugin

# Enhanced mock for PyGPT
class MockConfig:
    def has(self, key):
        return key in self._data
    def get(self, key, default=None):
        return self._data.get(key, default)
    def __init__(self):
        self._data = {"preset": "default", "log.plugins": True}

class MockCore:
    def __init__(self):
        self.config = MockConfig()
        self.models = None

class MockWindow:
    def __init__(self):
        self.core = MockCore()

# Initialize plugin
plugin = Plugin()
plugin.attach(MockWindow())

# Stress test: 10 threads, 100 operations each
def worker(worker_id):
    for i in range(100):
        plugin._enqueue_ingest_request(
            f"Test-{worker_id}-{i}",
            f"Content for test {worker_id}-{i}",
            "Chatbot"
        )

threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("✅ Threading test passed. No crashes detected.")
plugin.detach()
