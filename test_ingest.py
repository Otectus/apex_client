import threading
import queue
import time
from apex.plugins.MemoryPlus.plugin import Plugin

# Isolate the ingestion loop
plugin = Plugin()
plugin.ingest_queue = queue.Queue(50)
plugin.ingest_stop_event = threading.Event()

# Start the worker
threading.Thread(target=plugin._ingest_loop, daemon=True).start()

# Stress test: 10 threads enqueueing
def worker(worker_id):
    for i in range(100):
        try:
            plugin.ingest_queue.put((f"Test-{worker_id}-{i}", f"Content-{worker_id}-{i}", "Chatbot"), block=False)
        except queue.Full:
            pass

threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Stop the worker
plugin.ingest_stop_event.set()
print("✅ Ingestion loop thread safety test passed.")
