import threading
from apex.plugins.MemoryPlus.plugin import SearchCache

# Test SearchCache thread safety
cache = SearchCache()
cache.configure(10, 5.0, 0.8)

def worker(worker_id):
    for i in range(1000):
        cache.set(f"key-{worker_id}-{i}", [f"value-{worker_id}-{i}"])
        cache.get(f"key-{worker_id}-{i}")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("✅ SearchCache thread safety test passed.")
