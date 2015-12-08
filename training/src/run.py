import time
import progress

with progress.start('Reading training tensor: %(percentage)i%%. Elapsed: %(elapsed)s', 99) as update:
    for i in range(100):
        update(i)
        time.sleep(0.02)