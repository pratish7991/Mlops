import time
from torch.utils.data import DataLoader


def benchmark(dataset, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    start = time.time()

    for _ in loader:
        pass

    end = time.time()

    print(f"Time taken with {num_workers} workers: {end - start:.2f} sec")
