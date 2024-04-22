import time

# Used to behcnmark each components
def benchmark(function):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = function(*args, **kwargs)
        t2 = time.time()
        print(f"{function.__name__} took {(t2-t1)*1000:.4f} ms.")
        return result
    return wrapper
