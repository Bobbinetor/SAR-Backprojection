import cupy as cp

# Print CUDA version
print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")

# Get device information
device_count = cp.cuda.runtime.getDeviceCount()
print(f"Number of CUDA devices: {device_count}")

for i in range(device_count):
    props = cp.cuda.runtime.getDeviceProperties(i)
    print(f"Device {i}: {props['name'].decode()}")
    print(f"  Memory: {props['totalGlobalMem'] / (1024**3):.2f} GB")
    print(f"  Compute Capability: {props['major']}.{props['minor']}")
    
# Run a simple matrix operation on GPU
x_gpu = cp.arange(1_000_000, dtype=cp.float32)
y_gpu = cp.arange(1_000_000, dtype=cp.float32)
z_gpu = cp.add(x_gpu, y_gpu)
print(f"CUDA computation result (first 5 elements): {z_gpu[:5]}")
print("CUDA is working correctly!")