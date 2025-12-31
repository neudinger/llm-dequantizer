# llm-dequantizer

## Install ZIG

```bash
curl https://raw.githubusercontent.com/tristanisham/zvm/master/install.sh | bash

zvm i 0.15.2
zvm use 0.15.2
```

## Dowload a safetensor file

### Direct safetensors url


```bash
wget -O gptoss-model.safetensors https://huggingface.co/openai/gpt-oss-20b/resolve/main/original/model.safetensors
```

Require

```zig
const tensor_name = "block.0.attn.qkv.weight";
```

----


```bash
wget -O tinyllama-model.safetensors https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
```

Require

```zig
const tensor_name = "model.layers.0.self_attn.q_proj.weight";
```


## Build dequantizer

gpt-oss-20b ready

```bash
zig build -Doptimize=ReleaseSafe run -- ./gptoss-model.safetensors
```

OR

```bash
zig build -Doptimize=ReleaseSafe
./zig-out/bin/dequantizer ./model.safetensors
```

## ZLS

```bash
git clone --depth 1 --branch 0.15.1 https://github.com/zigtools/zls
cd zls
zig build -Doptimize=ReleaseSafe
```


## Summary

This repository implements a "Full-Stack" performance strategy, optimizing everything from how the Operating System handles the file on disk down to the specific CPU instructions used for math.

Here is a summary of every performance paradigm and strategy used in the code:

### 1. Memory & I/O Optimization (OS Level)

The code bypasses standard "file reading" to work directly with the OS kernel's memory management.

* **Memory Mapping (`mmap`):**
* **Strategy:** Instead of `read()`, it maps the model file directly into the process's virtual memory address space.
* **Benefit:** Zero-copy access. The OS lazily loads pages from the disk only when the code touches them.


* **Memory Advisories (`madvise`):**
* **`MADV_SEQUENTIAL`:** Tells the OS kernel, "I will read this data in order," triggering aggressive **Read-Ahead** (loading future pages into RAM before the code asks for them).
* **`MADV_HUGEPAGE`:** Forces the OS to use **2MB memory pages** (instead of the standard 4KB). This reduces **TLB (Translation Lookaside Buffer) misses**, a common bottleneck when jumping through large arrays.


* **Aligned Allocation:**
* **Strategy:** `allocator.alignedAlloc(..., 4096)`.
* **Benefit:** Ensures data starts at a memory boundary compatible with **AVX-512** instructions, preventing crashes or performance penalties associated with unaligned SIMD loads.



### 2. Parallelism & Scheduling (Thread Level)

The code takes manual control over how the CPU schedules its work, rather than trusting the OS default scheduler.

* **Explicit Thread Pooling:**
* **Strategy:** Spawns exactly  threads (where  = CPU count) and divides the work into disjoint "slices" so no locks or synchronization (mutexes) are needed.


* **CPU Affinity / Pinning (`sched_setaffinity`):**
* **Strategy:** The `pinToCpu` function forces each thread to run on a specific physical core.
* **Benefit:** **Cache Locality**. If a thread pauses and resumes, it stays on the same core, meaning the data it was working on is likely still hot in the L1/L2 cache. It also prevents the OS from constantly moving threads between cores (context switching).



### 3. SIMD & Vectorization (Instruction Level)

Instead of processing one number at a time (Scalar), the code uses **SIMD (Single Instruction, Multiple Data)** to process 16 or 32 numbers per cycle.

* **Zig `@Vector` Intrinsics:**
* **Strategy:** Uses types like `@Vector(32, f32)`.
* **Benefit:** Compiles directly to hardware vector instructions (AVX2/AVX-512 on x86, NEON on ARM).


* **Branchless Nibble Unpacking:**
* **Strategy:** To read 4-bit weights, it uses bitwise operators (`&`, `>>`) and **Vector Shuffling** (`@shuffle`) to unpack bytes into integers without using `if/else` statements.


* **Lookup Table (LUT) Gather:**
* **Strategy:** Instead of calculating values, it uses the 4-bit integer as an index to instantly "gather" the correct floating-point value from a pre-defined constant vector (`lut_vec`).



### 4. Micro-Architecture Optimization (Cache Level)

The code is written to keep the CPU execution units fed with data, minimizing idle time.

* **Software Prefetching (`@prefetch`):**
* **Strategy:** Explicitly issues `prefetch` instructions for data `VecWidth * 4` steps ahead.
* **Benefit:** Hides **Memory Latency**. While the CPU calculates the current chunk, the memory controller fetches the *next* chunk from RAM into L1 cache, so it's ready exactly when needed.


* **Loop Tiling / Blocking:**
* **Strategy:** The `matVecMul` function processes **4 rows at a time** (`BlockRows = 4`).
* **Benefit:** **Register Reuse**. It loads the input vector (`x_vec`) once and uses it against 4 different weight rows, reducing the total memory bandwidth required.



### 5. Compiler-Level Optimization (Language Level)

The code leverages Zig's unique features to remove runtime overhead.

* **Comptime Calculation:**
* **Strategy:** `comptime generateInterleaveMask()`.
* **Benefit:** Complex mask generation happens during *compilation*. The final binary just contains the hardcoded constant, costing 0 CPU cycles at runtime.


* **Unsafe Math Modes:**
* **Strategy:** `@setFloatMode(.optimized)` and `@setRuntimeSafety(false)`.
* **Benefit:** Allows the compiler to re-order floating point operations for speed (breaking strict IEEE-754 compliance) and removes array bounds checking in the hot loop.



### Summary Table

| Paradigm | Function/Keyword Used | Goal |
| --- | --- | --- |
| **I/O** | `mmap`, `madvise` | Zero-copy, OS-managed caching. |
| **Virtual Memory** | `MADV_HUGEPAGE` | Reduce TLB cache misses. |
| **Concurrency** | `std.Thread`, `setaffinity` | Maximize core usage, preserve L1 cache. |
| **Vectorization** | `@Vector`, `@shuffle` | Process 32 floats per instruction (SIMD). |
| **Latency Hiding** | `@prefetch` | Fetch data before it is needed. |
| **Register Usage** | Loop Tiling (4x) | Reuse loaded data multiple times. |
| **Zero-Overhead** | `comptime` | Pre-calculate constants at build time. |


----

## Possible improvement

### 1. Kernel Fusion (On-the-Fly Dequantization)

**Impact: High (2x-4x Speedup)**
The current code performs dequantization in two distinct steps:

1. **Expand:** Reads compressed `int4` from disk  Writes massive `f32` array to RAM.
2. **Compute:** Reads massive `f32` array from RAM  Computes Matrix Multiplication.

**The Optimization:**
Merge these steps. Do not write the intermediate `f32` tensor to RAM.
Instead, keep the weights compressed in RAM. During the Matrix Multiplication loop, load the `int4` data into a register, "unpack" it to `f32` *inside the CPU register*, and immediately multiply-accumulate with the input.

* **Why:** Memory bandwidth is usually the bottleneck in LLM inference. Expanding 4-bit weights to 32-bit floats increases memory traffic by **8x**. Fusing them eliminates this traffic.

### 2. Parallelized Matrix Multiplication

**Impact: Massive (Linear scaling with cores)**
The provided code parallelizes the *loading* (`loadTensorParallel`), but the actual math (`matVecMul`) runs on a **single thread** in the main function.

```zig
// Current:
st_file.loadTensorParallel(...) // Uses 16 cores
matVecMul(...)                  // Uses 1 core (Slow!)

```

**The Optimization:**
Apply the same `std.Thread.spawn` pattern to `matVecMul`. Split the rows of the weight matrix across available cores.

* **Strategy:** If you have 16 cores and 4096 rows, each core computes the dot product for 256 rows independently.

### 3. VNNI / Integer Arithmetic Instructions

**Impact: Medium/High (Latency Reduction)**
The current code casts 4-bit integers to 32-bit floats (`f32`) for the math.

```zig
result_vec[idx] = lut_vec[_idx]; // Converts to f32
sum += x * w;                    // f32 multiplication

```

**The Optimization:**
Modern CPUs (Intel Cascade Lake+, AMD Zen 4+) have **VNNI** (Vector Neural Network Instructions) or **AMX** (Advanced Matrix Extensions).
Instead of converting to float, you can perform the dot product using integers:

1. Convert 4-bit weight  8-bit integer.
2. Quantize Input vector  8-bit integer.
3. Use `vpdpbusd` (AVX-512) to do `int8 * int8` accumulation (4x faster than float math).

### 4. NUMA-Aware Compute (Data Locality)

**Impact: High (On Multi-Socket Servers)**
On high-end servers (e.g., Dual Xeon or Threadripper), RAM is physically attached to specific CPU dies (NUMA nodes).

* **Current Issue:** One thread might allocate memory on Node 0, but a thread on Node 1 tries to read it. This traffic must cross the slow inter-socket link (UPI/Infinity Fabric).
* **The Optimization:** Ensure that the thread *computing* rows  is the exact same thread (pinned to the same core) that *loaded* rows . This keeps data in the local RAM bank of that specific CPU die.

### 5. Activation Fusion

**Impact: Medium**
In a real neural network, a matrix multiplication is usually followed by a non-linear activation (like ReLU, SiLU, or GeLU).

* **Current:** `MatMul -> Write RAM` ... `Read RAM -> SiLU -> Write RAM`.
* **Optimization:** Apply the SiLU function **inside the MatMul loop** (specifically, on the accumulation register) before writing the final result to RAM. This saves another round-trip of memory reads/writes.

### Summary Diagram: The Optimized Pipeline

Here is how the data flow changes with these optimizations:

| Stage | Current Implementation | Optimized Implementation |
| --- | --- | --- |
| **Storage** | 4-bit (Disk) | 4-bit (Disk) |
| **RAM** | **32-bit (Expanded 8x)** | **4-bit (Compressed)** |
| **L1 Cache** | 32-bit Floats | 4-bit Integers |
| **Registers** | Convert  F32 | Convert  F32 (or Int8) |
| **ALU** | F32 Multiply-Add | Fused Multiply-Add |
| **Output** | Write to RAM | Write to RAM |

By keeping the data compressed until it hits the CPU registers (Kernel Fusion), to reduce the stress on the RAM, which is the primary limiter for LLM speed.