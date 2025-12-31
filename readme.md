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
wget https://huggingface.co/openai/gpt-oss-20b/resolve/main/original/model.safetensors
```

Require

```zig
const tensor_name = "block.0.attn.qkv.weight";
```

----


```bash
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
```

Require

```zig
const tensor_name = "model.layers.0.self_attn.q_proj.weight";
```


## Build dequantizer

gpt-oss-20b ready

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
