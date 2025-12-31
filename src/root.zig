//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
const builtin = @import("builtin");

fn pinToCpu(cpu_id: usize) !void {
    if (builtin.os.tag == .linux) {
        var mask: std.os.linux.cpu_set_t = undefined;
        @memset(std.mem.asBytes(&mask), 0);

        const mask_bytes = std.mem.asBytes(&mask);
        const byte_index = cpu_id / 8;
        const bit_offset = @as(u3, @intCast(cpu_id % 8));

        if (byte_index < mask_bytes.len) {
            mask_bytes[byte_index] |= @as(u8, 1) << bit_offset;
            _ = std.os.linux.sched_setaffinity(0, &mask) catch {
                std.debug.print("Can not pin thread cpu continue with non pined core", .{});
            };
        }
    }
}

fn generateInterleaveMask() [32]i32 {
    var mask: [32]i32 = undefined;
    for (0..16) |idx| {
        mask[2 * idx] = @as(i32, @intCast(idx));
        mask[2 * idx + 1] = ~@as(i32, @intCast(idx));
    }
    return mask;
}

// https://huggingface.co/docs/optimum/concept_guides/quantization
fn dequantizeBlock(
    target: []f32,
    block_bytes: []const u8,
    scale: f32,
) void {
    const vector_width = 32;
    if (target.len < vector_width) {
        return;
    }

    const RawVec = @Vector(16, u8);

    // Load 16 bytes
    const packed_vec: RawVec = @as(*const [16]u8, @ptrCast(block_bytes.ptr)).*;
    // const packed_vec: RawVec = std.mem.bytesAsValue([16]u8, block_bytes[0..16]).*;

    const low_nibbles = packed_vec & @as(RawVec, @splat(0x0F));
    const high_nibbles = (packed_vec >> @as(RawVec, @splat(4))) & @as(RawVec, @splat(0x0F));

    const mask_indices = comptime generateInterleaveMask();
    const unpacked_u8: @Vector(32, u8) = @shuffle(u8, low_nibbles, high_nibbles, mask_indices);

    const lut_vals = [16]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 };
    const lut_vec: @Vector(16, f32) = lut_vals;

    var result_vec: @Vector(32, f32) = undefined;

    inline for (0..32) |idx| {
        const _idx = unpacked_u8[idx];
        if (_idx < 16) {
            result_vec[idx] = lut_vec[_idx];
        } else {
            result_vec[idx] = 0.0;
        }
    }

    const scale_vec: @Vector(32, f32) = @splat(scale);
    const final_vec = result_vec * scale_vec;

    const target_ptr: *[32]f32 = @ptrCast(target.ptr);
    target_ptr.* = final_vec;
}

fn dequantizeWorker(
    output_slice: []f32,
    raw_data_slice: []const u8,
    cpu_index: usize,
) void {
    try pinToCpu(cpu_index);
    // std.debug.print("cpu_index {}\n", .{cpu_index});
    const block_size_bytes = 20; // 4 bytes (scale) + 16 bytes (data)
    var idx: usize = 0;
    var out_idx: usize = 0;

    while (idx < raw_data_slice.len) : (idx += block_size_bytes) {
        // std.debug.print("cpu_index {}: {}\n", .{ cpu_index, idx });

        if (idx + block_size_bytes > raw_data_slice.len) {
            break;
        }

        const scale_bytes = raw_data_slice[idx .. idx + 4];
        const scale = std.mem.bytesAsValue(f32, scale_bytes[0..4]).*;

        const block_fp4 = raw_data_slice[idx + 4 .. idx + 20];

        dequantizeBlock(output_slice[out_idx .. out_idx + 32], block_fp4, scale);

        out_idx += 32;
    }
}

// https://github.com/neudinger/zerauth/blob/main/lattices/lattice_zkp.cpp#L213
fn matVecMul(
    output: []f32,
    input: []const f32,
    weights: []const f32,
    rows: usize,
    cols: usize,
) void {
    // https://ziglang.org/documentation/0.15.2/
    @setFloatMode(std.builtin.FloatMode.optimized);
    @setRuntimeSafety(false);
    // https://github.com/neudinger/zerauth/blob/main/lattices/blake3.hpp#L318
    // https://github.com/neudinger/zerauth/blob/main/lattices/lattice_zkp.cpp#L238
    const VecWidth = std.simd.suggestVectorLength(f32) orelse 16;
    // std.debug.print("VecWidth: {d}\n", .{VecWidth});
    const BlockRows = 4; // Process 4 rows at a time (Tiling)

    var row: usize = 0;

    // Blocked Loop: Process 4 rows per iteration
    // https: //github.com/neudinger/zerauth/blob/main/lattices/lattice_zkp.cpp#L246
    while (row + BlockRows <= rows) : (row += BlockRows) {
        var sum0: @Vector(VecWidth, f32) = @splat(0.0);
        var sum1: @Vector(VecWidth, f32) = @splat(0.0);
        var sum2: @Vector(VecWidth, f32) = @splat(0.0);
        var sum3: @Vector(VecWidth, f32) = @splat(0.0);

        const off0 = (row + 0) * cols;
        const off1 = (row + 1) * cols;
        const off2 = (row + 2) * cols;
        const off3 = (row + 3) * cols;

        var col: usize = 0;

        // https: //github.com/neudinger/zerauth/blob/main/lattices/lattice_zkp.cpp#L241
        while (col + VecWidth <= cols) : (col += VecWidth) {
            const prefetch_dist = VecWidth * 4;
            if (col + prefetch_dist < cols) {
                @prefetch(&weights[off0 + col + prefetch_dist], .{ .rw = .read, .locality = 1, .cache = std.builtin.PrefetchOptions.Cache.data });
                @prefetch(&weights[off1 + col + prefetch_dist], .{ .rw = .read, .locality = 1, .cache = std.builtin.PrefetchOptions.Cache.data });
                @prefetch(&weights[off2 + col + prefetch_dist], .{ .rw = .read, .locality = 1, .cache = std.builtin.PrefetchOptions.Cache.data });
                @prefetch(&weights[off3 + col + prefetch_dist], .{ .rw = .read, .locality = 1, .cache = std.builtin.PrefetchOptions.Cache.data });
            }

            const x_vec: @Vector(VecWidth, f32) = @as(*const [VecWidth]f32, @ptrCast(&input[col])).*;

            const w0: @Vector(VecWidth, f32) = @as(*const [VecWidth]f32, @ptrCast(&weights[off0 + col])).*;
            const w1: @Vector(VecWidth, f32) = @as(*const [VecWidth]f32, @ptrCast(&weights[off1 + col])).*;
            const w2: @Vector(VecWidth, f32) = @as(*const [VecWidth]f32, @ptrCast(&weights[off2 + col])).*;
            const w3: @Vector(VecWidth, f32) = @as(*const [VecWidth]f32, @ptrCast(&weights[off3 + col])).*;

            sum0 += x_vec * w0;
            sum1 += x_vec * w1;
            sum2 += x_vec * w2;
            sum3 += x_vec * w3;
        }

        output[row + 0] = @reduce(.Add, sum0);
        output[row + 1] = @reduce(.Add, sum1);
        output[row + 2] = @reduce(.Add, sum2);
        output[row + 3] = @reduce(.Add, sum3);

        // https://github.com/neudinger/equadiffMPI/blob/main/cartesian-split/cart_main.cc#L225
        if (col < cols) {
            while (col < cols) : (col += 1) {
                const x_val = input[col];
                output[row + 0] += x_val * weights[off0 + col];
                output[row + 1] += x_val * weights[off1 + col];
                output[row + 2] += x_val * weights[off2 + col];
                output[row + 3] += x_val * weights[off3 + col];
            }
        }
    }

    while (row < rows) : (row += 1) {
        const row_offset = row * cols;
        var sum_vec: @Vector(VecWidth, f32) = @splat(0.0);
        var col: usize = 0;
        while (col + VecWidth <= cols) : (col += VecWidth) {
            const x: @Vector(VecWidth, f32) = @as(*const [VecWidth]f32, @ptrCast(&input[col])).*;
            const w: @Vector(VecWidth, f32) = @as(*const [VecWidth]f32, @ptrCast(&weights[row_offset + col])).*;
            sum_vec += x * w;
        }
        var summed = @reduce(.Add, sum_vec);
        while (col < cols) : (col += 1) {
            summed += input[col] * weights[row_offset + col];
        }
        output[row] = summed;
    }
}

const SafeTensorsFile = struct {
    allocator: std.mem.Allocator,
    file: std.fs.File,
    mapped_memory: []align(std.heap.pageSize()) u8,
    header_json: std.json.Parsed(std.json.Value),

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !SafeTensorsFile {
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        errdefer file.close();

        const stat = try file.stat();
        const file_size = stat.size;
        if (file_size == 0) {
            return error.EmptyFile;
        }

        const memory_mapping = try std.posix.mmap(
            null,
            file_size,
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(memory_mapping);

        _ = try std.posix.madvise(memory_mapping.ptr, memory_mapping.len, std.posix.MADV.SEQUENTIAL);
        _ = try std.posix.madvise(memory_mapping.ptr, memory_mapping.len, std.posix.MADV.WILLNEED);
        _ = try std.posix.madvise(memory_mapping.ptr, memory_mapping.len, std.posix.MADV.HUGEPAGE);

        if (memory_mapping.len < 8) {
            return error.InvalidFile;
        }

        const header_len_bytes = memory_mapping[0..8];
        const header_len = std.mem.readInt(u64, header_len_bytes[0..8], .little);

        if (memory_mapping.len < 8 + header_len) {
            return error.InvalidFile;
        }

        const json_slice: []const u8 = memory_mapping[8 .. 8 + header_len];

        const parsed_json = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            json_slice,
            .{ .ignore_unknown_fields = true },
        );
        // std.debug.print("{f} \n", .{std.json.fmt(parsed_json.value, .{ .whitespace = .indent_4 })});

        return SafeTensorsFile{
            .allocator = allocator,
            .file = file,
            .mapped_memory = memory_mapping,
            .header_json = parsed_json,
        };
    }

    pub fn deinit(self: *SafeTensorsFile) void {
        self.header_json.deinit();
        std.posix.munmap(self.mapped_memory);
        self.file.close();
    }

    pub fn printLayerNames(self: *const SafeTensorsFile) void {
        const root = self.header_json.value;
        if (root != .object) return;

        var iter = root.object.iterator();
        std.debug.print("--- Available Tensors ---\n", .{});
        var count: usize = 0;

        while (iter.next()) |entry| {
            if (count < 10) {
                std.debug.print("Found tensor: {s}\n", .{entry.key_ptr.*});
            }
            count += 1;
        }
        std.debug.print("... (Total {d} tensors)\n", .{count});
        std.debug.print("-------------------------\n", .{});
    }

    pub fn loadTensorParallel(self: *const SafeTensorsFile, name: []const u8) ![]f32 {
        const root = self.header_json.value;
        const tensor_obj = root.object.get(name) orelse return error.TensorNotFound;

        const offsets_json = tensor_obj.object.get("data_offsets").?;
        const start = offsets_json.array.items[0].integer;
        const end = offsets_json.array.items[1].integer;

        const header_len_bytes = self.mapped_memory[0..8];
        const header_len = std.mem.readInt(u64, header_len_bytes[0..8], .little);
        const data_base_offset = 8 + header_len;

        const abs_start = data_base_offset + @as(u64, @intCast(start));
        const abs_end = data_base_offset + @as(u64, @intCast(end));

        if (abs_end > self.mapped_memory.len) {
            return error.BufferOverrun;
        }

        const raw_data = self.mapped_memory[abs_start..abs_end];

        const block_size_bytes = 20;
        const total_blocks = raw_data.len / block_size_bytes;
        const num_elements = total_blocks * 32;

        const alignment = comptime std.mem.Alignment.fromByteUnits(4096);
        const output = try self.allocator.alignedAlloc(f32, alignment, num_elements);
        errdefer self.allocator.free(output);

        //  mlock
        const byte_len = output.len * @sizeOf(f32);
        _ = try std.posix.madvise(@ptrCast(output.ptr), byte_len, std.posix.MADV.HUGEPAGE);

        const cpu_count = try std.Thread.getCpuCount();
        const num_threads = @min(cpu_count, total_blocks);

        const threads = try self.allocator.alloc(std.Thread, num_threads);
        defer self.allocator.free(threads);

        const blocks_per_thread = total_blocks / num_threads;
        var start_block: usize = 0;

        // Spawn Threads
        // #pragma omp parallel for
        for (0..num_threads) |thread_idx| {
            const end_block = if (thread_idx == num_threads - 1) total_blocks else start_block + blocks_per_thread;

            const byte_start = start_block * block_size_bytes;
            const byte_end = end_block * block_size_bytes;

            const out_start = start_block * 32;
            const out_end = end_block * 32;

            const raw_slice = raw_data[byte_start..byte_end];
            const out_slice = output[out_start..out_end];

            threads[thread_idx] = try std.Thread.spawn(.{}, dequantizeWorker, .{
                out_slice,
                raw_slice,
                thread_idx,
            });

            start_block = end_block;
        }

        for (threads) |thread| {
            thread.join();
        }

        return output;
    }
};

pub fn loadTensor(allocator: std.mem.Allocator, path: []const u8) !void {
    var st_file = try SafeTensorsFile.init(allocator, path);
    defer st_file.deinit();

    // st_file.printLayerNames();

    std.debug.print("File mapped (size: {d} bytes). Header parsed.\n", .{st_file.mapped_memory.len});
    const tensor_name = "block.0.attn.qkv.weight";
    // const tensor_name = "model.layers.0.self_attn.q_proj.weight"; // Llama
    // const tensor_name = "transformer.h.0.attn.c_attn.weight"; // GPT

    const start_load = std.time.nanoTimestamp();

    if (st_file.loadTensorParallel(tensor_name)) |weights| {
        defer allocator.free(weights);

        const end_load = std.time.nanoTimestamp();
        const load_ms = @as(f64, @floatFromInt(end_load - start_load)) / 1_000_000.0;
        std.debug.print("Successfully dequantized tensor '{s}'. Size: {d} floats in {d:.2} ms.\n", .{ tensor_name, weights.len, load_ms });

        const hidden_size = std.heap.pageSize();
        if (weights.len < hidden_size) {
            std.debug.print("Tensor too small for testing hidden_size=4096\n", .{});
            return;
        }

        const rows = weights.len / hidden_size;
        const cols = hidden_size;

        // Create Dummy Input
        // const alignment = std.mem.Alignment.@"64";
        const alignment = comptime std.mem.Alignment.fromByteUnits(4096);
        const input = try allocator.alignedAlloc(f32, alignment, cols);
        defer allocator.free(input);
        // Add some dummy data
        @memset(input, 0.5);

        // Allocate Output
        const output = try allocator.alloc(f32, rows);
        defer allocator.free(output);

        // Run MatMul
        const start_time = std.time.nanoTimestamp();
        matVecMul(output, input, weights, rows, cols);
        const end_time = std.time.nanoTimestamp();

        const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

        std.debug.print("Result[0..8]: {any}\n", .{output[0..8]});
        std.debug.print("Time: {d:.4} ms\n", .{duration_ms});
    } else |_| {
        std.debug.print("Tensor '{s}' not found or invalid.\n", .{tensor_name});
    }
}
