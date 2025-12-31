const std = @import("std");
const dequantizer = @import("dequantizer");

pub fn main() !void {
    // https://github.com/neudinger/equadiffMPI/blob/main/horizontal-split/mpi_pencil_main.cc#L272
    const allocator = std.heap.page_allocator;

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <model.safetensors>\n", .{args[0]});
        return;
    }

    const filename = args[1];
    try dequantizer.loadTensor(allocator, filename);
}