const std = @import("std");
const dequantizer_vulkan = @import("dequantizer_vulkan");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len > 1 and std.mem.eql(u8, args[1], "saxpy")) {
        std.debug.print("Running Saxpy App...\n", .{});
        var app = try dequantizer_vulkan.saxpy.SaxpyApp.init(allocator);
        defer app.deinit();
        try app.run();
        try app.verify();
    } else {
        std.debug.print("Running MatVec App...\n", .{});
        var app = try dequantizer_vulkan.matvec.MatvecApp.init(allocator);
        defer app.deinit();
        try app.run();
        try app.verify();
    }

    std.debug.print("Done.\n", .{});
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa);
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
