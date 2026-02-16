const std = @import("std");
const Io = std.Io;
const djbsort = @import("djbsort");

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout = &stdout_file_writer.interface;

    var x = [_]i64{ 42, -7, 13, 0, -99, 55, 3, 21, -13, 8 };

    try stdout.print("before: ", .{});
    for (x) |v| try stdout.print("{d} ", .{v});
    try stdout.print("\n", .{});

    djbsort.sort(i64, .asc, &x);

    try stdout.print("after:  ", .{});
    for (x) |v| try stdout.print("{d} ", .{v});
    try stdout.print("\n", .{});

    try stdout.flush();
}
