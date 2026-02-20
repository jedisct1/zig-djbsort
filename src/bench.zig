const std = @import("std");
const Io = std.Io;
const ctsort = @import("ctsort");

const WARMUP_ITERS = 5;
const BENCH_ITERS = 50;

fn median(times: []i96) i96 {
    const S = struct {
        fn lessThan(_: void, a: i96, b: i96) bool {
            return a < b;
        }
    };
    std.sort.pdq(i96, times, {}, S.lessThan);
    return times[times.len / 2];
}

fn benchSort(
    comptime T: type,
    allocator: std.mem.Allocator,
    n: usize,
    io: Io,
    writer: anytype,
) !void {
    var prng = std.Random.DefaultPrng.init(0xcafebabe);
    const random = prng.random();

    const original = try allocator.alloc(T, n);
    defer allocator.free(original);
    for (original) |*p| {
        p.* = if (@typeInfo(T) == .float)
            @as(T, @floatFromInt(random.intRangeLessThan(i32, -10000, 10000))) + random.float(T)
        else
            random.int(T);
    }

    const work = try allocator.alloc(T, n);
    defer allocator.free(work);

    var ct_times: [BENCH_ITERS]i96 = undefined;
    var pdq_times: [BENCH_ITERS]i96 = undefined;

    for (0..WARMUP_ITERS) |_| {
        @memcpy(work, original);
        ctsort.sort(T, .asc, work);
        std.mem.doNotOptimizeAway(work);
    }

    for (&ct_times) |*t| {
        @memcpy(work, original);
        const start = Io.Timestamp.now(io, .awake);
        ctsort.sort(T, .asc, work);
        std.mem.doNotOptimizeAway(work);
        const end = Io.Timestamp.now(io, .awake);
        t.* = end.nanoseconds - start.nanoseconds;
    }

    const cmp = std.sort.asc(T);
    for (0..WARMUP_ITERS) |_| {
        @memcpy(work, original);
        std.sort.pdq(T, work, {}, cmp);
        std.mem.doNotOptimizeAway(work);
    }

    for (&pdq_times) |*t| {
        @memcpy(work, original);
        const start = Io.Timestamp.now(io, .awake);
        std.sort.pdq(T, work, {}, cmp);
        std.mem.doNotOptimizeAway(work);
        const end = Io.Timestamp.now(io, .awake);
        t.* = end.nanoseconds - start.nanoseconds;
    }

    const ct_ns = median(&ct_times);
    const pdq_ns = median(&pdq_times);
    const ct_f: f64 = @floatFromInt(ct_ns);
    const pdq_f: f64 = @floatFromInt(pdq_ns);
    const ratio = if (pdq_f > 0) ct_f / pdq_f else 0.0;

    try writer.print("{s:<6} n={d:<8} ct={d:>10} ns  pdq={d:>10} ns  ratio={d:.2}x\n", .{
        @typeName(T),
        n,
        @as(i64, @intCast(ct_ns)),
        @as(i64, @intCast(pdq_ns)),
        ratio,
    });
}

fn benchSortBy(
    comptime T: type,
    allocator: std.mem.Allocator,
    n: usize,
    io: Io,
    writer: anytype,
) !void {
    var prng = std.Random.DefaultPrng.init(0xcafebabe);
    const random = prng.random();

    const original = try allocator.alloc(T, n);
    defer allocator.free(original);
    for (original) |*p| {
        p.* = if (@typeInfo(T) == .float)
            @as(T, @floatFromInt(random.intRangeLessThan(i32, -10000, 10000))) + random.float(T)
        else
            random.int(T);
    }

    const work = try allocator.alloc(T, n);
    defer allocator.free(work);

    const cmp = std.sort.asc(T);

    var ct_times: [BENCH_ITERS]i96 = undefined;
    var pdq_times: [BENCH_ITERS]i96 = undefined;

    for (0..WARMUP_ITERS) |_| {
        @memcpy(work, original);
        ctsort.sortWith(T, work, {}, cmp);
        std.mem.doNotOptimizeAway(work);
    }

    for (&ct_times) |*t| {
        @memcpy(work, original);
        const start = Io.Timestamp.now(io, .awake);
        ctsort.sortWith(T, work, {}, cmp);
        std.mem.doNotOptimizeAway(work);
        const end = Io.Timestamp.now(io, .awake);
        t.* = end.nanoseconds - start.nanoseconds;
    }

    for (0..WARMUP_ITERS) |_| {
        @memcpy(work, original);
        std.sort.pdq(T, work, {}, cmp);
        std.mem.doNotOptimizeAway(work);
    }

    for (&pdq_times) |*t| {
        @memcpy(work, original);
        const start = Io.Timestamp.now(io, .awake);
        std.sort.pdq(T, work, {}, cmp);
        std.mem.doNotOptimizeAway(work);
        const end = Io.Timestamp.now(io, .awake);
        t.* = end.nanoseconds - start.nanoseconds;
    }

    const ct_ns = median(&ct_times);
    const pdq_ns = median(&pdq_times);
    const ct_f: f64 = @floatFromInt(ct_ns);
    const pdq_f: f64 = @floatFromInt(pdq_ns);
    const ratio = if (pdq_f > 0) ct_f / pdq_f else 0.0;

    try writer.print("{s:<6} n={d:<8} ctBy={d:>10} ns  pdq={d:>10} ns  ratio={d:.2}x\n", .{
        @typeName(T),
        n,
        @as(i64, @intCast(ct_ns)),
        @as(i64, @intCast(pdq_ns)),
        ratio,
    });
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const writer = &stdout_file_writer.interface;

    try writer.print("\n* ctsort vs pdqsort benchmark *\n", .{});
    try writer.print("Warmup: {d} iters, Bench: {d} iters (median)\n\n", .{ WARMUP_ITERS, BENCH_ITERS });

    const sizes = [_]usize{ 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576 };

    try writer.print("--- sort (SIMD) ---\n", .{});
    inline for ([_]type{ i64, u64, i32, f64, f32 }) |T| {
        for (sizes) |n| try benchSort(T, allocator, n, io, writer);
        try writer.print("\n", .{});
    }

    try writer.print("--- sortBy (generic) ---\n", .{});
    inline for ([_]type{ i64, u64, i32, f64, f32 }) |T| {
        for (sizes) |n| try benchSortBy(T, allocator, n, io, writer);
        try writer.print("\n", .{});
    }

    try writer.flush();
}
