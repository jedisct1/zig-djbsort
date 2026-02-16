const std = @import("std");
const djbsort = @import("djbsort.zig");

const native = djbsort.native;
const generic = djbsort.generic;
const Order = djbsort.Order;

test "empty array" {
    var x = [_]i64{};
    native.sort(i64, .asc, &x);
}

test "single element" {
    var x = [_]i64{42};
    native.sort(i64, .asc, &x);
    try std.testing.expectEqual(42, x[0]);
}

test "two elements already sorted" {
    var x = [_]i64{ 1, 2 };
    native.sort(i64, .asc, &x);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2 }, &x);
}

test "two elements reversed" {
    var x = [_]i64{ 2, 1 };
    native.sort(i64, .asc, &x);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2 }, &x);
}

test "small sorted" {
    var x = [_]i64{ 1, 2, 3, 4, 5 };
    native.sort(i64, .asc, &x);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 5 }, &x);
}

test "small reversed" {
    var x = [_]i64{ 5, 4, 3, 2, 1 };
    native.sort(i64, .asc, &x);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 5 }, &x);
}

test "duplicates" {
    var x = [_]i64{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    native.sort(i64, .asc, &x);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 2, 3, 3, 4, 5, 5, 6, 9 }, &x);
}

test "negative numbers" {
    var x = [_]i64{ -3, -1, -4, -1, -5, -9, -2, -6, -5, -3 };
    native.sort(i64, .asc, &x);
    try std.testing.expectEqualSlices(i64, &.{ -9, -6, -5, -5, -4, -3, -3, -2, -1, -1 }, &x);
}

test "mixed positive and negative" {
    var x = [_]i64{ 3, -1, 4, -1, 5, -9, 2, -6 };
    native.sort(i64, .asc, &x);
    try std.testing.expectEqualSlices(i64, &.{ -9, -6, -1, -1, 2, 3, 4, 5 }, &x);
}

test "all same elements" {
    var x = [_]i64{ 7, 7, 7, 7, 7 };
    native.sort(i64, .asc, &x);
    try std.testing.expectEqualSlices(i64, &.{ 7, 7, 7, 7, 7 }, &x);
}

test "i64 extremes" {
    var x = [_]i64{ std.math.maxInt(i64), std.math.minInt(i64), 0, -1, 1 };
    native.sort(i64, .asc, &x);
    try std.testing.expectEqualSlices(i64, &.{ std.math.minInt(i64), -1, 0, 1, std.math.maxInt(i64) }, &x);
}

test "u8 basic" {
    var x = [_]u8{ 255, 0, 128, 1, 127 };
    native.sort(u8, .asc, &x);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 127, 128, 255 }, &x);
}

test "u32 basic" {
    var x = [_]u32{ 1000, 0, 500, std.math.maxInt(u32), 1 };
    native.sort(u32, .asc, &x);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 500, 1000, std.math.maxInt(u32) }, &x);
}

test "i16 basic" {
    var x = [_]i16{ 100, -100, 0, std.math.maxInt(i16), std.math.minInt(i16) };
    native.sort(i16, .asc, &x);
    try std.testing.expectEqualSlices(i16, &.{ std.math.minInt(i16), -100, 0, 100, std.math.maxInt(i16) }, &x);
}

test "f64 basic" {
    var x = [_]f64{ 3.14, -2.71, 0.0, 1.0, -1.0 };
    native.sort(f64, .asc, &x);
    try std.testing.expectEqualSlices(f64, &.{ -2.71, -1.0, 0.0, 1.0, 3.14 }, &x);
}

test "f64 special values" {
    const inf = std.math.inf(f64);
    var x = [_]f64{ inf, -inf, 0.0, -0.0, 1.0 };
    native.sort(f64, .asc, &x);
    // -inf < -0.0 < +0.0 < 1.0 < inf
    try std.testing.expect(x[0] == -inf);
    try std.testing.expect(x[1] == -0.0 and std.math.isNegativeZero(x[1]));
    try std.testing.expect(x[2] == 0.0 and !std.math.isNegativeZero(x[2]));
    try std.testing.expect(x[3] == 1.0);
    try std.testing.expect(x[4] == inf);
}

test "f32 basic" {
    var x = [_]f32{ 5.0, -3.0, 0.0, 2.5, -1.5 };
    native.sort(f32, .asc, &x);
    try std.testing.expectEqualSlices(f32, &.{ -3.0, -1.5, 0.0, 2.5, 5.0 }, &x);
}

test "desc: small reversed becomes ascending" {
    var x = [_]i64{ 1, 2, 3, 4, 5 };
    native.sort(i64, .desc, &x);
    try std.testing.expectEqualSlices(i64, &.{ 5, 4, 3, 2, 1 }, &x);
}

test "desc: mixed positive and negative" {
    var x = [_]i64{ 3, -1, 4, -1, 5, -9, 2, -6 };
    native.sort(i64, .desc, &x);
    try std.testing.expectEqualSlices(i64, &.{ 5, 4, 3, 2, -1, -1, -6, -9 }, &x);
}

test "desc: u8 basic" {
    var x = [_]u8{ 255, 0, 128, 1, 127 };
    native.sort(u8, .desc, &x);
    try std.testing.expectEqualSlices(u8, &.{ 255, 128, 127, 1, 0 }, &x);
}

test "desc: f64 basic" {
    var x = [_]f64{ 3.14, -2.71, 0.0, 1.0, -1.0 };
    native.sort(f64, .desc, &x);
    try std.testing.expectEqualSlices(f64, &.{ 3.14, 1.0, 0.0, -1.0, -2.71 }, &x);
}

test "desc: f64 special values" {
    const inf = std.math.inf(f64);
    var x = [_]f64{ inf, -inf, 0.0, -0.0, 1.0 };
    native.sort(f64, .desc, &x);
    // inf > 1.0 > +0.0 > -0.0 > -inf
    try std.testing.expect(x[0] == inf);
    try std.testing.expect(x[1] == 1.0);
    try std.testing.expect(x[2] == 0.0 and !std.math.isNegativeZero(x[2]));
    try std.testing.expect(x[3] == -0.0 and std.math.isNegativeZero(x[3]));
    try std.testing.expect(x[4] == -inf);
}

fn testNativeAgainstStdSort(comptime T: type, comptime order: Order) !void {
    var prng = std.Random.DefaultPrng.init(0xdeadbeef);
    const random = prng.random();

    for ([_]usize{ 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 255, 256, 1000 }) |size| {
        const allocator = std.testing.allocator;
        const x = try allocator.alloc(T, size);
        defer allocator.free(x);
        const reference = try allocator.alloc(T, size);
        defer allocator.free(reference);

        for (x, reference) |*xp, *rp| {
            const val: T = if (@typeInfo(T) == .float) blk: {
                const i = random.intRangeLessThan(i32, -1000, 1000);
                break :blk @as(T, @floatFromInt(i)) + random.float(T);
            } else random.int(T);
            xp.* = val;
            rp.* = val;
        }

        native.sort(T, order, x);
        const cmp = if (order == .asc) std.sort.asc(T) else std.sort.desc(T);
        std.sort.pdq(T, reference, {}, cmp);

        try std.testing.expectEqualSlices(T, reference, x);
    }
}

test "i64 various sizes against std.sort asc" {
    try testNativeAgainstStdSort(i64, .asc);
}

test "u64 various sizes against std.sort asc" {
    try testNativeAgainstStdSort(u64, .asc);
}

test "i32 various sizes against std.sort asc" {
    try testNativeAgainstStdSort(i32, .asc);
}

test "u8 various sizes against std.sort asc" {
    try testNativeAgainstStdSort(u8, .asc);
}

test "f64 various sizes against std.sort asc" {
    try testNativeAgainstStdSort(f64, .asc);
}

test "f32 various sizes against std.sort asc" {
    try testNativeAgainstStdSort(f32, .asc);
}

test "i64 various sizes against std.sort desc" {
    try testNativeAgainstStdSort(i64, .desc);
}

test "u64 various sizes against std.sort desc" {
    try testNativeAgainstStdSort(u64, .desc);
}

test "i32 various sizes against std.sort desc" {
    try testNativeAgainstStdSort(i32, .desc);
}

test "u8 various sizes against std.sort desc" {
    try testNativeAgainstStdSort(u8, .desc);
}

test "f64 various sizes against std.sort desc" {
    try testNativeAgainstStdSort(f64, .desc);
}

test "f32 various sizes against std.sort desc" {
    try testNativeAgainstStdSort(f32, .desc);
}

fn testGenericAgainstStdSort(comptime T: type, comptime order: Order) !void {
    var prng = std.Random.DefaultPrng.init(0xdeadbeef);
    const random = prng.random();

    const lessThanFn = if (order == .asc) std.sort.asc(T) else std.sort.desc(T);

    for ([_]usize{ 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 255, 256, 1000 }) |size| {
        const allocator = std.testing.allocator;
        const x = try allocator.alloc(T, size);
        defer allocator.free(x);
        const reference = try allocator.alloc(T, size);
        defer allocator.free(reference);

        for (x, reference) |*xp, *rp| {
            const val: T = if (@typeInfo(T) == .float) blk: {
                const i = random.intRangeLessThan(i32, -1000, 1000);
                break :blk @as(T, @floatFromInt(i)) + random.float(T);
            } else random.int(T);
            xp.* = val;
            rp.* = val;
        }

        generic.sort(T, x, {}, lessThanFn);
        std.sort.pdq(T, reference, {}, lessThanFn);

        try std.testing.expectEqualSlices(T, reference, x);
    }
}

test "generic i64 various sizes asc" {
    try testGenericAgainstStdSort(i64, .asc);
}

test "generic u64 various sizes asc" {
    try testGenericAgainstStdSort(u64, .asc);
}

test "generic i32 various sizes asc" {
    try testGenericAgainstStdSort(i32, .asc);
}

test "generic u8 various sizes asc" {
    try testGenericAgainstStdSort(u8, .asc);
}

test "generic f64 various sizes asc" {
    try testGenericAgainstStdSort(f64, .asc);
}

test "generic f32 various sizes asc" {
    try testGenericAgainstStdSort(f32, .asc);
}

test "generic i64 various sizes desc" {
    try testGenericAgainstStdSort(i64, .desc);
}

test "generic u64 various sizes desc" {
    try testGenericAgainstStdSort(u64, .desc);
}

test "generic i32 various sizes desc" {
    try testGenericAgainstStdSort(i32, .desc);
}

test "generic u8 various sizes desc" {
    try testGenericAgainstStdSort(u8, .desc);
}

test "generic f64 various sizes desc" {
    try testGenericAgainstStdSort(f64, .desc);
}

test "generic f32 various sizes desc" {
    try testGenericAgainstStdSort(f32, .desc);
}

test "generic sort: struct by single field" {
    const Point = struct { x: i32, y: i32 };
    var pts = [_]Point{
        .{ .x = 3, .y = 10 },
        .{ .x = 1, .y = 20 },
        .{ .x = 4, .y = 30 },
        .{ .x = 1, .y = 5 },
        .{ .x = 2, .y = 15 },
    };

    generic.sort(Point, &pts, {}, struct {
        fn lessThan(_: void, a: Point, b: Point) bool {
            return a.x < b.x;
        }
    }.lessThan);

    for (0..pts.len - 1) |i| {
        try std.testing.expect(pts[i].x <= pts[i + 1].x);
    }
}

test "generic sort: struct lexicographic" {
    const Point = struct { x: i32, y: i32 };
    var pts = [_]Point{
        .{ .x = 2, .y = 3 },
        .{ .x = 1, .y = 2 },
        .{ .x = 2, .y = 1 },
        .{ .x = 1, .y = 4 },
        .{ .x = 3, .y = 0 },
    };

    generic.sort(Point, &pts, {}, struct {
        fn lessThan(_: void, a: Point, b: Point) bool {
            if (a.x != b.x) return a.x < b.x;
            return a.y < b.y;
        }
    }.lessThan);

    try std.testing.expectEqual(Point{ .x = 1, .y = 2 }, pts[0]);
    try std.testing.expectEqual(Point{ .x = 1, .y = 4 }, pts[1]);
    try std.testing.expectEqual(Point{ .x = 2, .y = 1 }, pts[2]);
    try std.testing.expectEqual(Point{ .x = 2, .y = 3 }, pts[3]);
    try std.testing.expectEqual(Point{ .x = 3, .y = 0 }, pts[4]);
}

test "generic sort: struct descending" {
    const Item = struct { key: u32 };
    var items = [_]Item{
        .{ .key = 10 },
        .{ .key = 50 },
        .{ .key = 20 },
        .{ .key = 40 },
        .{ .key = 30 },
    };

    generic.sort(Item, &items, {}, struct {
        fn lessThan(_: void, a: Item, b: Item) bool {
            return a.key > b.key;
        }
    }.lessThan);

    try std.testing.expectEqual(@as(u32, 50), items[0].key);
    try std.testing.expectEqual(@as(u32, 40), items[1].key);
    try std.testing.expectEqual(@as(u32, 30), items[2].key);
    try std.testing.expectEqual(@as(u32, 20), items[3].key);
    try std.testing.expectEqual(@as(u32, 10), items[4].key);
}

test "generic sort: struct with context" {
    const Entry = struct { val: i32 };
    var entries = [_]Entry{
        .{ .val = 5 },
        .{ .val = -3 },
        .{ .val = 7 },
        .{ .val = -1 },
        .{ .val = 0 },
    };

    // Sort by absolute value, using a threshold from context
    const Ctx = struct {
        threshold: i32,
        fn lessThan(ctx: @This(), a: Entry, b: Entry) bool {
            const abs_a = @as(i32, if (a.val < 0) -a.val else a.val);
            const abs_b = @as(i32, if (b.val < 0) -b.val else b.val);
            // Values below threshold sort first, then by absolute value
            const a_below = @intFromBool(abs_a < ctx.threshold);
            const b_below = @intFromBool(abs_b < ctx.threshold);
            if (a_below != b_below) return a_below > b_below;
            return abs_a < abs_b;
        }
    };

    generic.sort(Entry, &entries, Ctx{ .threshold = 4 }, Ctx.lessThan);

    // Below threshold (|v| < 4): -1(1), 0(0), -3(3) sorted by abs → 0, -1, -3
    // At/above threshold: 5(5), 7(7) sorted by abs → 5, 7
    try std.testing.expectEqual(@as(i32, 0), entries[0].val);
    try std.testing.expectEqual(@as(i32, -1), entries[1].val);
    try std.testing.expectEqual(@as(i32, -3), entries[2].val);
    try std.testing.expectEqual(@as(i32, 5), entries[3].val);
    try std.testing.expectEqual(@as(i32, 7), entries[4].val);
}

test "generic sort: empty and single element" {
    const S = struct { v: u8 };
    var empty = [_]S{};
    generic.sort(S, &empty, {}, struct {
        fn f(_: void, a: S, b: S) bool {
            return a.v < b.v;
        }
    }.f);

    var single = [_]S{.{ .v = 42 }};
    generic.sort(S, &single, {}, struct {
        fn f(_: void, a: S, b: S) bool {
            return a.v < b.v;
        }
    }.f);
    try std.testing.expectEqual(@as(u8, 42), single[0].v);
}

test "generic sort: large struct" {
    const Big = struct {
        key: i64,
        payload: [56]u8, // 64 bytes total
    };

    var items: [20]Big = undefined;
    var prng = std.Random.DefaultPrng.init(0xabcd);
    const random = prng.random();
    for (&items) |*item| {
        item.key = random.intRangeLessThan(i64, -100, 100);
        random.bytes(&item.payload);
    }

    var original_keys: [20]i64 = undefined;
    var original_payloads: [20][56]u8 = undefined;
    for (items, 0..) |item, i| {
        original_keys[i] = item.key;
        original_payloads[i] = item.payload;
    }

    generic.sort(Big, &items, {}, struct {
        fn lessThan(_: void, a: Big, b: Big) bool {
            return a.key < b.key;
        }
    }.lessThan);

    for (0..items.len - 1) |i| {
        try std.testing.expect(items[i].key <= items[i + 1].key);
    }

    for (items) |item| {
        var found = false;
        for (original_keys, original_payloads) |ok, op| {
            if (ok == item.key and std.mem.eql(u8, &op, &item.payload)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
}
