const std = @import("std");
const builtin = @import("builtin");

pub const Order = enum { asc, desc };

const side_channels_mitigations = std.options.side_channels_mitigations;

/// SIMD-optimized path for native numeric types (int/float).
pub const native = struct {
    /// Whether we can use `@min`/`@max` for the compare-and-swap.
    /// With no side-channel mitigations, always true — we don't care about timing.
    /// Otherwise, only on architectures known to lower these to constant-time
    /// instructions (e.g. cmov on x86, csel on aarch64).
    const has_ct_minmax = side_channels_mitigations == .none or switch (builtin.cpu.arch) {
        .x86, .x86_64, .aarch64, .aarch64_be => true,
        else => false,
    };

    /// XOR-based transform mapping IEEE 754 bits to a signed-integer-comparable
    /// representation. Self-inverse: applying it twice yields the original value.
    /// Total order: -NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN.
    inline fn floatSortKey(comptime T: type, s: std.meta.Int(.signed, @bitSizeOf(T))) std.meta.Int(.signed, @bitSizeOf(T)) {
        const mask = s >> (@bitSizeOf(T) - 1);
        return s ^ (mask & comptime std.math.maxInt(std.meta.Int(.signed, @bitSizeOf(T))));
    }

    /// Branchless constant-time compare-and-swap for native numeric types.
    /// On architectures with known constant-time min/max (x86, aarch64),
    /// uses `@min`/`@max` directly. Otherwise falls back to XOR-masked swap.
    inline fn minmax(comptime T: type, comptime order: Order, a: *T, b: *T) void {
        if (has_ct_minmax) {
            const lo, const hi = if (@typeInfo(T) == .float) blk: {
                const SInt = std.meta.Int(.signed, @bitSizeOf(T));
                const sa = floatSortKey(T, @as(SInt, @bitCast(a.*)));
                const sb = floatSortKey(T, @as(SInt, @bitCast(b.*)));
                break :blk .{
                    @as(T, @bitCast(floatSortKey(T, @min(sa, sb)))),
                    @as(T, @bitCast(floatSortKey(T, @max(sa, sb)))),
                };
            } else .{ @min(a.*, b.*), @max(a.*, b.*) };
            a.* = if (order == .asc) lo else hi;
            b.* = if (order == .asc) hi else lo;
        } else {
            // Compute swap mask arithmetically — no comparison operators, no branches.
            // Widen to (bits+1)-bit signed integer to prevent overflow, subtract,
            // then extract the sign bit to build the XOR mask.
            const bits = @bitSizeOf(T);
            const WInt = std.meta.Int(.signed, bits + 1);
            const a_int: WInt, const b_int: WInt = if (@typeInfo(T) == .float) .{
                floatSortKey(T, @as(std.meta.Int(.signed, bits), @bitCast(a.*))),
                floatSortKey(T, @as(std.meta.Int(.signed, bits), @bitCast(b.*))),
            } else .{ @intCast(a.*), @intCast(b.*) };

            // diff < 0 : the pair is out of order and needs swapping.
            const diff = if (order == .asc) b_int - a_int else a_int - b_int;
            const UWInt = std.meta.Int(.unsigned, bits + 1);
            const sign_bit: u1 = @truncate(@as(UWInt, @bitCast(diff)) >> @intCast(bits));
            var mask_word = @as(usize, 0) -% @as(usize, sign_bit);
            mask_word = asm volatile (""
                : [mask] "=r" (-> usize),
                : [_] "0" (mask_word),
            );

            const a_bytes = std.mem.asBytes(a);
            const b_bytes = std.mem.asBytes(b);
            const len = @sizeOf(T);
            comptime var i = 0;
            inline for (.{ usize, u32, u16, u8 }) |W| {
                const w = @sizeOf(W);
                if (w <= @sizeOf(usize)) {
                    inline while (i + w <= len) : (i += w) {
                        const mask: W = @truncate(mask_word);
                        const aw: *align(1) W = @ptrCast(a_bytes[i..][0..w]);
                        const bw: *align(1) W = @ptrCast(b_bytes[i..][0..w]);
                        const d = aw.* ^ bw.*;
                        aw.* ^= d & mask;
                        bw.* ^= d & mask;
                    }
                }
            }
        }
    }

    /// Vectorized compare-and-swap using packed SIMD min/max.
    inline fn vecSortedPair(comptime T: type, comptime order: Order, comptime N: comptime_int, a: @Vector(N, T), b: @Vector(N, T)) struct { @Vector(N, T), @Vector(N, T) } {
        const lo, const hi = if (@typeInfo(T) == .float) blk: {
            const SInt = std.meta.Int(.signed, @bitSizeOf(T));
            const bits = @bitSizeOf(T);
            const sa: @Vector(N, SInt) = @bitCast(a);
            const sb: @Vector(N, SInt) = @bitCast(b);
            const max_val: @Vector(N, SInt) = @splat(std.math.maxInt(SInt));
            const sorted_a = sa ^ (sa >> @splat(bits - 1) & max_val);
            const sorted_b = sb ^ (sb >> @splat(bits - 1) & max_val);
            const lo_s = @min(sorted_a, sorted_b);
            const hi_s = @max(sorted_a, sorted_b);
            break :blk .{
                @as(@Vector(N, T), @bitCast(lo_s ^ (lo_s >> @splat(bits - 1) & max_val))),
                @as(@Vector(N, T), @bitCast(hi_s ^ (hi_s >> @splat(bits - 1) & max_val))),
            };
        } else .{ @min(a, b), @max(a, b) };
        return if (order == .asc) .{ lo, hi } else .{ hi, lo };
    }

    inline fn cascade(comptime T: type, comptime order: Order, items: []T, j: usize, p: usize, q: usize) void {
        var a: T = items[j + p];
        var r = q;
        while (r > p) : (r = r >> 1) minmax(T, order, &a, &items[j + r]);
        items[j + p] = a;
    }

    pub fn sort(comptime T: type, comptime order: Order, items: []T) void {
        if (@typeInfo(T) != .int and @typeInfo(T) != .float)
            @compileError("sort requires an integer or floating-point type, got " ++ @typeName(T));

        const n = items.len;
        if (n < 2) return;

        const vec_len = comptime std.simd.suggestVectorLength(T) orelse 0;

        var top: usize = 1;
        while (top < n - top) top += top;

        var p = top;
        while (p >= 1) : (p = p >> 1) {
            // Loop 1: main minmax pairs
            var i: usize = 0;
            while (i + 2 * p <= n) {
                var k: usize = 0;
                if (vec_len > 0) {
                    while (k + vec_len <= p) : (k += vec_len) {
                        const a_vec: @Vector(vec_len, T) = items[i + k ..][0..vec_len].*;
                        const b_vec: @Vector(vec_len, T) = items[i + k + p ..][0..vec_len].*;
                        const first, const second = vecSortedPair(T, order, vec_len, a_vec, b_vec);
                        items[i + k ..][0..vec_len].* = first;
                        items[i + k + p ..][0..vec_len].* = second;
                    }
                }
                while (k < p) : (k += 1) minmax(T, order, &items[i + k], &items[i + k + p]);
                i += 2 * p;
            }

            // Loop 2: residual minmax
            var j = i;
            if (vec_len > 0) {
                while (j + vec_len + p <= n) : (j += vec_len) {
                    const a_vec: @Vector(vec_len, T) = items[j..][0..vec_len].*;
                    const b_vec: @Vector(vec_len, T) = items[j + p ..][0..vec_len].*;
                    const first, const second = vecSortedPair(T, order, vec_len, a_vec, b_vec);
                    items[j..][0..vec_len].* = first;
                    items[j + p ..][0..vec_len].* = second;
                }
            }
            while (j + p < n) : (j += 1) minmax(T, order, &items[j], &items[j + p]);

            i = 0;
            j = 0;
            var q = top;
            q_loop: while (q > p) : (q = q >> 1) {
                // j != i section: kept scalar (early-exit logic)
                if (j != i) while (true) {
                    if (j + q == n) continue :q_loop;
                    cascade(T, order, items, j, p, q);
                    j += 1;
                    if (j == i + p) {
                        i += 2 * p;
                        break;
                    }
                };

                // Loop 3: cascade groups
                while (i + p + q <= n) {
                    var k: usize = 0;
                    if (vec_len > 0) {
                        while (k + vec_len <= p) : (k += vec_len) {
                            var a_vec: @Vector(vec_len, T) = items[i + k + p ..][0..vec_len].*;
                            var r = q;
                            while (r > p) : (r >>= 1) {
                                const c_vec: @Vector(vec_len, T) = items[i + k + r ..][0..vec_len].*;
                                const acc, const comp = vecSortedPair(T, order, vec_len, a_vec, c_vec);
                                a_vec = acc;
                                items[i + k + r ..][0..vec_len].* = comp;
                            }
                            items[i + k + p ..][0..vec_len].* = a_vec;
                        }
                    }
                    while (k < p) : (k += 1) cascade(T, order, items, i + k, p, q);
                    i += 2 * p;
                }

                // Loop 4: residual cascades
                j = i;
                if (vec_len > 0 and p >= vec_len) {
                    while (j + vec_len + q <= n) : (j += vec_len) {
                        var a_vec: @Vector(vec_len, T) = items[j + p ..][0..vec_len].*;
                        var r = q;
                        while (r > p) : (r >>= 1) {
                            const c_vec: @Vector(vec_len, T) = items[j + r ..][0..vec_len].*;
                            const acc, const comp = vecSortedPair(T, order, vec_len, a_vec, c_vec);
                            a_vec = acc;
                            items[j + r ..][0..vec_len].* = comp;
                        }
                        items[j + p ..][0..vec_len].* = a_vec;
                    }
                }
                while (j + q < n) : (j += 1) cascade(T, order, items, j, p, q);
            }
        }
    }
};

/// Generic path for any type. Matches the std.sort.pdq interface.
/// Constant-time as long as lessThanFn is constant-time.
pub const generic = struct {
    /// Constant-time conditional swap using XOR masking with an
    /// optimization barrier to prevent LLVM from converting it to a
    /// conditional branch. Processes data in the widest chunks that
    /// fit, cascading down through word sizes for any remainder.
    inline fn ctCondSwap(comptime T: type, a: *T, b: *T, should_swap: bool) void {
        const len = @sizeOf(T);
        var mask_word = @as(usize, 0) -% @intFromBool(should_swap);
        mask_word = asm volatile (""
            : [mask] "=r" (-> usize),
            : [_] "0" (mask_word),
        );

        const a_bytes = std.mem.asBytes(a);
        const b_bytes = std.mem.asBytes(b);

        comptime var i = 0;
        inline for (.{ usize, u32, u16, u8 }) |W| {
            const w = @sizeOf(W);
            if (w <= @sizeOf(usize)) {
                inline while (i + w <= len) : (i += w) {
                    const mask: W = @truncate(mask_word);
                    const aw: *align(1) W = @ptrCast(a_bytes[i..][0..w]);
                    const bw: *align(1) W = @ptrCast(b_bytes[i..][0..w]);
                    const diff = aw.* ^ bw.*;
                    aw.* ^= diff & mask;
                    bw.* ^= diff & mask;
                }
            }
        }
    }

    /// Generic compare-and-swap using constant-time conditional swap.
    inline fn minmax(
        comptime T: type,
        context: anytype,
        comptime lessThanFn: fn (context: @TypeOf(context), lhs: T, rhs: T) bool,
        a: *T,
        b: *T,
    ) void {
        const should_swap = lessThanFn(context, b.*, a.*);
        if (side_channels_mitigations == .none) {
            if (should_swap) std.mem.swap(T, a, b);
        } else {
            ctCondSwap(T, a, b, should_swap);
        }
    }

    inline fn cascade(
        comptime T: type,
        context: anytype,
        comptime lessThanFn: fn (context: @TypeOf(context), lhs: T, rhs: T) bool,
        items: []T,
        j: usize,
        p: usize,
        q: usize,
    ) void {
        var a: T = items[j + p];
        var r = q;
        while (r > p) : (r = r >> 1) minmax(T, context, lessThanFn, &a, &items[j + r]);
        items[j + p] = a;
    }

    pub fn sort(
        comptime T: type,
        items: []T,
        context: anytype,
        comptime lessThanFn: fn (context: @TypeOf(context), lhs: T, rhs: T) bool,
    ) void {
        const n = items.len;
        if (n < 2) return;

        var top: usize = 1;
        while (top < n - top) top += top;

        var p = top;
        while (p >= 1) : (p = p >> 1) {
            // Loop 1: main minmax pairs
            var i: usize = 0;
            while (i + 2 * p <= n) {
                var k: usize = 0;
                while (k < p) : (k += 1) minmax(T, context, lessThanFn, &items[i + k], &items[i + k + p]);
                i += 2 * p;
            }

            // Loop 2: residual minmax
            var j = i;
            while (j + p < n) : (j += 1) minmax(T, context, lessThanFn, &items[j], &items[j + p]);

            i = 0;
            j = 0;
            var q = top;
            q_loop: while (q > p) : (q = q >> 1) {
                if (j != i) while (true) {
                    if (j + q == n) continue :q_loop;
                    cascade(T, context, lessThanFn, items, j, p, q);
                    j += 1;
                    if (j == i + p) {
                        i += 2 * p;
                        break;
                    }
                };

                while (i + p + q <= n) {
                    var k: usize = 0;
                    while (k < p) : (k += 1) cascade(T, context, lessThanFn, items, i + k, p, q);
                    i += 2 * p;
                }

                j = i;
                while (j + q < n) : (j += 1) cascade(T, context, lessThanFn, items, j, p, q);
            }
        }
    }
};

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
