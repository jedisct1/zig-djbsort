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
    fn floatSortKey(comptime SInt: type, s: SInt) SInt {
        const mask = s >> (@bitSizeOf(SInt) - 1);
        return s ^ (mask & comptime std.math.maxInt(SInt));
    }

    /// Applies floatSortKey to every element of a slice. Uses SIMD when available.
    /// Self-inverse: applying it twice restores the original values.
    fn applyFloatSortKey(comptime SInt: type, items: []SInt) void {
        const bits = @bitSizeOf(SInt);
        const n = items.len;
        const vec_len = comptime std.simd.suggestVectorLength(SInt) orelse 0;
        var j: usize = 0;
        if (vec_len > 0) {
            const max_val: @Vector(vec_len, SInt) = @splat(std.math.maxInt(SInt));
            while (j + vec_len <= n) : (j += vec_len) {
                const v: @Vector(vec_len, SInt) = items[j..][0..vec_len].*;
                items[j..][0..vec_len].* = v ^ (v >> @splat(bits - 1) & max_val);
            }
        }
        while (j < n) : (j += 1) {
            items[j] = floatSortKey(SInt, items[j]);
        }
    }

    /// Branchless constant-time compare-and-swap for integer types.
    /// On architectures with known constant-time min/max (x86, aarch64),
    /// uses `@min`/`@max` directly. Otherwise falls back to XOR-masked swap.
    /// Float types never reach here — they are converted to integers by sort().
    fn minmax(comptime T: type, comptime order: Order, a: *T, b: *T) void {
        if (has_ct_minmax) {
            const lo, const hi = .{ @min(a.*, b.*), @max(a.*, b.*) };
            a.* = if (order == .asc) lo else hi;
            b.* = if (order == .asc) hi else lo;
        } else {
            // Compute swap mask arithmetically — no comparison operators, no branches.
            // Widen to (bits+1)-bit signed integer to prevent overflow, subtract,
            // then extract the sign bit to build the XOR mask.
            const bits = @bitSizeOf(T);
            const WInt = std.meta.Int(.signed, bits + 1);
            const a_int: WInt = @intCast(a.*);
            const b_int: WInt = @intCast(b.*);

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
    /// Float types never reach here — they are converted to integers by sort().
    fn vecSortedPair(comptime T: type, comptime order: Order, comptime N: comptime_int, a: @Vector(N, T), b: @Vector(N, T)) struct { @Vector(N, T), @Vector(N, T) } {
        const lo, const hi = .{ @min(a, b), @max(a, b) };
        return if (order == .asc) .{ lo, hi } else .{ hi, lo };
    }

    fn cascade(comptime T: type, comptime order: Order, items: []T, j: usize, p: usize, q: usize) void {
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

        // "useint" technique from djbsort: transform float bits to sortable
        // integers, sort using the integer path, then transform back.
        if (@typeInfo(T) == .float) {
            const SInt = std.meta.Int(.signed, @bitSizeOf(T));
            const int_items: [*]SInt = @ptrCast(items.ptr);
            const int_slice = int_items[0..n];
            applyFloatSortKey(SInt, int_slice);
            sort(SInt, order, int_slice);
            applyFloatSortKey(SInt, int_slice);
            return;
        }

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
    fn ctCondSwap(comptime T: type, a: *T, b: *T, should_swap: bool) void {
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
    fn minmax(
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

    fn cascade(
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
