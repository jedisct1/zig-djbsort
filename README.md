# djbsort for Zig

A Zig implementation of DJB's constant-time sorting network ([djbsort](https://sorting.cr.yp.to/)).

The sorting network is data-oblivious: the sequence of comparisons and swaps depends only on the array length, never on the values being sorted. This makes it immune to timing side-channels, which matters when you need to sort sensitive data in cryptographic contexts.

## Usage

The module exposes two functions.

`sort` is the fast path for native numeric types (integers and floats of any width).

It uses SIMD vectorization and branchless min/max internally:

```zig
const djbsort = @import("djbsort");

var data = [_]i32{ 42, -7, 13, 0, -99 };
djbsort.sort(i32, .asc, &data);
// data is now { -99, -7, 0, 13, 42 }
```

`sortWith` handles arbitrary types, including structs. It follows the same interface as `std.sort.pdq`: you provide a comparison function and an optional context. The sort is constant-time as long as your comparison function is:

```zig
const djbsort = @import("djbsort");

const Point = struct { x: i32, y: i32 };

var points = [_]Point{ .{ .x = 3, .y = 1 }, .{ .x = 1, .y = 2 } };
djbsort.sortWith(Point, &points, {}, struct {
    fn lessThan(_: void, a: Point, b: Point) bool {
        return a.x < b.x;
    }
}.lessThan);
```

### Performance

Compared to `std.sort.pdq` (ratio < 1 means djbsort is faster):

`sort` (SIMD) is consistently faster across all sizes and types tested. On AMD Zen4 with 64-bit integers, it reaches 5-8x faster for small arrays (n <= 16K) and stays 1.4x faster even at 1M elements. 32-bit integers are similar. Floats carry extra overhead from the total-order key transform but still run 2-3x faster at small sizes and hold an edge up to 1M.

`sortWith` (generic) uses no SIMD but still beats `pdq` up to ~65K elements (2-7x faster for small arrays on integers). Beyond that the `O(n log^2 n)` network scaling catches up, and at 1M elements it's roughly 2x slower than `pdq`.

These numbers were measured on AMD Zen4. On Apple Silicon the margins are similar: `sort` is 2-4x faster on integers across the board, and `sortWith` wins up to ~65K before converging.

Run `zig build bench` to reproduce.

### Float ordering

For floating-point types, `sort` imposes a total order: `-NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN`.

This differs from IEEE 754 where `NaN` is unordered.
