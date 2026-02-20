# ctsort for Zig

A constant-time sorting network for Zig, based on Dan Bernstein's [djbsort](https://sorting.cr.yp.to/).

The sorting network is data-oblivious: the sequence of comparisons and swaps depends only on the array length, never on the values being sorted. This makes it immune to timing side-channels, which matters when you need to sort sensitive data in cryptographic contexts.

## Usage

The module exposes two functions.

`sort` is the fast path for native numeric types (integers and floats of any width).

It uses SIMD vectorization and branchless min/max internally:

```zig
const ctsort = @import("ctsort");

var data = [_]i32{ 42, -7, 13, 0, -99 };
ctsort.sort(i32, .asc, &data);
// data is now { -99, -7, 0, 13, 42 }
```

`sortWith` handles arbitrary types, including structs. It follows the same interface as `std.sort.pdq`: you provide a comparison function and an optional context. The sort is constant-time as long as your comparison function is:

```zig
const ctsort = @import("ctsort");

const Point = struct { x: i32, y: i32 };

var points = [_]Point{ .{ .x = 3, .y = 1 }, .{ .x = 1, .y = 2 } };
ctsort.sortWith(Point, &points, {}, struct {
    fn lessThan(_: void, a: Point, b: Point) bool {
        return a.x < b.x;
    }
}.lessThan);
```

### Performance

Compared to `std.sort.pdq` (ratio < 1 means ctsort is faster):

`sort` (SIMD) is consistently faster across all sizes and types tested. Floats use the "useint" technique (bulk-transform to sortable integers, sort, transform back) so they run at the same speed as their integer counterparts.

On AMD Zen4, `sort` is 4-9x faster for small arrays (n <= 16K) and stays 1.3-1.5x faster at 1M elements. Floats and integers of the same width produce nearly identical timings.

On Apple Silicon, `sort` is 2.5-5x faster for small-to-mid sizes and 1.8-4x faster at 1M elements. The 32-bit types (i32, f32) benefit the most at large sizes due to double the SIMD lane count.

`sortWith` (generic) uses no SIMD but still beats `pdq` up to ~65K elements (2-7x faster for small arrays). Beyond that the O(n log^2 n) network scaling catches up, and at 1M elements it's roughly 2x slower than `pdq` on Zen4 (closer to 1.2x on Apple Silicon).

Run `zig build bench` to reproduce.

### Float ordering

For floating-point types, `sort` imposes a total order: `-NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN`.

This differs from IEEE 754 where `NaN` is unordered.
