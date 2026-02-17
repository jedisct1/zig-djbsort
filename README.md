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

On Apple Silicon, `sort` runs 2x to 4x faster than `std.sort.pdq` on integers across array sizes from 16 to 1 million elements.

Floats are faster up to ~256K elements; at 1M f64 elements the `O(n log^2 n)` scaling starts to show.

`sortWith` (generic, no SIMD) beats `pdq` by 1.2x to 3x on integers up to 65K, then gradually converges at larger sizes.

Run `zig build bench` to reproduce.

### Float ordering

For floating-point types, `sort` imposes a total order: `-NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN`.

This differs from IEEE 754 where `NaN` is unordered.
