const djbsort = @import("djbsort.zig");

pub const Order = djbsort.Order;

/// Sorts native numeric types using a data-oblivious sorting network with
/// SIMD-optimized branchless compare-and-swap.
///
/// Based on DJB's sorting network (djbsort). The comparison sequence depends
/// only on the array length, not on the data, making this suitable for
/// cryptographic applications where timing side-channels must be avoided.
///
/// Supports all integer types (signed and unsigned) and floating-point types.
/// For floats, -0.0 sorts before +0.0 and NaN values sort to the extremes.
///
/// For sorting arbitrary types with a custom comparison function, use `sortWith`.
pub const sort = djbsort.native.sort;

/// Sorts a slice using a data-oblivious sorting network.
///
/// The comparison sequence depends only on the array length, not on the data.
/// The conditional swap is implemented in constant time via XOR masking on the
/// raw byte representation, so the sort is timing-safe as long as `lessThanFn`
/// is itself constant-time.
///
/// Matches the `std.sort.pdq` interface: accepts any type `T` with a
/// caller-supplied comparison function and optional context.
///
/// For native integer and floating-point types where maximum throughput is
/// needed, `sort` provides a SIMD-optimized path with identical network topology.
pub const sortWith = djbsort.generic.sort;
