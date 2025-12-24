# Concat Communication Optimization Patterns

This document describes the concat-like communication optimization patterns added to address issue #[number].

## Overview

Two new optimization patterns have been added to `OptimizeCommunication.cpp` to improve communication efficiency for concat operations:

1. **ConcatSliceOptimize**: Optimizes concat of two slices from the same source
2. **ConcatLargestOperandOptimize**: Optimizes concat by padding the largest operand and using DUS for others

## Implementation Status

### ✅ Completed
- Pattern detection infrastructure
- Pass options (`concat_slice_optimize` and `concat_largest_operand`)
- Pattern registration in OptimizeCommunicationPass
- Test cases demonstrating expected usage
- Detailed documentation of transformation strategies

### 🚧 TODO (Future Work)
The actual transformations are not yet implemented. Each pattern includes detailed TODO comments explaining:
- The transformation steps required
- Which existing patterns to reference (RotateCommOptimize, ConcatTwoDUSLike, etc.)
- The expected benefits

## Pattern 1: ConcatSliceOptimize

### What it detects:
Concatenations of two slices from the same source tensor, specifically:
- **Rotation pattern**: `concat(slice(A, [N:end]), slice(A, [0:N]))`
- **Asymmetric pattern**: `concat(small_slice, large_slice)` where one is much smaller

### Example from issue:
```mlir
%21 = stablehlo.slice %15 [8:12, 6:6134, 12271:12272]  // Last element
%31 = stablehlo.slice %15 [8:12, 6:6134, 0:12272]      // All except last
%723 = stablehlo.concatenate %21, %31, dim = 2         // Rotated result
```

### Planned transformation:
1. Recognize this is equivalent to a rotation
2. Pad the source if needed
3. Use enzymexla::RotateOp or collective permute pattern
4. Replace concat with rotated result

### Benefits:
- Reduces to a single rotate operation
- More efficient for distributed/sharded computation
- Leverages existing rotate communication optimizations

## Pattern 2: ConcatLargestOperandOptimize

### What it detects:
Concatenations of 3+ operands where one operand is significantly larger than others (>= 50% of total size)

### Example from issue:
```mlir
%895 = stablehlo.slice %827 [0:4, 0:1, 0:12272]        // Small: 4x1x12272
%1190 = stablehlo.slice %893 [0:4, 0:1, 0:12272]       // Small: 4x1x12272
%1189 = stablehlo.add %1149, %1188                     // Large: 4x6124x12272
%894 = stablehlo.slice %893 [0:4, 6125:6126, 0:12272]  // Small: 4x1x12272
%896 = stablehlo.slice %827 [0:4, 6127:6128, 0:12272]  // Small: 4x1x12272
%1191 = stablehlo.concatenate %895, %1190, %1189, %894, %896, dim = 1
       // Result: 4x6128x12272
```

### Planned transformation:
1. Identify %1189 as the largest operand (6124/6128 ≈ 99.9%)
2. Pad %1189 to the full result size (4x6128x12272)
3. Use DUS to insert each small operand at its correct position
4. Replace concat with the DUS result

### Benefits:
- Reduces N-way concat to 1 pad + (N-1) DUS operations
- More efficient when largest operand dominates
- Better for communication in sharded scenarios
- Leverages existing DUS optimization patterns

## Usage

Enable these patterns via pass options:

```bash
enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{concat_slice_optimize=1})" input.mlir
enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{concat_largest_operand=1})" input.mlir
```

Default: Both disabled (0) until full implementation is complete.

## Testing

Test files are provided:
- `test/lit_tests/communication/concat_slice_optimize.mlir`
- `test/lit_tests/communication/concat_largest_operand.mlir`

Currently these tests verify pattern detection only. Once transformations are implemented, test expectations should be updated to check the generated code.

## Implementation Notes

### For Pattern 1 (ConcatSliceOptimize):
- Reference existing patterns: `RotateCommOptimize`, `PeriodicConcatSimplify`
- May need to handle edge cases with non-evenly-divisible sizes
- Should integrate with existing rotation optimization infrastructure

### For Pattern 2 (ConcatLargestOperandOptimize):
- Reference existing patterns: `ConcatTwoDUSLike`, `DUSToPadManualCompComm`
- Need to handle padding to make dimensions evenly divisible by shard count
- May need ManualComputationOp with multiDimensionalSelect for complex cases
- Should preserve sharding attributes correctly

## Related Work

This addresses patterns described in the GitHub issue showing real-world cases where concat operations could be optimized for better communication efficiency in distributed/sharded scenarios.
