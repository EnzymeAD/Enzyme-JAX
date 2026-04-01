// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Single reversed dimension, slice picks one element -> reverse eliminated.
func.func @slice_single_element_reversed_dim(%arg0: tensor<4x8xf32>) -> tensor<4x1xf32> {
  %0 = stablehlo.reverse %arg0, dims = [1] : tensor<4x8xf32>
  %1 = stablehlo.slice %0 [0:4, 6:7] : (tensor<4x8xf32>) -> tensor<4x1xf32>
  return %1 : tensor<4x1xf32>
}
// N=8, s=6 -> new_start=1, new_limit=2
// CHECK-LABEL: func.func @slice_single_element_reversed_dim
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:4, 1:2] : (tensor<4x8xf32>) -> tensor<4x1xf32>
// CHECK-NEXT:    return %0 : tensor<4x1xf32>
// CHECK-NEXT:  }

// Both dimensions reversed, slice picks one element on each -> both eliminated.
func.func @slice_both_dims_single_element(%arg0: tensor<4x8xf32>) -> tensor<1x1xf32> {
  %0 = stablehlo.reverse %arg0, dims = [0, 1] : tensor<4x8xf32>
  %1 = stablehlo.slice %0 [1:2, 6:7] : (tensor<4x8xf32>) -> tensor<1x1xf32>
  return %1 : tensor<1x1xf32>
}
// dim0: N=4, s=1 -> 2:3; dim1: N=8, s=6 -> 1:2
// CHECK-LABEL: func.func @slice_both_dims_single_element
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [2:3, 1:2] : (tensor<4x8xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:    return %0 : tensor<1x1xf32>
// CHECK-NEXT:  }

// Two reversed dimensions, only one picks a single element.
// The residual reverse (dim 1) is placed *before* the slice so that the
// original slice indices for dim 1 remain correct.
func.func @slice_partial_single_element(%arg0: tensor<4x8xf32>) -> tensor<1x4xf32> {
  %0 = stablehlo.reverse %arg0, dims = [0, 1] : tensor<4x8xf32>
  %1 = stablehlo.slice %0 [1:2, 2:6] : (tensor<4x8xf32>) -> tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}
// dim0: N=4, s=1 -> 2:3 (single, eliminated); dim1: 4 elements (multi, kept)
// residual reverse on dim1 applied to x first, then slice with original dim1 indices
// CHECK-LABEL: func.func @slice_partial_single_element
// CHECK-NEXT:    %0 = stablehlo.reverse %arg0, dims = [1] : tensor<4x8xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [2:3, 2:6] : (tensor<4x8xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:    return %1 : tensor<1x4xf32>
// CHECK-NEXT:  }

// Reversed dimension not touched by slice at single-element granularity -> no change.
func.func @slice_no_single_element(%arg0: tensor<4x8xf32>) -> tensor<4x4xf32> {
  %0 = stablehlo.reverse %arg0, dims = [1] : tensor<4x8xf32>
  %1 = stablehlo.slice %0 [0:4, 2:6] : (tensor<4x8xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
// dim1 picks 4 elements (not single) -> no transformation possible
// CHECK-LABEL: func.func @slice_no_single_element
// CHECK:         stablehlo.reverse
// CHECK:         stablehlo.slice

// Reverse with multiple uses and remaining dims -> no transformation (duplicating
// the reverse would not be beneficial).
func.func @slice_multi_use_no_transform(%arg0: tensor<4x8xf32>) -> (tensor<1x4xf32>, tensor<1x4xf32>) {
  %0 = stablehlo.reverse %arg0, dims = [0, 1] : tensor<4x8xf32>
  %1 = stablehlo.slice %0 [1:2, 2:6] : (tensor<4x8xf32>) -> tensor<1x4xf32>
  %2 = stablehlo.slice %0 [2:3, 2:6] : (tensor<4x8xf32>) -> tensor<1x4xf32>
  return %1, %2 : tensor<1x4xf32>, tensor<1x4xf32>
}
// dim0 is single-element for both slices, dim1 picks 4 elements -> remainingDims=[1]
// reverse has 2 uses -> not beneficial to duplicate, no transformation applied
// CHECK-LABEL: func.func @slice_multi_use_no_transform
// CHECK:         stablehlo.reverse
// CHECK:         stablehlo.slice
// CHECK:         stablehlo.slice
