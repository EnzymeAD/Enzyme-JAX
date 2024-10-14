// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// TODO: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%arg0 : tensor<64x3xf32>, %7 : tensor<45x1xi32>) -> tensor<45x3xf32> {
    %8 = "stablehlo.gather"(%arg0, %7) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<64x3xf32>, tensor<45x1xi32>) -> tensor<45x3xf32>
    return %8 : tensor<45x3xf32>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<64x3xf32>, %arg1: tensor<64x3xf32>, %arg2: tensor<45x1xi32>, %arg3: tensor<45x1xi32>) -> (tensor<45x3xf32>, tensor<45x3xf32>) {
// FORWARD-NEXT:    %0 = "stablehlo.gather"(%arg1, %arg2) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<64x3xf32>, tensor<45x1xi32>) -> tensor<45x3xf32>
// FORWARD-NEXT:    %1 = "stablehlo.gather"(%arg0, %arg2) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<64x3xf32>, tensor<45x1xi32>) -> tensor<45x3xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<45x3xf32>, tensor<45x3xf32>
// FORWARD-NEXT:  }
