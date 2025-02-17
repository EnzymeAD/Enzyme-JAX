// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE

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

// REVERSE:  func.func @main(%arg0: tensor<64x3xf32>, %arg1: tensor<45x1xi32>, %arg2: tensor<45x3xf32>) -> (tensor<64x3xf32>, tensor<45x1xi32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<64x3xf32>
// REVERSE-NEXT:    %c = stablehlo.constant dense<0> : tensor<45x1xi32>
// REVERSE-NEXT:    %0 = "stablehlo.scatter"(%cst, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// REVERSE-NEXT:        ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// REVERSE-NEXT:            %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
// REVERSE-NEXT:            stablehlo.return %1 : tensor<f32>
// REVERSE-NEXT:        }) : (tensor<64x3xf32>, tensor<45x1xi32>, tensor<45x3xf32>) -> tensor<64x3xf32>
// REVERSE-NEXT:    return %0, %c : tensor<64x3xf32>, tensor<45x1xi32>
// REVERSE-NEXT:  }
