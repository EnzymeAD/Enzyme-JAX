// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @unaryscatter(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<24xf32>
    %c = stablehlo.constant dense<1> : tensor<24x2xi64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf32>
    %0 = stablehlo.concatenate %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, dim = 0 : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<24xi64>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<6xi64>) -> tensor<6x4xi64>
    %2 = stablehlo.reshape %0 : (tensor<24xi64>) -> tensor<24x1xi64>
    %3 = stablehlo.reshape %1 : (tensor<6x4xi64>) -> tensor<24x1xi64>
    %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<24x1xi64>, tensor<24x1xi64>) -> tensor<24x2xi64>
    %5 = stablehlo.subtract %4, %c : tensor<24x2xi64>
    %6 = "stablehlo.scatter"(%cst_0, %5, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<1024x1024xf32>, tensor<24x2xi64>, tensor<24xf32>) -> tensor<1024x1024xf32>
    %7 = stablehlo.sine %6 : tensor<1024x1024xf32>
    %8 = stablehlo.transpose %7, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %8 : tensor<1024x1024xf32>
}

// CHECK: func.func @unaryscatter(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.909297406> : tensor<24xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf32>
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<24x2xi64>
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xi64>) -> tensor<6x4xi64>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<6xi64>) -> tensor<6x4xi64>
// CHECK-NEXT:     %2 = stablehlo.reshape %0 : (tensor<6x4xi64>) -> tensor<24x1xi64>
// CHECK-NEXT:     %3 = stablehlo.reshape %1 : (tensor<6x4xi64>) -> tensor<24x1xi64>
// CHECK-NEXT:     %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<24x1xi64>, tensor<24x1xi64>) -> tensor<24x2xi64>
// CHECK-NEXT:     %5 = stablehlo.subtract %4, %c : tensor<24x2xi64>
// CHECK-NEXT:     %6 = "stablehlo.scatter"(%cst_0, %5, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
// CHECK-NEXT:     ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK-NEXT:       stablehlo.return %arg4 : tensor<f32>
// CHECK-NEXT:     }) {enzymexla.guaranteed_symmetric = false} : (tensor<1024x1024xf32>, tensor<24x2xi64>, tensor<24xf32>) -> tensor<1024x1024xf32>
// CHECK-NEXT:     %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// CHECK-NEXT:     return %7 : tensor<1024x1024xf32>
// CHECK-NEXT: }

func.func @expscatter(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<24xf32>
    %c = stablehlo.constant dense<1> : tensor<24x2xi64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf32>
    %0 = stablehlo.concatenate %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, dim = 0 : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<24xi64>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<6xi64>) -> tensor<6x4xi64>
    %2 = stablehlo.reshape %0 : (tensor<24xi64>) -> tensor<24x1xi64>
    %3 = stablehlo.reshape %1 : (tensor<6x4xi64>) -> tensor<24x1xi64>
    %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<24x1xi64>, tensor<24x1xi64>) -> tensor<24x2xi64>
    %5 = stablehlo.subtract %4, %c : tensor<24x2xi64>
    %6 = "stablehlo.scatter"(%cst_0, %5, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<1024x1024xf32>, tensor<24x2xi64>, tensor<24xf32>) -> tensor<1024x1024xf32>
    %7 = stablehlo.exponential %6 : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %8 = stablehlo.transpose %7, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %8 : tensor<1024x1024xf32>
}

// CHECK: func.func @expscatter(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<7.3890562> : tensor<24xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<1024x1024xf32>
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<24x2xi64>
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xi64>) -> tensor<6x4xi64>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<6xi64>) -> tensor<6x4xi64>
// CHECK-NEXT:     %2 = stablehlo.reshape %0 : (tensor<6x4xi64>) -> tensor<24x1xi64>
// CHECK-NEXT:     %3 = stablehlo.reshape %1 : (tensor<6x4xi64>) -> tensor<24x1xi64>
// CHECK-NEXT:     %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<24x1xi64>, tensor<24x1xi64>) -> tensor<24x2xi64>
// CHECK-NEXT:     %5 = stablehlo.subtract %4, %c : tensor<24x2xi64>
// CHECK-NEXT:     %6 = "stablehlo.scatter"(%cst_0, %5, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
// CHECK-NEXT:     ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK-NEXT:       stablehlo.return %arg4 : tensor<f32>
// CHECK-NEXT:     }) {enzymexla.guaranteed_symmetric = false} : (tensor<1024x1024xf32>, tensor<24x2xi64>, tensor<24xf32>) -> tensor<1024x1024xf32>
// CHECK-NEXT:     %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// CHECK-NEXT:     return %7 : tensor<1024x1024xf32>
// CHECK-NEXT: }

func.func @convertscatter(%arg0: tensor<5x4xf32>, %arg1: tensor<5xui32>) -> tensor<5x4xf32> {
    %c = stablehlo.constant dense<[[4, 5], [4, 5], [4, 5], [4, 5], [4, 5]]> : tensor<5x2xi64>
    %c_0 = stablehlo.constant dense<[-1, 3, 7, 11, 15]> : tensor<5xi64>
    %c_1 = stablehlo.constant dense<true> : tensor<5xi1>
    %c_2 = stablehlo.constant dense<4> : tensor<5xi64>
    %c_3 = stablehlo.constant dense<false> : tensor<4x5xi1>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<5x4xf32>) -> tensor<4x5xf32>
    %1 = stablehlo.convert %arg1 : (tensor<5xui32>) -> tensor<5xi64>
    %2 = stablehlo.add %1, %c_0 : tensor<5xi64>
    %3 = stablehlo.divide %2, %c_2 : tensor<5xi64>
    %4 = stablehlo.reshape %2 : (tensor<5xi64>) -> tensor<5x1xi64>
    %5 = stablehlo.reshape %3 : (tensor<5xi64>) -> tensor<5x1xi64>
    %6 = stablehlo.concatenate %4, %5, dim = 1 : (tensor<5x1xi64>, tensor<5x1xi64>) -> tensor<5x2xi64>
    %7 = stablehlo.remainder %6, %c : tensor<5x2xi64>
    %8 = "stablehlo.scatter"(%c_3, %7, %c_1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      stablehlo.return %arg3 : tensor<i1>
    }) : (tensor<4x5xi1>, tensor<5x2xi64>, tensor<5xi1>) -> tensor<4x5xi1>
    %9 = stablehlo.convert %8 : (tensor<4x5xi1>) -> tensor<4x5xf32>
    %10 = stablehlo.multiply %0, %9 : tensor<4x5xf32>
    %11 = stablehlo.transpose %10, dims = [1, 0] : (tensor<4x5xf32>) -> tensor<5x4xf32>
    return %11 : tensor<5x4xf32>
}

// CHECK: func.func @convertscatter(%arg0: tensor<5x4xf32>, %arg1: tensor<5xui32>) -> tensor<5x4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x5xf32>
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[4, 5], [4, 5], [4, 5], [4, 5], [4, 5]]> : tensor<5x2xi64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<[-1, 3, 7, 11, 15]> : tensor<5xi64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<4> : tensor<5xi64>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<5x4xf32>) -> tensor<4x5xf32>
// CHECK-NEXT:     %1 = stablehlo.convert %arg1 : (tensor<5xui32>) -> tensor<5xi64>
// CHECK-NEXT:     %2 = stablehlo.add %1, %c_0 : tensor<5xi64>
// CHECK-NEXT:     %3 = stablehlo.divide %2, %c_1 : tensor<5xi64>
// CHECK-NEXT:     %4 = stablehlo.reshape %2 : (tensor<5xi64>) -> tensor<5x1xi64>
// CHECK-NEXT:     %5 = stablehlo.reshape %3 : (tensor<5xi64>) -> tensor<5x1xi64>
// CHECK-NEXT:     %6 = stablehlo.concatenate %4, %5, dim = 1 : (tensor<5x1xi64>, tensor<5x1xi64>) -> tensor<5x2xi64>
// CHECK-NEXT:     %7 = stablehlo.remainder %6, %c : tensor<5x2xi64>
// CHECK-NEXT:     %8 = "stablehlo.gather"(%0, %7) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<4x5xf32>, tensor<5x2xi64>) -> tensor<5xf32>
// CHECK-NEXT:     %9 = "stablehlo.scatter"(%cst, %7, %8) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:       stablehlo.return %arg3 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<4x5xf32>, tensor<5x2xi64>, tensor<5xf32>) -> tensor<4x5xf32>
// CHECK-NEXT:     %10 = stablehlo.transpose %9, dims = [1, 0] : (tensor<4x5xf32>) -> tensor<5x4xf32>
// CHECK-NEXT:     return %10 : tensor<5x4xf32>
// CHECK-NEXT: }
