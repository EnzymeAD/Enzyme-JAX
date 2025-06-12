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
// CHECK-NEXT:     %0 = stablehlo.concatenate %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, dim = 0 : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<24xi64>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<6xi64>) -> tensor<6x4xi64>
// CHECK-NEXT:     %2 = stablehlo.reshape %0 : (tensor<24xi64>) -> tensor<24x1xi64>
// CHECK-NEXT:     %3 = stablehlo.reshape %1 : (tensor<6x4xi64>) -> tensor<24x1xi64>
// CHECK-NEXT:     %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<24x1xi64>, tensor<24x1xi64>) -> tensor<24x2xi64>
// CHECK-NEXT:     %5 = stablehlo.subtract %4, %c : tensor<24x2xi64>
// CHECK-NEXT:     %6 = "stablehlo.scatter"(%cst_0, %5, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
// CHECK-NEXT:     ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK-NEXT:       stablehlo.return %arg4 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<1024x1024xf32>, tensor<24x2xi64>, tensor<24xf32>) -> tensor<1024x1024xf32>
// CHECK-NEXT:     %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// CHECK-NEXT:     return %7 : tensor<1024x1024xf32>
// CHECK-NEXT: }

func.func @convertscatter(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
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

// CHECK: func.func @convertscatter(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<7.3890562> : tensor<24xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<1024x1024xf32>
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<24x2xi64>
// CHECK-NEXT:     %0 = stablehlo.concatenate %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, dim = 0 : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<24xi64>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<6xi64>) -> tensor<6x4xi64>
// CHECK-NEXT:     %2 = stablehlo.reshape %0 : (tensor<24xi64>) -> tensor<24x1xi64>
// CHECK-NEXT:     %3 = stablehlo.reshape %1 : (tensor<6x4xi64>) -> tensor<24x1xi64>
// CHECK-NEXT:     %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<24x1xi64>, tensor<24x1xi64>) -> tensor<24x2xi64>
// CHECK-NEXT:     %5 = stablehlo.subtract %4, %c : tensor<24x2xi64>
// CHECK-NEXT:     %6 = "stablehlo.scatter"(%cst_0, %5, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
// CHECK-NEXT:     ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK-NEXT:       stablehlo.return %arg4 : tensor<f32>
// CHECK-NEXT:     }) : (tensor<1024x1024xf32>, tensor<24x2xi64>, tensor<24xf32>) -> tensor<1024x1024xf32>
// CHECK-NEXT:     %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// CHECK-NEXT:     return %7 : tensor<1024x1024xf32>
// CHECK-NEXT: }
