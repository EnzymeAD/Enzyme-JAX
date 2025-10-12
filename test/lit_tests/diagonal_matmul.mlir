// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main1(%arg0: tensor<50xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<50x8000xf64>, %arg2: tensor<8000xf64> {tf.aliasing_output = 1 : i32}, %arg3: tensor<8000xf64>, %arg4: tensor<50x8000xf64>, %arg5: tensor<8000xf64>, %arg6: tensor<8000xf64> {tf.aliasing_output = 2 : i32}) -> (tensor<50xf64>, tensor<8000xf64>, tensor<8000xf64>) {
    %cst = stablehlo.constant dense<-1.000000e+00> : tensor<8000xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<8000x8000xf64>
    %0 = stablehlo.reshape %arg5 : (tensor<8000xf64>) -> tensor<8000x1xf64>
    %1 = stablehlo.dot_general %arg1, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<50x8000xf64>, tensor<8000x1xf64>) -> tensor<50x1xf64>
    %2 = stablehlo.reshape %1 : (tensor<50x1xf64>) -> tensor<50xf64>
    %3 = stablehlo.dot_general %arg4, %1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<50x8000xf64>, tensor<50x1xf64>) -> tensor<8000x1xf64>
    %4 = stablehlo.reshape %3 : (tensor<8000x1xf64>) -> tensor<8000xf64>
    %5 = stablehlo.multiply %4, %cst : tensor<8000xf64>
    %6 = stablehlo.add %5, %arg5 : tensor<8000xf64>
    %7 = stablehlo.iota dim = 0 : tensor<8000x2xi64>
    %8 = "stablehlo.scatter"(%cst_0, %7, %arg3) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg7: tensor<f64>, %arg8: tensor<f64>):
      stablehlo.return %arg8 : tensor<f64>
    }) : (tensor<8000x8000xf64>, tensor<8000x2xi64>, tensor<8000xf64>) -> tensor<8000x8000xf64>
    %9 = stablehlo.reshape %6 : (tensor<8000xf64>) -> tensor<8000x1xf64>
    %10 = stablehlo.dot_general %8, %9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8000x8000xf64>, tensor<8000x1xf64>) -> tensor<8000x1xf64>
    %11 = stablehlo.reshape %10 : (tensor<8000x1xf64>) -> tensor<8000xf64>
    return %2, %6, %11 : tensor<50xf64>, tensor<8000xf64>, tensor<8000xf64>
}

// CHECK: func.func @main1(%arg0: tensor<50xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<50x8000xf64>, %arg2: tensor<8000xf64> {tf.aliasing_output = 1 : i32}, %arg3: tensor<8000xf64>, %arg4: tensor<50x8000xf64>, %arg5: tensor<8000xf64>, %arg6: tensor<8000xf64> {tf.aliasing_output = 2 : i32}) -> (tensor<50xf64>, tensor<8000xf64>, tensor<8000xf64>) {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg1, %arg5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<50x8000xf64>, tensor<8000xf64>) -> tensor<50xf64>
// CHECK-NEXT:     %1 = stablehlo.dot_general %arg4, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<50x8000xf64>, tensor<50xf64>) -> tensor<8000xf64>
// CHECK-NEXT:     %2 = stablehlo.subtract %arg5, %1 : tensor<8000xf64>
// CHECK-NEXT:     %3 = stablehlo.multiply %2, %arg3 : tensor<8000xf64>
// CHECK-NEXT:     return %0, %2, %3 : tensor<50xf64>, tensor<8000xf64>, tensor<8000xf64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<8000x8000xf64>, %arg1: tensor<8000xf64>) -> tensor<8000x8000xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<8000x8000xf64>
    %0 = stablehlo.iota dim = 0 : tensor<8000x2xi64>
    %1 = "stablehlo.scatter"(%cst, %0, %arg1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg3 : tensor<f64>
    }) : (tensor<8000x8000xf64>, tensor<8000x2xi64>, tensor<8000xf64>) -> tensor<8000x8000xf64>
    %2 = stablehlo.dot_general %1, %arg0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8000x8000xf64>, tensor<8000x8000xf64>) -> tensor<8000x8000xf64>
    return %2 : tensor<8000x8000xf64>
}

// CHECK: func.func @main2(%arg0: tensor<8000x8000xf64>, %arg1: tensor<8000xf64>) -> tensor<8000x8000xf64> {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<8000xf64>) -> tensor<8000x8000xf64>
// CHECK-NEXT:     %1 = stablehlo.multiply %0, %arg0 : tensor<8000x8000xf64>
// CHECK-NEXT:     return %1 : tensor<8000x8000xf64>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<8000xf64>, %arg1: tensor<8000x8000xf64>) -> tensor<8000x8000xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<8000x8000xf64>
    %0 = stablehlo.iota dim = 0 : tensor<8000x2xi64>
    %1 = "stablehlo.scatter"(%cst, %0, %arg0) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg3 : tensor<f64>
    }) : (tensor<8000x8000xf64>, tensor<8000x2xi64>, tensor<8000xf64>) -> tensor<8000x8000xf64>
    %2 = stablehlo.dot_general %arg1, %1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8000x8000xf64>, tensor<8000x8000xf64>) -> tensor<8000x8000xf64>
    return %2 : tensor<8000x8000xf64>
}

// CHECK: func.func @main3(%arg0: tensor<8000xf64>, %arg1: tensor<8000x8000xf64>) -> tensor<8000x8000xf64> {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<8000xf64>) -> tensor<8000x8000xf64>
// CHECK-NEXT:     %1 = stablehlo.multiply %0, %arg1 : tensor<8000x8000xf64>
// CHECK-NEXT:     return %1 : tensor<8000x8000xf64>
// CHECK-NEXT: }

func.func @main4(%arg0: tensor<1024xf32> {enzymexla.memory_effects = []}, %arg1: tensor<32x1024xf32> {enzymexla.memory_effects = []}, %arg2: tensor<32x1024xf32> {enzymexla.memory_effects = []}) -> tensor<f32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf32>
    %0 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<32x1024xf32>) -> tensor<1024x32xf32>
    %1 = stablehlo.iota dim = 0 : tensor<1024x2xi64>
    %2 = "stablehlo.scatter"(%cst_0, %1, %arg0) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<1024x1024xf32>, tensor<1024x2xi64>, tensor<1024xf32>) -> tensor<1024x1024xf32>
    %3 = stablehlo.dot_general %2, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x1024xf32>, tensor<32x1024xf32>) -> tensor<1024x32xf32>
    %4 = stablehlo.add %3, %0 : tensor<1024x32xf32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<1024x32xf32>, tensor<f32>) -> tensor<f32>
    return %5 : tensor<f32>
}

// CHECK: func.func @main4(%arg0: tensor<1024xf32> {enzymexla.memory_effects = []}, %arg1: tensor<32x1024xf32> {enzymexla.memory_effects = []}, %arg2: tensor<32x1024xf32> {enzymexla.memory_effects = []}) -> tensor<f32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024xf32>
// CHECK-NEXT:     %1 = stablehlo.multiply %0, %arg1 : tensor<32x1024xf32>
// CHECK-NEXT:     %2 = stablehlo.add %1, %arg2 : tensor<32x1024xf32>
// CHECK-NEXT:     %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x1024xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:     return %3 : tensor<f32>
// CHECK-NEXT: }

func.func @main5(%arg0: tensor<1024xf32> {enzymexla.memory_effects = []}, %arg1: tensor<1024x32xf32> {enzymexla.memory_effects = []}, %arg2: tensor<1024x32xf32> {enzymexla.memory_effects = []}) -> tensor<f32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf32>
    %0 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<1024x32xf32>) -> tensor<32x1024xf32>
    %1 = stablehlo.iota dim = 0 : tensor<1024x2xi64>
    %2 = "stablehlo.scatter"(%cst_0, %1, %arg0) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) : (tensor<1024x1024xf32>, tensor<1024x2xi64>, tensor<1024xf32>) -> tensor<1024x1024xf32>
    %3 = stablehlo.dot_general %arg1, %2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x32xf32>, tensor<1024x1024xf32>) -> tensor<32x1024xf32>
    %4 = stablehlo.add %3, %0 : tensor<32x1024xf32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x1024xf32>, tensor<f32>) -> tensor<f32>
    return %5 : tensor<f32>
}

// CHECK: func.func @main5(%arg0: tensor<1024xf32> {enzymexla.memory_effects = []}, %arg1: tensor<1024x32xf32> {enzymexla.memory_effects = []}, %arg2: tensor<1024x32xf32> {enzymexla.memory_effects = []}) -> tensor<f32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x32xf32>
// CHECK-NEXT:     %1 = stablehlo.multiply %0, %arg1 : tensor<1024x32xf32>
// CHECK-NEXT:     %2 = stablehlo.add %1, %arg2 : tensor<1024x32xf32>
// CHECK-NEXT:     %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<1024x32xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:     return %3 : tensor<f32>
// CHECK-NEXT: }
