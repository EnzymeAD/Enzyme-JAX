// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<3xf64> {enzymexla.memory_effects = []}, %arg1: tensor<3xf64> {enzymexla.memory_effects = []}, %arg2: tensor<2x3xf64> {enzymexla.memory_effects = []}) -> tensor<2x3xf64> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<[[0, 1], [1, 1], [2, 1]]> : tensor<3x2xi64>
    %c_0 = stablehlo.constant dense<[[0, 0], [1, 0], [2, 0]]> : tensor<3x2xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3x2xf64>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f64>
    %cst_2 = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %cst_3 = stablehlo.constant dense<5.000000e+00> : tensor<f64>
    %0 = stablehlo.add %arg0, %arg0 : tensor<3xf64>
    %1 = stablehlo.multiply %arg1, %0 : tensor<3xf64>
    %2 = stablehlo.multiply %arg1, %arg1 : tensor<3xf64>
    %3 = stablehlo.slice %arg2 [0:1, 0:3] : (tensor<2x3xf64>) -> tensor<1x3xf64>
    %4 = stablehlo.reshape %3 : (tensor<1x3xf64>) -> tensor<3xf64>
    %5 = stablehlo.slice %arg2 [1:2, 0:3] : (tensor<2x3xf64>) -> tensor<1x3xf64>
    %6 = stablehlo.reshape %5 : (tensor<1x3xf64>) -> tensor<3xf64>
    %7 = stablehlo.multiply %1, %4 : tensor<3xf64>
    %8 = stablehlo.multiply %4, %2 : tensor<3xf64>
    %9 = stablehlo.multiply %arg0, %4 : tensor<3xf64>
    %10 = stablehlo.multiply %4, %arg1 : tensor<3xf64>
    %11 = stablehlo.multiply %9, %arg1 : tensor<3xf64>
    %12 = stablehlo.multiply %arg0, %10 : tensor<3xf64>
    %13 = stablehlo.add %11, %12 : tensor<3xf64>
    %14 = stablehlo.multiply %10, %arg1 : tensor<3xf64>
    %15 = stablehlo.add %7, %13 : tensor<3xf64>
    %16 = stablehlo.add %8, %14 : tensor<3xf64>
    %17 = stablehlo.add %15, %13 : tensor<3xf64>
    %18 = stablehlo.add %16, %14 : tensor<3xf64>
    // This update computation is invalid, as it mixes different operand indices
    // We are testing that the pattern correctly detects this and doesn't split
    // CHECK: %19:2 = "stablehlo.scatter"(%cst_2, %cst_2, %c_1, %18, %16)
    %19:2 = "stablehlo.scatter"(%cst, %cst, %c_0, %18, %17) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>, %arg5: tensor<f64>, %arg6: tensor<f64>):
      %add = stablehlo.add %arg3, %cst_0 : tensor<f64>
      %mul = stablehlo.multiply %arg4, %cst_1 : tensor<f64>
      %sub = stablehlo.subtract %arg5, %add : tensor<f64>
      %mul_1 = stablehlo.multiply %arg6, %mul : tensor<f64>
      %mul_2 = stablehlo.multiply %mul_1, %sub : tensor<f64>
      stablehlo.return %sub, %mul_2 : tensor<f64>, tensor<f64>
    }) : (tensor<3x2xf64>, tensor<3x2xf64>, tensor<3x2xi64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<3x2xf64>, tensor<3x2xf64>)
    %20 = stablehlo.multiply %1, %6 : tensor<3xf64>
    %21 = stablehlo.multiply %6, %2 : tensor<3xf64>
    %22 = stablehlo.multiply %arg0, %6 : tensor<3xf64>
    %23 = stablehlo.multiply %6, %arg1 : tensor<3xf64>
    %24 = stablehlo.multiply %22, %arg1 : tensor<3xf64>
    %25 = stablehlo.multiply %arg0, %23 : tensor<3xf64>
    %26 = stablehlo.add %24, %25 : tensor<3xf64>
    %27 = stablehlo.multiply %23, %arg1 : tensor<3xf64>
    %28 = stablehlo.add %20, %26 : tensor<3xf64>
    %29 = stablehlo.add %21, %27 : tensor<3xf64>
    %30 = stablehlo.add %28, %26 : tensor<3xf64>
    %31 = stablehlo.add %29, %27 : tensor<3xf64>
    // CHECK: %32 = "stablehlo.scatter"(%19#0, %c, %31) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    // CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
    // CHECK-NEXT:     %36 = stablehlo.subtract %arg3, %cst_0 : tensor<f64>
    // CHECK-NEXT:     %37 = stablehlo.subtract %36, %arg4 : tensor<f64>
    // CHECK-NEXT:     stablehlo.return %37 : tensor<f64>
    // CHECK-NEXT:   }) : (tensor<3x2xf64>, tensor<3x2xi64>, tensor<3xf64>) -> tensor<3x2xf64>
    // CHECK-NEXT:  %33 = "stablehlo.scatter"(%19#1, %c, %29) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    // CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
    // CHECK-NEXT:     %36 = stablehlo.multiply %arg3, %cst : tensor<f64>
    // CHECK-NEXT:     %37 = stablehlo.multiply %arg4, %36 : tensor<f64>
    // CHECK-NEXT:     stablehlo.return %37 : tensor<f64>
    // CHECK-NEXT:   }) : (tensor<3x2xf64>, tensor<3x2xi64>, tensor<3xf64>) -> tensor<3x2xf64>
    %32:2 = "stablehlo.scatter"(%19#0, %19#1, %c, %31, %30) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>, %arg5: tensor<f64>, %arg6: tensor<f64>):
      %sub = stablehlo.subtract %arg3, %cst_2 : tensor<f64>
      %mul = stablehlo.multiply %arg4, %cst_3 : tensor<f64>
      %neg = stablehlo.negate %arg5 : tensor<f64>
      %mul_1 = stablehlo.multiply %arg6, %mul : tensor<f64>
      %neg_1 = stablehlo.add %neg, %sub : tensor<f64>
      stablehlo.return %neg_1, %mul_1 : tensor<f64>, tensor<f64>
    }) : (tensor<3x2xf64>, tensor<3x2xf64>, tensor<3x2xi64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<3x2xf64>, tensor<3x2xf64>)
    %33 = stablehlo.transpose %32#1, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %34 = stablehlo.transpose %32#0, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    // CHECK: %34 = stablehlo.add %33, %32 : tensor<3x2xf64>
    %35 = stablehlo.add %33, %34 : tensor<2x3xf64>
    // CHECK: %35 = stablehlo.transpose %34, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    return %35 : tensor<2x3xf64>
  }
}
