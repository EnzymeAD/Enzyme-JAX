// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s --check-prefix=NAN
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" %s | FileCheck %s --check-prefix=NONAN

module {
  func.func @main(%arg0: tensor<3000x3000xf64>, %arg1: tensor<3000x3000xf64>, %arg2: tensor<3000x3000xf64>) -> tensor<3000x3000xf64> {
    %cst = stablehlo.constant dense<1.000000e-01> : tensor<3000xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3000x3000xf64>
    %cst_1 = stablehlo.constant dense<5.000000e+00> : tensor<3000x3000xf64>

    %0 = stablehlo.iota dim = 0 : tensor<3000x2xi64>
    %1 = "stablehlo.scatter"(%cst_0, %0, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      stablehlo.return %arg4 : tensor<f64>
    }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>

    %2 = stablehlo.add %arg2, %1 : tensor<3000x3000xf64>

    %3 = stablehlo.multiply %cst_1, %1 : tensor<3000x3000xf64>
    %4 = stablehlo.add %arg1, %3 : tensor<3000x3000xf64>

    %5 = stablehlo.add %2, %4 : tensor<3000x3000xf64>
    return %5 : tensor<3000x3000xf64>
  }
}

// With no_nan both users of the zero-initialized scatter are simplified: the
// multiply folds into the scatter updates (5.0 * 0.1 = 0.5) and the scatter
// itself dies.
// NONAN: func.func @main(%arg0: tensor<3000x3000xf64>, %arg1: tensor<3000x3000xf64>, %arg2: tensor<3000x3000xf64>) -> tensor<3000x3000xf64> {
// NONAN-NEXT:   %cst = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// NONAN-NEXT:   %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<3000xf64>
// NONAN-NEXT:   %cst_1 = stablehlo.constant dense<1.000000e-01> : tensor<f64>
// NONAN-NEXT:   %cst_2 = stablehlo.constant dense<1.000000e-01> : tensor<3000xf64>
// NONAN-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<3000x2xi64>
// NONAN-NEXT:   %1 = "stablehlo.scatter"(%arg2, %0, %cst_2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// NONAN-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// NONAN-NEXT:     %4 = stablehlo.add %arg3, %cst_1 : tensor<f64>
// NONAN-NEXT:     stablehlo.return %4 : tensor<f64>
// NONAN-NEXT:   }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>
// NONAN-NEXT:   %2 = "stablehlo.scatter"(%arg1, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// NONAN-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// NONAN-NEXT:     %4 = stablehlo.add %arg3, %cst : tensor<f64>
// NONAN-NEXT:     stablehlo.return %4 : tensor<f64>
// NONAN-NEXT:   }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>
// NONAN-NEXT:   %3 = stablehlo.add %1, %2 : tensor<3000x3000xf64>
// NONAN-NEXT:   return %3 : tensor<3000x3000xf64>
// NONAN-NEXT: }

// Without no_nan the multiply over the zero-initialized scatter is not
// simplified, so that scatter stays alive; the add over it still is.
// NAN: func.func @main(%arg0: tensor<3000x3000xf64>, %arg1: tensor<3000x3000xf64>, %arg2: tensor<3000x3000xf64>) -> tensor<3000x3000xf64> {
// NAN-NEXT:   %cst = stablehlo.constant dense<1.000000e-01> : tensor<f64>
// NAN-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e-01> : tensor<3000xf64>
// NAN-NEXT:   %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<3000x3000xf64>
// NAN-NEXT:   %cst_2 = stablehlo.constant {enzymexla.finite = [#enzymexla<guaranteed GUARANTEED>]} dense<5.000000e+00> : tensor<3000x3000xf64>
// NAN-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<3000x2xi64>
// NAN-NEXT:   %1 = "stablehlo.scatter"(%cst_1, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// NAN-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// NAN-NEXT:     stablehlo.return %cst : tensor<f64>
// NAN-NEXT:   }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>
// NAN-NEXT:   %2 = "stablehlo.scatter"(%arg2, %0, %cst_0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// NAN-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// NAN-NEXT:     %6 = stablehlo.add %arg3, %cst : tensor<f64>
// NAN-NEXT:     stablehlo.return %6 : tensor<f64>
// NAN-NEXT:   }) : (tensor<3000x3000xf64>, tensor<3000x2xi64>, tensor<3000xf64>) -> tensor<3000x3000xf64>
// NAN-NEXT:   %3 = stablehlo.multiply %cst_2, %1 {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<3000x3000xf64>
// NAN-NEXT:   %4 = stablehlo.add %arg1, %3 : tensor<3000x3000xf64>
// NAN-NEXT:   %5 = stablehlo.add %2, %4 : tensor<3000x3000xf64>
// NAN-NEXT:   return %5 : tensor<3000x3000xf64>
// NAN-NEXT: }
