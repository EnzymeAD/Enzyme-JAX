// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=greedy_while_loop_batch_fission" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Each iteration scatters the induction variable, as a value its update
// region reads from the loop body, into a fresh scratch row at positions the
// induction variable selects, and stores the row into its carried buffer.
// The scatter cannot leave the loop: its region reads a value the loop
// makes. Lifting it used to clone the region with that reference intact.
func.func @main(%buf: tensor<4x8xf64>, %idx: tensor<4x2x1xi64>) -> tensor<4x8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %buf) : tensor<i64>, tensor<4x8xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %zero = stablehlo.convert %i : (tensor<i64>) -> tensor<f64>
    %rowi = stablehlo.dynamic_slice %idx, %i, %c0, %c0, sizes = [1, 2, 1] : (tensor<4x2x1xi64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x2x1xi64>
    %ri = stablehlo.reshape %rowi : (tensor<1x2x1xi64>) -> tensor<2x1xi64>
    %r = stablehlo.constant dense<1.0> : tensor<8xf64>
    %upd = stablehlo.constant dense<0.0> : tensor<2xf64>
    %s = "stablehlo.scatter"(%r, %ri, %upd) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %zero : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<8xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<8xf64>
    %s2 = stablehlo.reshape %s : (tensor<8xf64>) -> tensor<1x8xf64>
    %nacc = stablehlo.dynamic_update_slice %acc, %s2, %i, %c0 : (tensor<4x8xf64>, tensor<1x8xf64>, tensor<i64>, tensor<i64>) -> tensor<4x8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %nacc : tensor<i64>, tensor<4x8xf64>
  }
  return %0#1 : tensor<4x8xf64>
}

// CHECK:  func.func @main(%arg0: tensor<4x8xf64>, %arg1: tensor<4x2x1xi64>) -> tensor<4x8xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<8xf64>
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0) : tensor<i64>, tensor<4x8xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:     %2 = stablehlo.dynamic_slice %arg1, %iterArg, %c, %c, sizes = [1, 2, 1] : (tensor<4x2x1xi64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x2x1xi64>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<1x2x1xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:     %4 = "stablehlo.scatter"(%cst_0, %3, %cst) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %1 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<8xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<8xf64>
// CHECK-NEXT:     %5 = stablehlo.reshape %4 : (tensor<8xf64>) -> tensor<1x8xf64>
// CHECK-NEXT:     %6 = stablehlo.dynamic_update_slice %iterArg_3, %5, %iterArg, %c : (tensor<4x8xf64>, tensor<1x8xf64>, tensor<i64>, tensor<i64>) -> tensor<4x8xf64>
// CHECK-NEXT:     %7 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %7, %6 : tensor<i64>, tensor<4x8xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<4x8xf64>
// CHECK-NEXT: }
