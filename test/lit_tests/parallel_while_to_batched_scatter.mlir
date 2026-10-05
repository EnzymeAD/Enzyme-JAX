// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

func.func @set_subvector(%idx: tensor<8xi64>, %y: tensor<32xf64>, %s: tensor<f64>) -> tensor<32xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<32xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %j = stablehlo.dynamic_slice %idx, %i, sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
    %u = "stablehlo.scatter"(%acc, %j, %s) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<32xf64>, tensor<1xi64>, tensor<f64>) -> tensor<32xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<32xf64>
  }
  return %0#1 : tensor<32xf64>
}

// CHECK:  func.func @set_subvector(%arg0: tensor<8xi64>, %arg1: tensor<32xf64>, %arg2: tensor<f64>) -> tensor<32xf64> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT:   %2 = "stablehlo.scatter"(%arg1, %0, %1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<32xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<32xf64>
// CHECK-NEXT:   return %2 : tensor<32xf64>
// CHECK-NEXT: }

// -----

func.func @strided_accumulate(%x: tensor<8xf64>, %y: tensor<32xf64>) -> tensor<32xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<32xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
    %vs = stablehlo.reshape %v : (tensor<1xf64>) -> tensor<f64>
    %m = stablehlo.multiply %i, %c2 : tensor<i64>
    %j = stablehlo.add %m, %c1 : tensor<i64>
    %jr = stablehlo.reshape %j : (tensor<i64>) -> tensor<1xi64>
    %u = "stablehlo.scatter"(%acc, %jr, %vs) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      %r = stablehlo.add %a, %b : tensor<f64>
      stablehlo.return %r : tensor<f64>
    }) : (tensor<32xf64>, tensor<1xi64>, tensor<f64>) -> tensor<32xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<32xf64>
  }
  return %0#1 : tensor<32xf64>
}

// CHECK:  func.func @strided_accumulate(%arg0: tensor<8xf64>, %arg1: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<8xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %4 = stablehlo.multiply %0, %3 : tensor<8xi64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %6 = stablehlo.add %4, %5 : tensor<8xi64>
// CHECK-NEXT:   %7 = stablehlo.reshape %6 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %8 = "stablehlo.scatter"(%arg1, %7, %2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     %9 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %9 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<32xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<32xf64>
// CHECK-NEXT:   return %8 : tensor<32xf64>
// CHECK-NEXT: }

// -----

func.func @reads_carried(%idx: tensor<8xi64>, %y: tensor<32xf64>) -> tensor<32xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<32xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %j = stablehlo.dynamic_slice %idx, %i, sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
    %v = stablehlo.dynamic_slice %acc, %i, sizes = [1] : (tensor<32xf64>, tensor<i64>) -> tensor<1xf64>
    %vs = stablehlo.reshape %v : (tensor<1xf64>) -> tensor<f64>
    %u = "stablehlo.scatter"(%acc, %j, %vs) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<32xf64>, tensor<1xi64>, tensor<f64>) -> tensor<32xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<32xf64>
  }
  return %0#1 : tensor<32xf64>
}

// CHECK:  func.func @reads_carried(%arg0: tensor<8xi64>, %arg1: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %1 = stablehlo.slice %arg1 [0:8] : (tensor<32xf64>) -> tensor<8xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<8xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:   %3 = stablehlo.reshape %2 : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:   %4 = "stablehlo.scatter"(%arg1, %0, %3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<32xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<32xf64>
// CHECK-NEXT:   return %4 : tensor<32xf64>
// CHECK-NEXT: }

// -----

func.func @sequential(%idx: tensor<8xi64>, %y: tensor<32xf64>, %s: tensor<f64>) -> tensor<32xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<32xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %j = stablehlo.dynamic_slice %idx, %i, sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
    %u = "stablehlo.scatter"(%acc, %j, %s) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<32xf64>, tensor<1xi64>, tensor<f64>) -> tensor<32xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<32xf64>
  }
  return %0#1 : tensor<32xf64>
}

// CHECK:  func.func @sequential(%arg0: tensor<8xi64>, %arg1: tensor<32xf64>, %arg2: tensor<f64>) -> tensor<32xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg1) : tensor<i64>, tensor<32xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %arg0, %iterArg, sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:     %2 = "stablehlo.scatter"(%iterArg_2, %1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
// CHECK-NEXT:     ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<32xf64>, tensor<1xi64>, tensor<f64>) -> tensor<32xf64>
// CHECK-NEXT:     %3 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %3, %2 : tensor<i64>, tensor<32xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<32xf64>
// CHECK-NEXT: }

// -----

func.func @select_iv_predicate(%x: tensor<8x4xf64>, %y: tensor<8x4x4xf64>) -> tensor<8x4x4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %z = stablehlo.constant dense<0.0> : tensor<4xf64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<8x4x4xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 4] : (tensor<8x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
    %vs = stablehlo.reshape %v : (tensor<1x4xf64>) -> tensor<4xf64>
    %p = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %s = stablehlo.select %p, %vs, %z : tensor<i1>, tensor<4xf64>
    %sr = stablehlo.reshape %s : (tensor<4xf64>) -> tensor<1x1x4xf64>
    %u = stablehlo.dynamic_update_slice %acc, %sr, %i, %c0, %c0 : (tensor<8x4x4xf64>, tensor<1x1x4xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<8x4x4xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<8x4x4xf64>
  }
  return %0#1 : tensor<8x4x4xf64>
}

// CHECK:  func.func @select_iv_predicate(%arg0: tensor<8x4xf64>, %arg1: tensor<8x4x4xf64>) -> tensor<8x4x4xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<8x4xf64>) -> tensor<8x1x4xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<8x1x4xf64>) -> tensor<8x4xf64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %4 = stablehlo.compare LT, %0, %3 : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %cst, dims = [1] : (tensor<4xf64>) -> tensor<8x4xf64>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<8xi1>) -> tensor<8x4xi1>
// CHECK-NEXT:   %7 = stablehlo.select %6, %2, %5 : tensor<8x4xi1>, tensor<8x4xf64>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<8x4xf64>) -> tensor<8x1x1x4xf64>
// CHECK-NEXT:   %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %10 = stablehlo.clamp %c_1, %0, %c_0 : (tensor<i64>, tensor<8xi64>, tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %11 = stablehlo.reshape %10 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %12 = stablehlo.clamp %c_1, %9, %c : (tensor<i64>, tensor<8xi64>, tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %13 = stablehlo.reshape %12 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %14 = stablehlo.clamp %c_1, %9, %c_1 : (tensor<i64>, tensor<8xi64>, tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %15 = stablehlo.reshape %14 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %16 = stablehlo.concatenate %11, %13, %15, dim = 1 : (tensor<8x1xi64>, tensor<8x1xi64>, tensor<8x1xi64>) -> tensor<8x3xi64>
// CHECK-NEXT:   %17 = "stablehlo.scatter"(%arg1, %16, %8) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], scatter_dims_to_operand_dims = [0, 1, 2], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<8x4x4xf64>, tensor<8x3xi64>, tensor<8x1x1x4xf64>) -> tensor<8x4x4xf64>
// CHECK-NEXT:   return %17 : tensor<8x4x4xf64>
// CHECK-NEXT: }

// -----

func.func @invariant_reduce_captures_iv(%x: tensor<4xf64>, %y: tensor<8xf64>) -> tensor<8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %z = stablehlo.constant dense<0.0> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<8xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %fi = stablehlo.convert %i : (tensor<i64>) -> tensor<f64>
    %r = stablehlo.reduce(%x init: %z) across dimensions = [0] : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
     reducer(%a: tensor<f64>, %b: tensor<f64>) {
      %m = stablehlo.multiply %b, %fi : tensor<f64>
      %s = stablehlo.add %a, %m : tensor<f64>
      stablehlo.return %s : tensor<f64>
    }
    %rr = stablehlo.reshape %r : (tensor<f64>) -> tensor<1xf64>
    %u = stablehlo.dynamic_update_slice %acc, %rr, %i : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<8xf64>
  }
  return %0#1 : tensor<8xf64>
}

// CHECK:  func.func @invariant_reduce_captures_iv(%arg0: tensor<4xf64>, %arg1: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg1) : tensor<i64>, tensor<8xf64> attributes {enzymexla.parallel}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:     %2 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [0] : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:      reducer(%arg2: tensor<f64>, %arg3: tensor<f64>)  {
// CHECK-NEXT:       %6 = stablehlo.multiply %arg3, %1 : tensor<f64>
// CHECK-NEXT:       %7 = stablehlo.add %arg2, %6 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %7 : tensor<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:     %4 = stablehlo.dynamic_update_slice %iterArg_2, %3, %iterArg : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
// CHECK-NEXT:     %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %5, %4 : tensor<i64>, tensor<8xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<8xf64>
// CHECK-NEXT: }

// -----

func.func @dus_window_clamped(%x: tensor<8x4xf64>, %y: tensor<32xf64>) -> tensor<32xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<32xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 4] : (tensor<8x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
    %vs = stablehlo.reshape %v : (tensor<1x4xf64>) -> tensor<4xf64>
    %m = stablehlo.multiply %i, %c4 : tensor<i64>
    %j = stablehlo.add %m, %c1 : tensor<i64>
    %u = stablehlo.dynamic_update_slice %acc, %vs, %j : (tensor<32xf64>, tensor<4xf64>, tensor<i64>) -> tensor<32xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<32xf64>
  }
  return %0#1 : tensor<32xf64>
}

// CHECK:  func.func @dus_window_clamped(%arg0: tensor<8x4xf64>, %arg1: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<28> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<8x4xf64>) -> tensor<8x1x4xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<8x1x4xf64>) -> tensor<8x4xf64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %4 = stablehlo.multiply %0, %3 : tensor<8xi64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %6 = stablehlo.add %4, %5 : tensor<8xi64>
// CHECK-NEXT:   %7 = stablehlo.clamp %c_0, %6, %c : (tensor<i64>, tensor<8xi64>, tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %9 = "stablehlo.scatter"(%arg1, %8, %2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<32xf64>, tensor<8x1xi64>, tensor<8x4xf64>) -> tensor<32xf64>
// CHECK-NEXT:   return %9 : tensor<32xf64>
// CHECK-NEXT: }

// -----

func.func @carried_scatter_batching_dims(%idx: tensor<8x2x1xi64>, %y: tensor<2x32xf64>, %s: tensor<2xf64>) -> tensor<2x32xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<2x32xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %j = stablehlo.dynamic_slice %idx, %i, %c0, %c0, sizes = [1, 2, 1] : (tensor<8x2x1xi64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x2x1xi64>
    %jr = stablehlo.reshape %j : (tensor<1x2x1xi64>) -> tensor<2x1xi64>
    %u = "stablehlo.scatter"(%acc, %jr, %s) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<2x32xf64>, tensor<2x1xi64>, tensor<2xf64>) -> tensor<2x32xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<2x32xf64>
  }
  return %0#1 : tensor<2x32xf64>
}

// CHECK:  func.func @carried_scatter_batching_dims(%arg0: tensor<8x2x1xi64>, %arg1: tensor<2x32xf64>, %arg2: tensor<2xf64>) -> tensor<2x32xf64> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<8x2x1xi64>) -> tensor<8x1x2x1xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<8x1x2x1xi64>) -> tensor<8x2x1xi64>
// CHECK-NEXT:   %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2xf64>) -> tensor<8x2xf64>
// CHECK-NEXT:   %3 = "stablehlo.scatter"(%arg1, %1, %2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<2x32xf64>, tensor<8x2x1xi64>, tensor<8x2xf64>) -> tensor<2x32xf64>
// CHECK-NEXT:   return %3 : tensor<2x32xf64>
// CHECK-NEXT: }

// -----

func.func @gather_drops_sorted(%idx: tensor<8xi64>, %x: tensor<32xf64>, %y: tensor<8xf64>) -> tensor<8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<8xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %j = stablehlo.dynamic_slice %idx, %i, sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
    %g = "stablehlo.gather"(%x, %j) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<32xf64>, tensor<1xi64>) -> tensor<f64>
    %gr = stablehlo.reshape %g : (tensor<f64>) -> tensor<1xf64>
    %u = stablehlo.dynamic_update_slice %acc, %gr, %i : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<8xf64>
  }
  return %0#1 : tensor<8xf64>
}

// CHECK:  func.func @gather_drops_sorted(%arg0: tensor<8xi64>, %arg1: tensor<32xf64>, %arg2: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %2 = "stablehlo.gather"(%arg1, %1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<32xf64>, tensor<8x1xi64>) -> tensor<8xf64>
// CHECK-NEXT:   %3 = stablehlo.reshape %2 : (tensor<8xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:   %4 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<8xi64>, tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %6 = "stablehlo.scatter"(%arg2, %5, %3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<8xf64>, tensor<8x1xi64>, tensor<8x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:   return %6 : tensor<8xf64>
// CHECK-NEXT: }

// -----

func.func @invariant_side_effect(%x: tensor<8xf64>, %y: tensor<8xf64>) -> tensor<8xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<8xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %e = stablehlo.custom_call @effect(%x) {has_side_effect = true} : (tensor<8xf64>) -> tensor<f64>
    %er = stablehlo.reshape %e : (tensor<f64>) -> tensor<1xf64>
    %u = stablehlo.dynamic_update_slice %acc, %er, %i : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<8xf64>
  }
  return %0#1 : tensor<8xf64>
}

// CHECK:  func.func @invariant_side_effect(%arg0: tensor<8xf64>, %arg1: tensor<8xf64>) -> tensor<8xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg1) : tensor<i64>, tensor<8xf64> attributes {enzymexla.parallel}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.custom_call @effect(%arg0) {has_side_effect = true} : (tensor<8xf64>) -> tensor<f64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:     %3 = stablehlo.dynamic_update_slice %iterArg_2, %2, %iterArg : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
// CHECK-NEXT:     %4 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %4, %3 : tensor<i64>, tensor<8xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<8xf64>
// CHECK-NEXT: }

// -----

// y[2i] += x[i]; y[2i+1] += x[i]: each write reads what this iteration wrote
// before it.
func.func @accumulate(%x: tensor<8xf64>, %y: tensor<16xf64>) -> tensor<16xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<16xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
    %j = stablehlo.multiply %i, %c2 : tensor<i64>
    %old = stablehlo.dynamic_slice %acc, %j, sizes = [1] : (tensor<16xf64>, tensor<i64>) -> tensor<1xf64>
    %s = stablehlo.add %old, %v : tensor<1xf64>
    %acc1 = stablehlo.dynamic_update_slice %acc, %s, %j : (tensor<16xf64>, tensor<1xf64>, tensor<i64>) -> tensor<16xf64>
    %j1 = stablehlo.add %j, %c1 : tensor<i64>
    %old1 = stablehlo.dynamic_slice %acc1, %j1, sizes = [1] : (tensor<16xf64>, tensor<i64>) -> tensor<1xf64>
    %s1 = stablehlo.add %old1, %v : tensor<1xf64>
    %acc2 = stablehlo.dynamic_update_slice %acc1, %s1, %j1 : (tensor<16xf64>, tensor<1xf64>, tensor<i64>) -> tensor<16xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %acc2 : tensor<i64>, tensor<16xf64>
  }
  return %0#1 : tensor<16xf64>
}

// CHECK:  func.func @accumulate(%arg0: tensor<8xf64>, %arg1: tensor<16xf64>) -> tensor<16xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<15> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<8xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:   %2 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %3 = stablehlo.multiply %0, %2 : tensor<8xi64>
// CHECK-NEXT:   %4 = stablehlo.slice %arg1 [0:15] : (tensor<16xf64>) -> tensor<15xf64>
// CHECK-NEXT:   %5 = stablehlo.slice %4 [0:15:2] : (tensor<15xf64>) -> tensor<8xf64>
// CHECK-NEXT:   %6 = stablehlo.reshape %5 : (tensor<8xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:   %7 = stablehlo.add %6, %1 : tensor<8x1xf64>
// CHECK-NEXT:   %8 = stablehlo.clamp %c_0, %3, %c : (tensor<i64>, tensor<8xi64>, tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %9 = stablehlo.reshape %8 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %10 = "stablehlo.scatter"(%arg1, %9, %7) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<16xf64>, tensor<8x1xi64>, tensor<8x1xf64>) -> tensor<16xf64>
// CHECK-NEXT:   %11 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %12 = stablehlo.add %3, %11 : tensor<8xi64>
// CHECK-NEXT:   %13 = stablehlo.slice %10 [1:16] : (tensor<16xf64>) -> tensor<15xf64>
// CHECK-NEXT:   %14 = stablehlo.slice %13 [0:15:2] : (tensor<15xf64>) -> tensor<8xf64>
// CHECK-NEXT:   %15 = stablehlo.reshape %14 : (tensor<8xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:   %16 = stablehlo.add %15, %1 : tensor<8x1xf64>
// CHECK-NEXT:   %17 = stablehlo.clamp %c_0, %12, %c : (tensor<i64>, tensor<8xi64>, tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %18 = stablehlo.reshape %17 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %19 = "stablehlo.scatter"(%10, %18, %16) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<16xf64>, tensor<8x1xi64>, tensor<8x1xf64>) -> tensor<16xf64>
// CHECK-NEXT:   return %19 : tensor<16xf64>
// CHECK-NEXT: }

// -----

// A reduce with two inputs, a value and a flag; the flag's input is the same
// every iteration and is broadcast.
func.func @two_input_reduce(%x: tensor<32xf64>, %m: tensor<8xi1>) -> (tensor<4xf64>, tensor<4xi1>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %ninf = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
  %false = stablehlo.constant dense<false> : tensor<i1>
  %zv = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
  %zb = stablehlo.constant dense<false> : tensor<4xi1>
  %0:3 = stablehlo.while(%i = %c0, %av = %zv, %ab = %zb) : tensor<i64>, tensor<4xf64>, tensor<4xi1> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %s = stablehlo.multiply %i, %c8 : tensor<i64>
    %w = stablehlo.dynamic_slice %x, %s, sizes = [8] : (tensor<32xf64>, tensor<i64>) -> tensor<8xf64>
    %r:2 = stablehlo.reduce(%w init: %ninf), (%m init: %false) across dimensions = [0] : (tensor<8xf64>, tensor<8xi1>, tensor<f64>, tensor<i1>) -> (tensor<f64>, tensor<i1>)
     reducer(%a0: tensor<f64>, %b0: tensor<f64>) (%a1: tensor<i1>, %b1: tensor<i1>)  {
      %mx = stablehlo.maximum %a0, %b0 : tensor<f64>
      %o = stablehlo.or %a1, %b1 : tensor<i1>
      stablehlo.return %mx, %o : tensor<f64>, tensor<i1>
    }
    %rv = stablehlo.reshape %r#0 : (tensor<f64>) -> tensor<1xf64>
    %rb = stablehlo.reshape %r#1 : (tensor<i1>) -> tensor<1xi1>
    %uv = stablehlo.dynamic_update_slice %av, %rv, %i : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
    %ub = stablehlo.dynamic_update_slice %ab, %rb, %i : (tensor<4xi1>, tensor<1xi1>, tensor<i64>) -> tensor<4xi1>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %uv, %ub : tensor<i64>, tensor<4xf64>, tensor<4xi1>
  }
  return %0#1, %0#2 : tensor<4xf64>, tensor<4xi1>
}

// CHECK:  func.func @two_input_reduce(%arg0: tensor<32xf64>, %arg1: tensor<8xi1>) -> (tensor<4xf64>, tensor<4xi1>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg0 : (tensor<32xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<4x8xf64>) -> tensor<32xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<32xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<8xi1>) -> tensor<4x8xi1>
// CHECK-NEXT:    %5:2 = stablehlo.reduce(%3 init: %cst), (%4 init: %c_1) across dimensions = [1] : (tensor<4x8xf64>, tensor<4x8xi1>, tensor<f64>, tensor<i1>) -> (tensor<4xf64>, tensor<4xi1>)
// CHECK-NEXT:     reducer(%arg2: tensor<f64>, %arg4: tensor<f64>) (%arg3: tensor<i1>, %arg5: tensor<i1>)  {
// CHECK-NEXT:      %14 = stablehlo.maximum %arg2, %arg4 : tensor<f64>
// CHECK-NEXT:      %15 = stablehlo.or %arg3, %arg5 : tensor<i1>
// CHECK-NEXT:      stablehlo.return %14, %15 : tensor<f64>, tensor<i1>
// CHECK-NEXT:    }
// CHECK-NEXT:    %6 = stablehlo.reshape %5#0 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:    %7 = stablehlo.reshape %5#1 : (tensor<4xi1>) -> tensor<4x1xi1>
// CHECK-NEXT:    %8 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:    %10 = "stablehlo.scatter"(%cst_2, %9, %6) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<4xf64>, tensor<4x1xi64>, tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:    %11 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %12 = stablehlo.reshape %11 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:    %13 = "stablehlo.scatter"(%c_3, %12, %7) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<i1>
// CHECK-NEXT:    }) : (tensor<4xi1>, tensor<4x1xi64>, tensor<4x1xi1>) -> tensor<4xi1>
// CHECK-NEXT:    return %10, %13 : tensor<4xf64>, tensor<4xi1>
// CHECK-NEXT:  }

// -----

// Ops along one dimension of a window act along the same dimension past the
// batch one.
func.func @wrap_extend_rotate(%x: tensor<32xf64>) -> (tensor<4x10xf64>, tensor<4x10xf64>, tensor<4x8xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %z10 = stablehlo.constant dense<0.000000e+00> : tensor<4x10xf64>
  %z8 = stablehlo.constant dense<0.000000e+00> : tensor<4x8xf64>
  %0:4 = stablehlo.while(%i = %c0, %aw = %z10, %ae = %z10, %ar = %z8) : tensor<i64>, tensor<4x10xf64>, tensor<4x10xf64>, tensor<4x8xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %s = stablehlo.multiply %i, %c8 : tensor<i64>
    %w = stablehlo.dynamic_slice %x, %s, sizes = [8] : (tensor<32xf64>, tensor<i64>) -> tensor<8xf64>
    %wr = "enzymexla.wrap"(%w) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<8xf64>) -> tensor<10xf64>
    %ex = "enzymexla.extend"(%w) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<8xf64>) -> tensor<10xf64>
    %ro = "enzymexla.rotate"(%w) <{amount = 3 : i32, dimension = 0 : i32}> : (tensor<8xf64>) -> tensor<8xf64>
    %wr2 = stablehlo.reshape %wr : (tensor<10xf64>) -> tensor<1x10xf64>
    %ex2 = stablehlo.reshape %ex : (tensor<10xf64>) -> tensor<1x10xf64>
    %ro2 = stablehlo.reshape %ro : (tensor<8xf64>) -> tensor<1x8xf64>
    %uw = stablehlo.dynamic_update_slice %aw, %wr2, %i, %c0 : (tensor<4x10xf64>, tensor<1x10xf64>, tensor<i64>, tensor<i64>) -> tensor<4x10xf64>
    %ue = stablehlo.dynamic_update_slice %ae, %ex2, %i, %c0 : (tensor<4x10xf64>, tensor<1x10xf64>, tensor<i64>, tensor<i64>) -> tensor<4x10xf64>
    %ur = stablehlo.dynamic_update_slice %ar, %ro2, %i, %c0 : (tensor<4x8xf64>, tensor<1x8xf64>, tensor<i64>, tensor<i64>) -> tensor<4x8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %uw, %ue, %ur : tensor<i64>, tensor<4x10xf64>, tensor<4x10xf64>, tensor<4x8xf64>
  }
  return %0#1, %0#2, %0#3 : tensor<4x10xf64>, tensor<4x10xf64>, tensor<4x8xf64>
}

// CHECK:  func.func @wrap_extend_rotate(%arg0: tensor<32xf64>) -> (tensor<4x10xf64>, tensor<4x10xf64>, tensor<4x8xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x10xf64>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4x8xf64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg0 : (tensor<32xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<4x8xf64>) -> tensor<32xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<32xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:    %4 = "enzymexla.wrap"(%3) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x8xf64>) -> tensor<4x10xf64>
// CHECK-NEXT:    %5 = "enzymexla.extend"(%3) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x8xf64>) -> tensor<4x10xf64>
// CHECK-NEXT:    %6 = "enzymexla.rotate"(%3) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:    %7 = stablehlo.reshape %4 : (tensor<4x10xf64>) -> tensor<4x1x10xf64>
// CHECK-NEXT:    %8 = stablehlo.reshape %5 : (tensor<4x10xf64>) -> tensor<4x1x10xf64>
// CHECK-NEXT:    %9 = stablehlo.reshape %6 : (tensor<4x8xf64>) -> tensor<4x1x8xf64>
// CHECK-NEXT:    %10 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %11 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %12 = stablehlo.reshape %11 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:    %13 = stablehlo.clamp %c_0, %10, %c_0 : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %14 = stablehlo.reshape %13 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:    %15 = stablehlo.concatenate %12, %14, dim = 1 : (tensor<4x1xi64>, tensor<4x1xi64>) -> tensor<4x2xi64>
// CHECK-NEXT:    %16 = "stablehlo.scatter"(%cst, %15, %7) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg2 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<4x10xf64>, tensor<4x2xi64>, tensor<4x1x10xf64>) -> tensor<4x10xf64>
// CHECK-NEXT:    %17 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %18 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %19 = stablehlo.reshape %18 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:    %20 = stablehlo.clamp %c_0, %17, %c_0 : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %21 = stablehlo.reshape %20 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:    %22 = stablehlo.concatenate %19, %21, dim = 1 : (tensor<4x1xi64>, tensor<4x1xi64>) -> tensor<4x2xi64>
// CHECK-NEXT:    %23 = "stablehlo.scatter"(%cst, %22, %8) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg2 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<4x10xf64>, tensor<4x2xi64>, tensor<4x1x10xf64>) -> tensor<4x10xf64>
// CHECK-NEXT:    %24 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %25 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %26 = stablehlo.reshape %25 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:    %27 = stablehlo.clamp %c_0, %24, %c_0 : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %28 = stablehlo.reshape %27 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:    %29 = stablehlo.concatenate %26, %28, dim = 1 : (tensor<4x1xi64>, tensor<4x1xi64>) -> tensor<4x2xi64>
// CHECK-NEXT:    %30 = "stablehlo.scatter"(%cst_1, %29, %9) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg2 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<4x8xf64>, tensor<4x2xi64>, tensor<4x1x8xf64>) -> tensor<4x8xf64>
// CHECK-NEXT:    return %16, %23, %30 : tensor<4x10xf64>, tensor<4x10xf64>, tensor<4x8xf64>
// CHECK-NEXT:  }
// A constant-trip loop nested in the parallel loop: it keeps running, over
// the batched values, and the buffer it carries continues the write chain.
// for i: acc = 0; for j < 3: acc += x[i] * j; y[3i + j] = acc
func.func @inner_loop(%x: tensor<8xf64>, %y: tensor<24xf64>) -> tensor<24xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %buf = %y) : tensor<i64>, tensor<24xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
    %vs = stablehlo.reshape %v : (tensor<1xf64>) -> tensor<f64>
    %base = stablehlo.multiply %i, %c3 : tensor<i64>
    %1:3 = stablehlo.while(%j = %c0, %acc = %zero, %b = %buf) : tensor<i64>, tensor<f64>, tensor<24xf64>
     cond {
      %c = stablehlo.compare LT, %j, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %jf = stablehlo.convert %j : (tensor<i64>) -> tensor<f64>
      %t = stablehlo.multiply %vs, %jf : tensor<f64>
      %acc2 = stablehlo.add %acc, %t : tensor<f64>
      %idx = stablehlo.add %base, %j : tensor<i64>
      %idxr = stablehlo.reshape %idx : (tensor<i64>) -> tensor<1xi64>
      %b2 = "stablehlo.scatter"(%b, %idxr, %acc2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%p: tensor<f64>, %q: tensor<f64>):
        stablehlo.return %q : tensor<f64>
      }) : (tensor<24xf64>, tensor<1xi64>, tensor<f64>) -> tensor<24xf64>
      %jn = stablehlo.add %j, %c1 : tensor<i64>
      stablehlo.return %jn, %acc2, %b2 : tensor<i64>, tensor<f64>, tensor<24xf64>
    }
    %in = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %in, %1#2 : tensor<i64>, tensor<24xf64>
  }
  return %0#1 : tensor<24xf64>
}

// CHECK:  func.func @inner_loop(%arg0: tensor<8xf64>, %arg1: tensor<24xf64>) -> tensor<24xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<8xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %4 = stablehlo.multiply %0, %3 : tensor<8xi64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT:   %6:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %5, %iterArg_3 = %arg1) : tensor<i64>, tensor<8xf64>, tensor<24xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %7 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %7 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %7 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:     %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT:     %9 = stablehlo.multiply %2, %8 : tensor<8xf64>
// CHECK-NEXT:     %10 = stablehlo.add %iterArg_2, %9 : tensor<8xf64>
// CHECK-NEXT:     %11 = stablehlo.broadcast_in_dim %iterArg, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:     %12 = stablehlo.add %4, %11 : tensor<8xi64>
// CHECK-NEXT:     %13 = stablehlo.reshape %12 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:     %14 = "stablehlo.scatter"(%iterArg_3, %13, %10) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<24xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<24xf64>
// CHECK-NEXT:     %15 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %15, %10, %14 : tensor<i64>, tensor<8xf64>, tensor<24xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %6#2 : tensor<24xf64>
// CHECK-NEXT: }

// -----

// The buffer goes into a nested loop twice: one copy is written, the other
// handed back unchanged and read after it. Only the first use is a link of
// the chain; the second is neither a link, nor a read of elements, nor the
// yield, so the loop is kept rather than batch a copy no chain follows.
func.func @forked_buffer(%v: tensor<4xf64>, %buf: tensor<4xf64>, %out: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %0:3 = stablehlo.while(%i = %c0, %b = %buf, %o = %out) : tensor<i64>, tensor<4xf64>, tensor<4xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %x1 = stablehlo.dynamic_slice %v, %i, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %1:3 = stablehlo.while(%j = %c0, %w = %b, %keep = %b) : tensor<i64>, tensor<4xf64>, tensor<4xf64>
     cond {
      %g = stablehlo.compare LT, %j, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %g : tensor<i1>
    } do {
      %w1 = stablehlo.dynamic_update_slice %w, %x1, %i : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
      %jn = stablehlo.add %j, %c1 : tensor<i64>
      stablehlo.return %jn, %w1, %keep : tensor<i64>, tensor<4xf64>, tensor<4xf64>
    }
    %r = stablehlo.dynamic_slice %1#2, %i, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %o1 = stablehlo.dynamic_update_slice %o, %r, %i : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
    %ni = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %ni, %1#1, %o1 : tensor<i64>, tensor<4xf64>, tensor<4xf64>
  }
  return %0#1, %0#2 : tensor<4xf64>, tensor<4xf64>
}

// CHECK:  func.func @forked_buffer(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:   %0:3 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg1, %iterArg_4 = %arg2) : tensor<i64>, tensor<4xf64>, tensor<4xf64> attributes {enzymexla.parallel}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %arg0, %iterArg, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %2:3 = stablehlo.while(%iterArg_5 = %c, %iterArg_6 = %iterArg_3, %iterArg_7 = %iterArg_3) : tensor<i64>, tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %6 = stablehlo.compare LT, %iterArg_5, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %6 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %6 = stablehlo.dynamic_update_slice %iterArg_6, %1, %iterArg : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:       %7 = stablehlo.add %iterArg_5, %c_0 : tensor<i64>
// CHECK-NEXT:       stablehlo.return %7, %6, %iterArg_7 : tensor<i64>, tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %3 = stablehlo.dynamic_slice %2#2, %iterArg, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %4 = stablehlo.dynamic_update_slice %iterArg_4, %3, %iterArg : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:     %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %5, %2#1, %4 : tensor<i64>, tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1, %0#2 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT: }

