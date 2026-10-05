// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// Every iteration halves its value until it is at most one, as many times
// as it takes: the nested loop runs while any iteration's condition holds,
// carrying which iterations are still in it, and an iteration that has left
// keeps its values.
func.func @halve(%v: tensor<4xf64>, %out: tensor<4xf64>) -> tensor<4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %one = stablehlo.constant dense<1.0> : tensor<f64>
  %half = stablehlo.constant dense<0.5> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %o = %out) : tensor<i64>, tensor<4xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %x1 = stablehlo.dynamic_slice %v, %i, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %x0 = stablehlo.reshape %x1 : (tensor<1xf64>) -> tensor<f64>
    %1:2 = stablehlo.while(%x = %x0, %n = %c0) : tensor<f64>, tensor<i64>
     cond {
      %g = stablehlo.compare GT, %x, %one : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %g : tensor<i1>
    } do {
      %xh = stablehlo.multiply %x, %half : tensor<f64>
      %nn = stablehlo.add %n, %c1 : tensor<i64>
      stablehlo.return %xh, %nn : tensor<f64>, tensor<i64>
    }
    %nf = stablehlo.convert %1#1 : (tensor<i64>) -> tensor<f64>
    %r = stablehlo.add %1#0, %nf : tensor<f64>
    %rr = stablehlo.reshape %r : (tensor<f64>) -> tensor<1xf64>
    %o1 = stablehlo.dynamic_update_slice %o, %rr, %i : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
    %ni = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %ni, %o1 : tensor<i64>, tensor<4xf64>
  }
  return %0#1 : tensor<4xf64>
}

// CHECK:  func.func @halve(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:   %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:   %5 = stablehlo.compare GT, %2, %4 : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
// CHECK-NEXT:   %6:3 = stablehlo.while(%iterArg = %2, %iterArg_4 = %3, %iterArg_5 = %5) : tensor<4xf64>, tensor<4xi64>, tensor<4xi1>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %13 = stablehlo.reduce(%iterArg_5 init: %c_0) applies stablehlo.or across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %13 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %13 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:     %14 = stablehlo.multiply %iterArg, %13 : tensor<4xf64>
// CHECK-NEXT:     %15 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:     %16 = stablehlo.add %iterArg_4, %15 : tensor<4xi64>
// CHECK-NEXT:     %17 = stablehlo.select %iterArg_5, %14, %iterArg : tensor<4xi1>, tensor<4xf64>
// CHECK-NEXT:     %18 = stablehlo.select %iterArg_5, %16, %iterArg_4 : tensor<4xi1>, tensor<4xi64>
// CHECK-NEXT:     %19 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:     %20 = stablehlo.compare GT, %17, %19 : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
// CHECK-NEXT:     %21 = stablehlo.and %iterArg_5, %20 : tensor<4xi1>
// CHECK-NEXT:     stablehlo.return %17, %18, %21 : tensor<4xf64>, tensor<4xi64>, tensor<4xi1>
// CHECK-NEXT:   }
// CHECK-NEXT:   %7 = stablehlo.convert %6#1 : (tensor<4xi64>) -> tensor<4xf64>
// CHECK-NEXT:   %8 = stablehlo.add %6#0, %7 : tensor<4xf64>
// CHECK-NEXT:   %9 = stablehlo.reshape %8 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:   %10 = stablehlo.clamp %c_1, %0, %c : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %11 = stablehlo.reshape %10 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:   %12 = "stablehlo.scatter"(%arg1, %11, %9) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<4xf64>, tensor<4x1xi64>, tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:   return %12 : tensor<4xf64>
// CHECK-NEXT: }


// -----

// The nested loop writes the carried buffer as it goes, up to four times per
// iteration: an iteration that has left the loop sends its rows out of the
// buffer, where the scatter drops them.
func.func @trace(%v: tensor<4xf64>, %out: tensor<16xf64>) -> tensor<16xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %one = stablehlo.constant dense<1.0> : tensor<f64>
  %half = stablehlo.constant dense<0.5> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %o = %out) : tensor<i64>, tensor<16xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %x1 = stablehlo.dynamic_slice %v, %i, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %x0 = stablehlo.reshape %x1 : (tensor<1xf64>) -> tensor<f64>
    %base = stablehlo.multiply %i, %c4 : tensor<i64>
    %1:3 = stablehlo.while(%x = %x0, %n = %c0, %ob = %o) : tensor<f64>, tensor<i64>, tensor<16xf64>
     cond {
      %g = stablehlo.compare GT, %x, %one : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %lt = stablehlo.compare LT, %n, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %both = stablehlo.and %g, %lt : tensor<i1>
      stablehlo.return %both : tensor<i1>
    } do {
      %xh = stablehlo.multiply %x, %half : tensor<f64>
      %at = stablehlo.add %base, %n : tensor<i64>
      %xr = stablehlo.reshape %xh : (tensor<f64>) -> tensor<1xf64>
      %ob1 = stablehlo.dynamic_update_slice %ob, %xr, %at : (tensor<16xf64>, tensor<1xf64>, tensor<i64>) -> tensor<16xf64>
      %nn = stablehlo.add %n, %c1 : tensor<i64>
      stablehlo.return %xh, %nn, %ob1 : tensor<f64>, tensor<i64>, tensor<16xf64>
    }
    %ni = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %ni, %1#2 : tensor<i64>, tensor<16xf64>
  }
  return %0#1 : tensor<16xf64>
}

// CHECK:  func.func @trace(%arg0: tensor<4xf64>, %arg1: tensor<16xf64>) -> tensor<16xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<16> : tensor<4x1xi64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<15> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_4 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:   %cst_5 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %4 = stablehlo.multiply %0, %3 : tensor<4xi64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:   %7 = stablehlo.compare GT, %2, %6 : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
// CHECK-NEXT:   %8 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %9 = stablehlo.compare LT, %5, %8 : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
// CHECK-NEXT:   %10 = stablehlo.and %7, %9 : tensor<4xi1>
// CHECK-NEXT:   %11:4 = stablehlo.while(%iterArg = %2, %iterArg_6 = %5, %iterArg_7 = %arg1, %iterArg_8 = %10) : tensor<4xf64>, tensor<4xi64>, tensor<16xf64>, tensor<4xi1>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %12 = stablehlo.reduce(%iterArg_8 init: %c_1) applies stablehlo.or across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %12 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %12 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:     %13 = stablehlo.multiply %iterArg, %12 : tensor<4xf64>
// CHECK-NEXT:     %14 = stablehlo.add %4, %iterArg_6 : tensor<4xi64>
// CHECK-NEXT:     %15 = stablehlo.reshape %13 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:     %16 = stablehlo.clamp %c_2, %14, %c_0 : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:     %17 = stablehlo.reshape %16 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:     %18 = stablehlo.broadcast_in_dim %iterArg_8, dims = [0] : (tensor<4xi1>) -> tensor<4x1xi1>
// CHECK-NEXT:     %19 = stablehlo.select %18, %17, %c : tensor<4x1xi1>, tensor<4x1xi64>
// CHECK-NEXT:     %20 = "stablehlo.scatter"(%iterArg_7, %19, %15) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<16xf64>, tensor<4x1xi64>, tensor<4x1xf64>) -> tensor<16xf64>
// CHECK-NEXT:     %21 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:     %22 = stablehlo.add %iterArg_6, %21 : tensor<4xi64>
// CHECK-NEXT:     %23 = stablehlo.select %iterArg_8, %13, %iterArg : tensor<4xi1>, tensor<4xf64>
// CHECK-NEXT:     %24 = stablehlo.select %iterArg_8, %22, %iterArg_6 : tensor<4xi1>, tensor<4xi64>
// CHECK-NEXT:     %25 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:     %26 = stablehlo.compare GT, %23, %25 : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
// CHECK-NEXT:     %27 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:     %28 = stablehlo.compare LT, %24, %27 : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
// CHECK-NEXT:     %29 = stablehlo.and %26, %28 : tensor<4xi1>
// CHECK-NEXT:     %30 = stablehlo.and %iterArg_8, %29 : tensor<4xi1>
// CHECK-NEXT:     stablehlo.return %23, %24, %20, %30 : tensor<4xf64>, tensor<4xi64>, tensor<16xf64>, tensor<4xi1>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %11#2 : tensor<16xf64>
// CHECK-NEXT: }


// -----

// A loop whose trip count varies is not followed into another such loop.
// Kept.
func.func @masked_in_masked(%v: tensor<4xf64>, %out: tensor<4xf64>) -> tensor<4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %one = stablehlo.constant dense<1.0> : tensor<f64>
  %half = stablehlo.constant dense<0.5> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %o = %out) : tensor<i64>, tensor<4xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %x1 = stablehlo.dynamic_slice %v, %i, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %x0 = stablehlo.reshape %x1 : (tensor<1xf64>) -> tensor<f64>
    %1 = stablehlo.while(%x = %x0) : tensor<f64>
     cond {
      %g = stablehlo.compare GT, %x, %one : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %g : tensor<i1>
    } do {
      %2 = stablehlo.while(%y = %x) : tensor<f64>
       cond {
        %h = stablehlo.compare GT, %y, %x : (tensor<f64>, tensor<f64>) -> tensor<i1>
        stablehlo.return %h : tensor<i1>
      } do {
        %yh = stablehlo.multiply %y, %half : tensor<f64>
        stablehlo.return %yh : tensor<f64>
      }
      %xh = stablehlo.multiply %2, %half : tensor<f64>
      stablehlo.return %xh : tensor<f64>
    }
    %rr = stablehlo.reshape %1 : (tensor<f64>) -> tensor<1xf64>
    %o1 = stablehlo.dynamic_update_slice %o, %rr, %i : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
    %ni = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %ni, %o1 : tensor<i64>, tensor<4xf64>
  }
  return %0#1 : tensor<4xf64>
}

// CHECK:  func.func @masked_in_masked(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:   %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg1) : tensor<i64>, tensor<4xf64> attributes {enzymexla.parallel}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %arg0, %iterArg, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:     %3 = stablehlo.while(%iterArg_4 = %2) : tensor<f64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %7 = stablehlo.compare GT, %iterArg_4, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %7 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %7 = stablehlo.while(%iterArg_5 = %iterArg_4) : tensor<f64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:         %9 = stablehlo.compare GT, %iterArg_5, %iterArg_4 : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %9 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %9 = stablehlo.multiply %iterArg_5, %cst_2 : tensor<f64>
// CHECK-NEXT:         stablehlo.return %9 : tensor<f64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %8 = stablehlo.multiply %7, %cst_2 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %8 : tensor<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %4 = stablehlo.reshape %3 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:     %5 = stablehlo.dynamic_update_slice %iterArg_3, %4, %iterArg : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:     %6 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %6, %5 : tensor<i64>, tensor<4xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<4xf64>
// CHECK-NEXT: }


// -----

// The nested loop's limit is an argument: not known here, but the same in
// every iteration, so the nested loop runs as many times in each and is
// batched as it is, with no mask.
func.func @uniform_limit(%v: tensor<4xf64>, %out: tensor<4xf64>, %lim: tensor<i64>) -> tensor<4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %half = stablehlo.constant dense<0.5> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %o = %out) : tensor<i64>, tensor<4xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %x1 = stablehlo.dynamic_slice %v, %i, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %x0 = stablehlo.reshape %x1 : (tensor<1xf64>) -> tensor<f64>
    %1:2 = stablehlo.while(%j = %c0, %x = %x0) : tensor<i64>, tensor<f64>
     cond {
      %g = stablehlo.compare LT, %j, %lim : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %g : tensor<i1>
    } do {
      %xh = stablehlo.multiply %x, %half : tensor<f64>
      %jn = stablehlo.add %j, %c1 : tensor<i64>
      stablehlo.return %jn, %xh : tensor<i64>, tensor<f64>
    }
    %rr = stablehlo.reshape %1#1 : (tensor<f64>) -> tensor<1xf64>
    %o1 = stablehlo.dynamic_update_slice %o, %rr, %i : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
    %ni = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %ni, %o1 : tensor<i64>, tensor<4xf64>
  }
  return %0#1 : tensor<4xf64>
}

// CHECK:  func.func @uniform_limit(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<i64>) -> tensor<4xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %3:2 = stablehlo.while(%iterArg = %c_0, %iterArg_2 = %2) : tensor<i64>, tensor<4xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %8 = stablehlo.compare LT, %iterArg, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %8 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:     %9 = stablehlo.multiply %iterArg_2, %8 : tensor<4xf64>
// CHECK-NEXT:     %10 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %10, %9 : tensor<i64>, tensor<4xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   %4 = stablehlo.reshape %3#1 : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:   %5 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %6 = stablehlo.reshape %5 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:   %7 = "stablehlo.scatter"(%arg1, %6, %4) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<4xf64>, tensor<4x1xi64>, tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:   return %7 : tensor<4xf64>
// CHECK-NEXT: }


// -----

// The nested loop's limit varies with the iteration but the raiser bounded
// it: the loop unrolls to its most trips instead (WhileUnroll), so it is
// not batched here, and the parallel loop is kept for now.
func.func @bounded_limit(%v: tensor<4xf64>, %out: tensor<4xf64>) -> tensor<4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %half = stablehlo.constant dense<0.5> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %o = %out) : tensor<i64>, tensor<4xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %x1 = stablehlo.dynamic_slice %v, %i, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
    %x0 = stablehlo.reshape %x1 : (tensor<1xf64>) -> tensor<f64>
    %lim = stablehlo.add %i, %c1 {enzymexla.bounds = [[1, 4]]} : tensor<i64>
    %1:2 = stablehlo.while(%j = %c0, %x = %x0) : tensor<i64>, tensor<f64>
     cond {
      %g = stablehlo.compare LT, %j, %lim : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %g : tensor<i1>
    } do {
      %xh = stablehlo.multiply %x, %half : tensor<f64>
      %jn = stablehlo.add %j, %c1 : tensor<i64>
      stablehlo.return %jn, %xh : tensor<i64>, tensor<f64>
    }
    %rr = stablehlo.reshape %1#1 : (tensor<f64>) -> tensor<1xf64>
    %o1 = stablehlo.dynamic_update_slice %o, %rr, %i : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
    %ni = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %ni, %o1 : tensor<i64>, tensor<4xf64>
  }
  return %0#1 : tensor<4xf64>
}

// CHECK:  func.func @bounded_limit(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg1) : tensor<i64>, tensor<4xf64> attributes {enzymexla.parallel}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %arg0, %iterArg, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:     %3 = stablehlo.add %iterArg, %c_0 {enzymexla.bounds = {{\[\[}}1, 4{{\]\]}}} : tensor<i64>
// CHECK-NEXT:     %4:2 = stablehlo.while(%iterArg_3 = %c, %iterArg_4 = %2) : tensor<i64>, tensor<f64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %8 = stablehlo.compare LT, %iterArg_3, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %8 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %8 = stablehlo.multiply %iterArg_4, %cst : tensor<f64>
// CHECK-NEXT:       %9 = stablehlo.add %iterArg_3, %c_0 : tensor<i64>
// CHECK-NEXT:       stablehlo.return %9, %8 : tensor<i64>, tensor<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %5 = stablehlo.reshape %4#1 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:     %6 = stablehlo.dynamic_update_slice %iterArg_2, %5, %iterArg : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:     %7 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %7, %6 : tensor<i64>, tensor<4xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<4xf64>
// CHECK-NEXT: }

