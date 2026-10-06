// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reduce_slice_to_batch" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A reduce with two results (a value picked where a mask holds, and whether
// any lane held) over slices of the same buffers: the batch op would carry
// one result only, so the slices stay as they are.
func.func @variadic_reduce(%x: tensor<2x4x3xf64>, %m: tensor<2x4x3xi1>) -> (tensor<3xf64>, tensor<3xi1>, tensor<3xf64>, tensor<3xi1>) {
  %cst = stablehlo.constant dense<0.0> : tensor<f64>
  %f = stablehlo.constant dense<false> : tensor<i1>
  %x0 = stablehlo.slice %x [0:1, 0:4, 0:3] : (tensor<2x4x3xf64>) -> tensor<1x4x3xf64>
  %x1 = stablehlo.slice %x [1:2, 0:4, 0:3] : (tensor<2x4x3xf64>) -> tensor<1x4x3xf64>
  %m0 = stablehlo.slice %m [0:1, 0:4, 0:3] : (tensor<2x4x3xi1>) -> tensor<1x4x3xi1>
  %m1 = stablehlo.slice %m [1:2, 0:4, 0:3] : (tensor<2x4x3xi1>) -> tensor<1x4x3xi1>
  %a0 = stablehlo.reshape %x0 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
  %a1 = stablehlo.reshape %x1 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
  %b0 = stablehlo.reshape %m0 : (tensor<1x4x3xi1>) -> tensor<4x3xi1>
  %b1 = stablehlo.reshape %m1 : (tensor<1x4x3xi1>) -> tensor<4x3xi1>
  %r0:2 = stablehlo.reduce(%a0 init: %cst), (%b0 init: %f) across dimensions = [0] : (tensor<4x3xf64>, tensor<4x3xi1>, tensor<f64>, tensor<i1>) -> (tensor<3xf64>, tensor<3xi1>)
   reducer(%p: tensor<f64>, %q: tensor<f64>) (%u: tensor<i1>, %v: tensor<i1>) {
    %s = stablehlo.select %u, %p, %q : tensor<i1>, tensor<f64>
    %o = stablehlo.or %u, %v : tensor<i1>
    stablehlo.return %s, %o : tensor<f64>, tensor<i1>
  }
  %r1:2 = stablehlo.reduce(%a1 init: %cst), (%b1 init: %f) across dimensions = [0] : (tensor<4x3xf64>, tensor<4x3xi1>, tensor<f64>, tensor<i1>) -> (tensor<3xf64>, tensor<3xi1>)
   reducer(%p: tensor<f64>, %q: tensor<f64>) (%u: tensor<i1>, %v: tensor<i1>) {
    %s = stablehlo.select %u, %p, %q : tensor<i1>, tensor<f64>
    %o = stablehlo.or %u, %v : tensor<i1>
    stablehlo.return %s, %o : tensor<f64>, tensor<i1>
  }
  return %r0#0, %r0#1, %r1#0, %r1#1 : tensor<3xf64>, tensor<3xi1>, tensor<3xf64>, tensor<3xi1>
}

// CHECK:  func.func @variadic_reduce(%arg0: tensor<2x4x3xf64>, %arg1: tensor<2x4x3xi1>) -> (tensor<3xf64>, tensor<3xi1>, tensor<3xf64>, tensor<3xi1>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:1, 0:4, 0:3] : (tensor<2x4x3xf64>) -> tensor<1x4x3xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %arg0 [1:2, 0:4, 0:3] : (tensor<2x4x3xf64>) -> tensor<1x4x3xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %arg1 [0:1, 0:4, 0:3] : (tensor<2x4x3xi1>) -> tensor<1x4x3xi1>
// CHECK-NEXT:   %3 = stablehlo.slice %arg1 [1:2, 0:4, 0:3] : (tensor<2x4x3xi1>) -> tensor<1x4x3xi1>
// CHECK-NEXT:   %4 = stablehlo.reshape %0 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:   %5 = stablehlo.reshape %1 : (tensor<1x4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:   %6 = stablehlo.reshape %2 : (tensor<1x4x3xi1>) -> tensor<4x3xi1>
// CHECK-NEXT:   %7 = stablehlo.reshape %3 : (tensor<1x4x3xi1>) -> tensor<4x3xi1>
// CHECK-NEXT:   %8:2 = stablehlo.reduce(%4 init: %cst), (%6 init: %c) across dimensions = [0] : (tensor<4x3xf64>, tensor<4x3xi1>, tensor<f64>, tensor<i1>) -> (tensor<3xf64>, tensor<3xi1>)
// CHECK-NEXT:    reducer(%arg2: tensor<f64>, %arg4: tensor<f64>) (%arg3: tensor<i1>, %arg5: tensor<i1>)  {
// CHECK-NEXT:     %10 = stablehlo.select %arg3, %arg2, %arg4 : tensor<i1>, tensor<f64>
// CHECK-NEXT:     %11 = stablehlo.or %arg3, %arg5 : tensor<i1>
// CHECK-NEXT:     stablehlo.return %10, %11 : tensor<f64>, tensor<i1>
// CHECK-NEXT:   }
// CHECK-NEXT:   %9:2 = stablehlo.reduce(%5 init: %cst), (%7 init: %c) across dimensions = [0] : (tensor<4x3xf64>, tensor<4x3xi1>, tensor<f64>, tensor<i1>) -> (tensor<3xf64>, tensor<3xi1>)
// CHECK-NEXT:    reducer(%arg2: tensor<f64>, %arg4: tensor<f64>) (%arg3: tensor<i1>, %arg5: tensor<i1>)  {
// CHECK-NEXT:     %10 = stablehlo.select %arg3, %arg2, %arg4 : tensor<i1>, tensor<f64>
// CHECK-NEXT:     %11 = stablehlo.or %arg3, %arg5 : tensor<i1>
// CHECK-NEXT:     stablehlo.return %10, %11 : tensor<f64>, tensor<i1>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %8#0, %8#1, %9#0, %9#1 : tensor<3xf64>, tensor<3xi1>, tensor<3xf64>, tensor<3xi1>
// CHECK-NEXT: }
