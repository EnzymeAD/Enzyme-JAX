// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=CHECK-LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=CHECK-TUPLE %s

func.func @main(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
  %0 = stablehlo.abs %arg0 : tensor<2xf64>
  %1 = stablehlo.abs %arg1 : tensor<2xf64>
  return %0, %1 : tensor<2xf64>, tensor<2xf64>
}

// CHECK-LABEL: func.func @main
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: stablehlo.reduce(%{{.*}} init: %[[CST]])
// CHECK: stablehlo.reduce(%{{.*}} init: %[[CST]])

// CHECK-LAST-LABEL: func.func @main
// CHECK-LAST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-LAST: stablehlo.reduce(%{{.*}} init: %[[CST]])
// CHECK-LAST: stablehlo.reduce(%{{.*}} init: %[[CST]])

// CHECK-TUPLE-LABEL: func.func @main
// CHECK-TUPLE-NOT: stablehlo.reduce
