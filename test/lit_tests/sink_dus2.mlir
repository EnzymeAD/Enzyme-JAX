// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=sink_dus" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s | FileCheck %s

// SinkDUS used to assert(DUS2) when walking back from a DUS's destination
// operand hit a value that isn't itself a dynamic_update_slice (here, an
// stablehlo.add). It should instead bail out of sinking for that candidate.

module {
  func.func @main(%215621: tensor<52x20x38xf32>, %215622: tensor<52x20x38xf32>) -> tensor<52x20x38xf32> {
    %cst_558 = stablehlo.constant dense<0.000000e+00> : tensor<52x18x36xf32>
    %cst_560 = stablehlo.constant dense<0.000000e+00> : tensor<50x18x36xf32>
    %c_56 = stablehlo.constant dense<1> : tensor<i32>
    %c_57 = stablehlo.constant dense<1> : tensor<i32>
    %c_iv0 = stablehlo.constant dense<0> : tensor<i64>
    %c_iv1 = stablehlo.constant dense<1> : tensor<i64>
    %c_n = stablehlo.constant dense<10> : tensor<i64>
    %loop:3 = stablehlo.while(%iterArg = %c_iv0, %iterArg_2814 = %215621, %iterArg_2815 = %cst_560) : tensor<i64>, tensor<52x20x38xf32>, tensor<50x18x36xf32>
    cond {
      %cond = stablehlo.compare LT, %iterArg, %c_n : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %112734 = stablehlo.slice %iterArg_2814 [1:51, 1:19, 1:37]
          : (tensor<52x20x38xf32>) -> tensor<50x18x36xf32>
      %215623 = stablehlo.add %215621, %215622 : tensor<52x20x38xf32>
      %215634 = stablehlo.dynamic_update_slice %215623, %cst_558, %c_56, %c_57, %c_57
          : (tensor<52x20x38xf32>, tensor<52x18x36xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<52x20x38xf32>
      %new_iv = stablehlo.add %iterArg, %c_iv1 : tensor<i64>
      stablehlo.return %new_iv, %215634, %112734 : tensor<i64>, tensor<52x20x38xf32>, tensor<50x18x36xf32>
    }

    return %loop#1 : tensor<52x20x38xf32>
  }
}

// CHECK-LABEL: func.func @main(%arg0: tensor<52x20x38xf32>, %arg1: tensor<52x20x38xf32>) -> tensor<52x20x38xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<52x18x36xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<50x18x36xf32>
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %0:3 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg0, %iterArg_5 = %cst_0) : tensor<i64>, tensor<52x20x38xf32>, tensor<50x18x36xf32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %1 = stablehlo.slice %iterArg_4 [1:51, 1:19, 1:37] : (tensor<52x20x38xf32>) -> tensor<50x18x36xf32>
// CHECK-NEXT:      %2 = stablehlo.add %arg0, %arg1 : tensor<52x20x38xf32>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %2, %cst, %c, %c, %c : (tensor<52x20x38xf32>, tensor<52x18x36xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<52x20x38xf32>
// CHECK-NEXT:      %4 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %4, %3, %1 : tensor<i64>, tensor<52x20x38xf32>, tensor<50x18x36xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#1 : tensor<52x20x38xf32>
// CHECK-NEXT:  }
