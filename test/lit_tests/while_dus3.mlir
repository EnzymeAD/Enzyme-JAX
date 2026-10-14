// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=while_dus;dynamic_slice_to_static;slice_of_dynamic_update" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s --enzyme-hlo-opt | FileCheck %s

module @"reactant_loop!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg31: tensor<24x38x62xf32>, %arg2 : tensor<i64>) -> (tensor<24x38x62xf32>) {
    %c_271 = stablehlo.constant dense<0> : tensor<i64>
    %c_270 = stablehlo.constant dense<1> : tensor<i64>
    %c_266 = stablehlo.constant dense<7> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_271, %iterArg_291 = %arg31) : tensor<i64>, tensor<24x38x62xf32>
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %108 = stablehlo.slice %iterArg_291 [7:17, 7:31, 7:55] : (tensor<24x38x62xf32>) -> tensor<10x24x48xf32>
      "test.use"(%108) : (tensor<10x24x48xf32>) -> ()
      %9222 = "test.update"() : () -> tensor<24x24x62xf32>

      %10860 = stablehlo.slice %iterArg_291 [0:24, 0:14, 0:62] : (tensor<24x38x62xf32>) -> tensor<24x14x62xf32>
      %9223 = stablehlo.concatenate %10860, %9222, dim = 1 : (tensor<24x14x62xf32>, tensor<24x24x62xf32>) -> tensor<24x38x62xf32>

      %9224 = stablehlo.add %iterArg, %c_270 : tensor<i64>
      stablehlo.return %9224, %9223 : tensor<i64>, tensor<24x38x62xf32>
    }
    return %0#1 : tensor<24x38x62xf32>
  }
}

// Only the updated trailing 24 columns remain loop-carried. The leading 14
// columns are recovered from the input, including for the slice inside the
// body. Condition carrying happens after this buffer-shape optimization.
// CHECK-LABEL: func.func @main(%arg0: tensor<24x38x62xf32>, %arg1: tensor<i64>) -> tensor<24x38x62xf32> {
// CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[ONE:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[INITIAL:.*]] = stablehlo.slice %arg0 [0:24, 14:38, 0:62] : (tensor<24x38x62xf32>) -> tensor<24x24x62xf32>
// CHECK-NEXT: %[[FIRST_TEST:.*]] = stablehlo.compare LT, %[[ZERO]], %arg1
// CHECK-NEXT: %[[WHILE:.*]]:3 = stablehlo.while(%[[I:.*]] = %[[ZERO]], %[[DATA:.*]] = %[[INITIAL]], %[[PRED:.*]] = %[[FIRST_TEST]]) : tensor<i64>, tensor<24x24x62xf32>, tensor<i1>
// CHECK-NEXT: cond {
// CHECK-NEXT: stablehlo.return %[[PRED]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[LEFT:.*]] = stablehlo.slice %arg0 [7:17, 7:14, 7:55] : (tensor<24x38x62xf32>) -> tensor<10x7x48xf32>
// CHECK-NEXT: %[[RIGHT:.*]] = stablehlo.slice %[[DATA]] [7:17, 0:17, 7:55] : (tensor<24x24x62xf32>) -> tensor<10x17x48xf32>
// CHECK-NEXT: %[[SLICE:.*]] = stablehlo.concatenate %[[LEFT]], %[[RIGHT]], dim = 1 : (tensor<10x7x48xf32>, tensor<10x17x48xf32>) -> tensor<10x24x48xf32>
// CHECK-NEXT: "test.use"(%[[SLICE]]) : (tensor<10x24x48xf32>) -> ()
// CHECK-NEXT: %[[UPDATE:.*]] = "test.update"() : () -> tensor<24x24x62xf32>
// CHECK-NEXT: %[[NEXT_I:.*]] = stablehlo.add %[[I]], %[[ONE]] : tensor<i64>
// CHECK-NEXT: %[[NEXT_TEST:.*]] = stablehlo.compare LT, %[[NEXT_I]], %arg1
// CHECK-NEXT: stablehlo.return %[[NEXT_I]], %[[UPDATE]], %[[NEXT_TEST]] : tensor<i64>, tensor<24x24x62xf32>, tensor<i1>
// CHECK-NEXT: }
// CHECK-NEXT: %[[PREFIX:.*]] = stablehlo.slice %arg0 [0:24, 0:14, 0:62] : (tensor<24x38x62xf32>) -> tensor<24x14x62xf32>
// CHECK-NEXT: %[[RESULT:.*]] = stablehlo.concatenate %[[PREFIX]], %[[WHILE]]#1, dim = 1 : (tensor<24x14x62xf32>, tensor<24x24x62xf32>) -> tensor<24x38x62xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<24x38x62xf32>
// CHECK-NEXT: }
