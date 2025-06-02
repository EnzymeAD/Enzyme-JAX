// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=while_dus;dynamic_slice_to_static;slice_of_dynamic_update" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s | FileCheck %s

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
      %9222 = "test.update"() : () -> tensor<10x24x48xf32>
      %9223 = stablehlo.dynamic_update_slice %iterArg_291, %9222, %c_266, %c_266, %c_266 : (tensor<24x38x62xf32>, tensor<10x24x48xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x38x62xf32>
      %9224 = stablehlo.add %iterArg, %c_270 : tensor<i64>
      stablehlo.return %9224, %9223 : tensor<i64>, tensor<24x38x62xf32>
    }
    return %0#1 : tensor<24x38x62xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<24x38x62xf32>, %arg1: tensor<i64>) -> tensor<24x38x62xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [7:17, 7:31, 7:55] : (tensor<24x38x62xf32>) -> tensor<10x24x48xf32>
// CHECK-NEXT:    %1:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %0) : tensor<i64>, tensor<10x24x48xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %3 = stablehlo.compare  LT, %iterArg, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %3 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %3 = stablehlo.slice %iterArg_2 [0:10, 0:24, 0:48] : (tensor<10x24x48xf32>) -> tensor<10x24x48xf32>
// CHECK-NEXT:      "test.use"(%3) : (tensor<10x24x48xf32>) -> ()
// CHECK-NEXT:      %4 = "test.update"() : () -> tensor<10x24x48xf32>
// CHECK-NEXT:      %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %5, %4 : tensor<i64>, tensor<10x24x48xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %arg0, %1#1, %c_1, %c_1, %c_1 : (tensor<24x38x62xf32>, tensor<10x24x48xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x38x62xf32>
// CHECK-NEXT:    return %2 : tensor<24x38x62xf32>
// CHECK-NEXT:  }
