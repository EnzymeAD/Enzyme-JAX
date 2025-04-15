// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=while_extend;while_wrap" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s

module @"reactant_loop!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main() -> tensor<1x39x62xf32> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<7> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0 = "test.get"() : () -> tensor<1x39x62xf32>
    %1 = "test.get"() : () -> tensor<i64>
    %2:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %0) : tensor<i64>, tensor<1x39x62xf32>
     cond {
      %3 = stablehlo.compare  LT, %iterArg, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = "test.get"() : () -> tensor<1x24x48xf32>
      %4 = stablehlo.dynamic_update_slice %iterArg_2, %3, %c_1, %c_0, %c_0 : (tensor<1x39x62xf32>, tensor<1x24x48xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x39x62xf32>
      %5 = stablehlo.slice %4 [0:1, 0:39, 7:55] : (tensor<1x39x62xf32>) -> tensor<1x39x48xf32>
      %6 = "enzymexla.wrap"(%5) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 7 : i64}> : (tensor<1x39x48xf32>) -> tensor<1x39x62xf32>
      %7 = stablehlo.add %iterArg, %c : tensor<i64>
      stablehlo.return %7, %6 : tensor<i64>, tensor<1x39x62xf32>
    }
    return %2#1 : tensor<1x39x62xf32>
  }
}

// CHECK:  func.func @main() -> tensor<1x39x62xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = "test.get"() : () -> tensor<1x39x62xf32>
// CHECK-NEXT:    %1 = "test.get"() : () -> tensor<i64>
// CHECK-NEXT:    %2 = stablehlo.slice %0 [0:1, 0:39, 7:55] : (tensor<1x39x62xf32>) -> tensor<1x39x48xf32>
// CHECK-NEXT:    %3:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %2) : tensor<i64>, tensor<1x39x48xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %5 = stablehlo.compare  LT, %iterArg, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %5 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %5 = "enzymexla.wrap"(%iterArg_2) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 7 : i64}> : (tensor<1x39x48xf32>) -> tensor<1x39x62xf32>
// CHECK-NEXT:      %6 = "test.get"() : () -> tensor<1x24x48xf32>
// CHECK-NEXT:      %7 = stablehlo.dynamic_update_slice %5, %6, %c_1, %c_0, %c_0 : (tensor<1x39x62xf32>, tensor<1x24x48xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x39x62xf32>
// CHECK-NEXT:      %8 = stablehlo.slice %7 [0:1, 0:39, 7:55] : (tensor<1x39x62xf32>) -> tensor<1x39x48xf32>
// CHECK-NEXT:      %9 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:      stablehlo.return %9, %8 : tensor<i64>, tensor<1x39x48xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %4 = "enzymexla.wrap"(%3#1) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 7 : i64}> : (tensor<1x39x48xf32>) -> tensor<1x39x62xf32>
// CHECK-NEXT:    return %4 : tensor<1x39x62xf32>
// CHECK-NEXT:  }