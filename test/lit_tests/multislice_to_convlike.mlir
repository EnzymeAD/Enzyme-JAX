// RUN: enzymexlamlir-opt %s --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.enzyme_hlo.self_add_to_convolution_like {parameter = false}

    } : !transform.any_op
    transform.yield 
  }
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func @multislice(%arg0: tensor<12x1024xi64> {enzymexla.memory_effects = [], sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<12x1019xi64>, tensor<12x1024xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<12x1024xi64>) -> tensor<1024x12xi64>
    %2 = stablehlo.slice %0 [1:1020, 0:12] : (tensor<1024x12xi64>) -> tensor<1019x12xi64>
    %3 = stablehlo.slice %0 [2:1021, 0:12] : (tensor<1024x12xi64>) -> tensor<1019x12xi64>
    %4 = stablehlo.slice %0 [3:1022, 0:12] : (tensor<1024x12xi64>) -> tensor<1019x12xi64>
    %5 = stablehlo.slice %0 [4:1023, 0:12] : (tensor<1024x12xi64>) -> tensor<1019x12xi64>
    %8 = stablehlo.add %2, %3 : tensor<1019x12xi64>
    %9 = stablehlo.add %8, %4 : tensor<1019x12xi64>
    %10 = stablehlo.add %9, %5 : tensor<1019x12xi64>
    %12 = stablehlo.transpose %10, dims = [1, 0] : (tensor<1019x12xi64>) -> tensor<12x1019xi64>
    %13 = stablehlo.transpose %0, dims = [1, 0] : (tensor<1024x12xi64>) -> tensor<12x1024xi64>
    return %12, %13 : tensor<12x1019xi64>, tensor<12x1024xi64>
  }
}

// CHECK: %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<12x1024xi64>) -> tensor<1024x12xi64>
// CHECK-NEXT:  %1 = stablehlo.slice %0 [1:1023, 0:12] : (tensor<1024x12xi64>) -> tensor<1022x12xi64>
// CHECK-NEXT:  %2 = "stablehlo.reduce_window"(%1, %c) <{window_dimensions = array<i64: 4, 1>}> ({
// CHECK-NEXT:  ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
// CHECK-NEXT:    %5 = stablehlo.add %arg1, %arg2 : tensor<i64>
// CHECK-NEXT:    stablehlo.return %5 : tensor<i64>
// CHECK-NEXT:  }) : (tensor<1022x12xi64>, tensor<i64>) -> tensor<1019x12xi64>
// CHECK-NEXT:  %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<1019x12xi64>) -> tensor<12x1019xi64>
// CHECK-NEXT:  %4 = stablehlo.transpose %0, dims = [1, 0] : (tensor<1024x12xi64>) -> tensor<12x1024xi64>
// CHECK-NEXT:  return %3, %4 : tensor<12x1019xi64>, tensor<12x1024xi64>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.enzyme_hlo.self_add_to_convolution_like {parameter = false}

    } : !transform.any_op
    transform.yield 
  }
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func @multislice(%arg0: tensor<12x1024xi64> {enzymexla.memory_effects = [], sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<12x1019xi64>, tensor<12x1024xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<12x1024xi64>) -> tensor<1024x12xi64>
    %1 = stablehlo.slice %0 [0:1019, 0:12] : (tensor<1024x12xi64>) -> tensor<1019x12xi64>
    %2 = stablehlo.slice %0 [1:1020, 0:12] : (tensor<1024x12xi64>) -> tensor<1019x12xi64>
    %3 = stablehlo.slice %0 [2:1021, 0:12] : (tensor<1024x12xi64>) -> tensor<1019x12xi64>
    %4 = stablehlo.slice %0 [3:1022, 0:12] : (tensor<1024x12xi64>) -> tensor<1019x12xi64>
    %5 = stablehlo.slice %0 [4:1023, 0:12] : (tensor<1024x12xi64>) -> tensor<1019x12xi64>
    %6 = stablehlo.slice %0 [5:1024, 0:12] : (tensor<1024x12xi64>) -> tensor<1019x12xi64>
    %7 = stablehlo.add %1, %2 : tensor<1019x12xi64>
    %8 = stablehlo.add %2, %3 : tensor<1019x12xi64>
    %9 = stablehlo.add %8, %4 : tensor<1019x12xi64>
    %10 = stablehlo.add %9, %5 : tensor<1019x12xi64>
    %11 = stablehlo.add %10, %6 : tensor<1019x12xi64>
    %12 = stablehlo.transpose %10, dims = [1, 0] : (tensor<1019x12xi64>) -> tensor<12x1019xi64>
    %13 = stablehlo.transpose %0, dims = [1, 0] : (tensor<1024x12xi64>) -> tensor<12x1024xi64>
    return %12, %13 : tensor<12x1019xi64>, tensor<12x1024xi64>
  }
}

// CHECK: %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<12x1024xi64>) -> tensor<1024x12xi64>
// CHECK-NEXT:  %1 = stablehlo.slice %0 [1:1023, 0:12] : (tensor<1024x12xi64>) -> tensor<1022x12xi64>
// CHECK-NEXT:  %2 = "stablehlo.reduce_window"(%1, %c) <{window_dimensions = array<i64: 4, 1>}> ({
// CHECK-NEXT:  ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
// CHECK-NEXT:    %5 = stablehlo.add %arg1, %arg2 : tensor<i64>
// CHECK-NEXT:    stablehlo.return %5 : tensor<i64>
// CHECK-NEXT:  }) : (tensor<1022x12xi64>, tensor<i64>) -> tensor<1019x12xi64>
// CHECK-NEXT:  %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<1019x12xi64>) -> tensor<12x1019xi64>
// CHECK-NEXT:  %4 = stablehlo.transpose %0, dims = [1, 0] : (tensor<1024x12xi64>) -> tensor<12x1024xi64>
// CHECK-NEXT:  return %3, %4 : tensor<12x1019xi64>, tensor<12x1024xi64>
