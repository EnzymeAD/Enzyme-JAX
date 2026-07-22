// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s

// CHECK: func.func private @"Const{typeof(myfunc)}_autodiff"

module @reactant_gradmyfunc attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"Const{typeof(myfunc)}_autodiff"(%arg0: tensor<3xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<3xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<3xf64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<10> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<3xf64> attributes {enzyme.disable_mincut,
                                                                                                       enzymexla.checkpoint_period = 3 : i64,
                                                                                                       enzymexla.binomial_checkpointing,
                                                                                                       enzymexla.enable_checkpointing = true}
    cond {
      %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_1 {enzymexla.bounds = [[1, 10]]} : tensor<i64>
      %2:2 = stablehlo.while(%iterArg_3 = %c, %iterArg_4 = %iterArg_2) : tensor<i64>, tensor<3xf64> attributes {enzyme.disable_mincut}
      cond {
        %3 = stablehlo.compare LT, %iterArg_3, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %3 : tensor<i1>
      } do {
        %3 = stablehlo.add %iterArg_3, %c_1 {enzymexla.bounds = [[1, 10]]} : tensor<i64>
        %4 = stablehlo.multiply %cst, %iterArg_4 : tensor<3xf64>
        %5 = stablehlo.add %iterArg_4, %4 : tensor<3xf64>
        stablehlo.return %3, %5 : tensor<i64>, tensor<3xf64>
      }
      stablehlo.return %1, %2#1 : tensor<i64>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }
  func.func @main(%arg0: tensor<3xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 2 : i32}, %arg1: tensor<3xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 1 : i32}) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3xf64>
    %0 = stablehlo.add %cst, %arg1 : tensor<3xf64>
    %1:2 = enzyme.autodiff @"Const{typeof(myfunc)}_autodiff"(%arg0, %0) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>]} : (tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>)
    return %1#0, %1#1, %arg0 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
  }
}
