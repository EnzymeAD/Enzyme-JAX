// RUN: enzymexlamlir-opt %s  --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --enzyme-hlo-opt | FileCheck %s

module @reactant_gradmyfunc attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"Const{typeof(myfunc)}_autodiff"(%arg0: tensor<3xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<3xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<3xf64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<10> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<3xf64> attributes {enzyme.disable_mincut, enzyme.checkpoint_period = 3 : i64, enzyme.enable_checkpointing = true}
    cond {
      %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      %2:2 = stablehlo.while(%iterArg_3 = %c, %iterArg_4 = %iterArg_2) : tensor<i64>, tensor<3xf64> attributes {enzyme.disable_mincut, enzyme.checkpoint_period = 3 : i64, enzyme.enable_checkpointing = true}
      cond {
        %3 = stablehlo.compare LT, %iterArg_3, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %3 : tensor<i1>
      } do {
        %3 = stablehlo.add %iterArg_3, %c_1 : tensor<i64>
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

// Checkpointed forward/recomputation loops retain their induction comparison.
// Unannotated dynamic reverse loops can carry the condition after differentiation;
// check the initial predicate, the returned next predicate, and all data results.
// CHECK-LABEL: func.func private @"diffeConst{typeof(myfunc)}_autodiff"(%arg0: tensor<3xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg1: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CHECK-NEXT:     %c = stablehlo.constant dense<-4> : tensor<i64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:     %c_3 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %c_4 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:     %c_5 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<3xf64>
// CHECK-NEXT:     %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %c_5, %iterArg_7 = %arg0) : tensor<i64>, tensor<3xf64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_segment}
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %2 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %2 = stablehlo.multiply %iterArg, %c {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:       %3 = stablehlo.add %c_4, %2 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:       %4 = stablehlo.minimum %c_2, %3 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:       %5:2 = stablehlo.while(%iterArg_8 = %c_5, %iterArg_9 = %iterArg_7) : tensor<i64>, tensor<3xf64> attributes {enzyme.disable_mincut, enzymexla.checkpoint_segment}
// CHECK-NEXT:       cond {
// CHECK-NEXT:         %7 = stablehlo.compare LT, %iterArg_8, %4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %7 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %7:2 = stablehlo.while(%iterArg_10 = %c_5, %iterArg_11 = %iterArg_9) : tensor<i64>, tensor<3xf64> attributes {enzyme.checkpoint_period = 3 : i64, enzyme.disable_mincut, enzyme.enable_checkpointing = true, enzymexla.checkpoint_segment}
// CHECK-NEXT:         cond {
// CHECK-NEXT:           %9 = stablehlo.compare LT, %iterArg_10, %c_4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:           stablehlo.return %9 : tensor<i1>
// CHECK-NEXT:         } do {
// CHECK-NEXT:           %9 = stablehlo.add %iterArg_10, %c_3 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:           %10 = stablehlo.multiply %cst, %iterArg_11 : tensor<3xf64>
// CHECK-NEXT:           %11 = stablehlo.add %iterArg_11, %10 : tensor<3xf64>
// CHECK-NEXT:           stablehlo.return %9, %11 : tensor<i64>, tensor<3xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:         %8 = stablehlo.add %iterArg_8, %c_3 : tensor<i64>
// CHECK-NEXT:         stablehlo.return %8, %7#1 : tensor<i64>, tensor<3xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %6 = stablehlo.add %iterArg, %c_3 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:       stablehlo.return %6, %5#1 : tensor<i64>, tensor<3xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %1:7 = stablehlo.while(%iterArg = %c_5, %iterArg_7 = %arg1, %iterArg_8 = %cst_6, %iterArg_9 = %cst_6, %iterArg_10 = %cst_6, %iterArg_11 = %cst_6, %iterArg_12 = %cst_6) : tensor<i64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %2 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %2 = stablehlo.subtract %c_0, %iterArg {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:       %3 = stablehlo.multiply %c, %2 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:       %4 = stablehlo.add %c_4, %3 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:       %5 = stablehlo.minimum %c_2, %4 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:       %6 = stablehlo.compare LT, %c_5, %5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       %7:8 = stablehlo.while(%iterArg_13 = %c_5, %iterArg_14 = %iterArg_7, %iterArg_15 = %iterArg_8, %iterArg_16 = %iterArg_9, %iterArg_17 = %iterArg_10, %iterArg_18 = %iterArg_11, %iterArg_19 = %iterArg_12, %iterArg_20 = %6) : tensor<i64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<i1>
// CHECK-NEXT:       cond {
// CHECK-NEXT:         stablehlo.return %iterArg_20 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %9 = stablehlo.add %iterArg_15, %iterArg_14 : tensor<3xf64>
// CHECK-NEXT:         %10:5 = stablehlo.while(%iterArg_21 = %c_5, %iterArg_22 = %9, %iterArg_23 = %iterArg_16, %iterArg_24 = %iterArg_17, %iterArg_25 = %iterArg_18) : tensor<i64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
// CHECK-NEXT:         cond {
// CHECK-NEXT:           %14 = stablehlo.compare LT, %iterArg_21, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:           stablehlo.return %14 : tensor<i1>
// CHECK-NEXT:         } do {
// CHECK-NEXT:           %14 = stablehlo.subtract %c_0, %iterArg_21 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:           %15 = stablehlo.multiply %c, %14 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:           %16 = stablehlo.add %c_4, %15 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:           %17 = stablehlo.minimum %c_2, %16 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:           %18 = stablehlo.compare LT, %c_5, %17 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:           %19:6 = stablehlo.while(%iterArg_26 = %c_5, %iterArg_27 = %iterArg_22, %iterArg_28 = %iterArg_23, %iterArg_29 = %iterArg_24, %iterArg_30 = %iterArg_25, %iterArg_31 = %18) : tensor<i64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<i1>
// CHECK-NEXT:           cond {
// CHECK-NEXT:             stablehlo.return %iterArg_31 : tensor<i1>
// CHECK-NEXT:           } do {
// CHECK-NEXT:             %21 = stablehlo.add %iterArg_28, %iterArg_27 : tensor<3xf64>
// CHECK-NEXT:             %22 = stablehlo.add %iterArg_29, %21 : tensor<3xf64>
// CHECK-NEXT:             %23 = stablehlo.add %iterArg_30, %21 : tensor<3xf64>
// CHECK-NEXT:             %24 = stablehlo.multiply %23, %cst : tensor<3xf64>
// CHECK-NEXT:             %25 = stablehlo.add %22, %24 : tensor<3xf64>
// CHECK-NEXT:             %26 = stablehlo.add %iterArg_26, %c_3 : tensor<i64>
// CHECK-NEXT:             %27 = stablehlo.compare LT, %26, %17 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:             stablehlo.return %26, %25, %cst_6, %cst_6, %cst_6, %27 : tensor<i64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<i1>
// CHECK-NEXT:           }
// CHECK-NEXT:           %20 = stablehlo.add %iterArg_21, %c_3 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:           stablehlo.return %20, %19#1, %19#2, %19#3, %19#4 : tensor<i64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:         %11 = stablehlo.add %iterArg_19, %10#1 : tensor<3xf64>
// CHECK-NEXT:         %12 = stablehlo.add %iterArg_13, %c_3 : tensor<i64>
// CHECK-NEXT:         %13 = stablehlo.compare LT, %12, %5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %12, %11, %cst_6, %10#2, %10#3, %10#4, %cst_6, %13 : tensor<i64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<i1>
// CHECK-NEXT:       }
// CHECK-NEXT:       %8 = stablehlo.add %iterArg, %c_3 {enzymexla.bounds = {{.*}} : tensor<i64>
// CHECK-NEXT:       stablehlo.return %8, %7#1, %7#2, %7#3, %7#4, %7#5, %7#6 : tensor<i64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1, %1#1 : tensor<3xf64>, tensor<3xf64>
// CHECK-NEXT:   }
