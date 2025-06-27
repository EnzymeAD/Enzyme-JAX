// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise | FileCheck %s --check-prefix=REVERSE

module @reactant_differe... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"Const{typeof(estimate_tracer_error)}(Main.estimate_tracer_error)_autodiff"(%arg0: tensor<63x63xf64>) -> (tensor<f64>, tensor<63x63xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<3> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %c_2 = stablehlo.constant dense<14> : tensor<i32>
    %c_3 = stablehlo.constant dense<7> : tensor<i32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<78x78x31xf64>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<63x63x16xf64>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0] : (tensor<63x63xf64>) -> tensor<63x63x1xf64>
    %1 = stablehlo.dynamic_update_slice %cst_4, %0, %c_3, %c_3, %c_2 : (tensor<78x78x31xf64>, tensor<63x63x1xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<78x78x31xf64>
    %2:3 = stablehlo.while(%iterArg = %c, %iterArg_6 = %1, %iterArg_7 = %cst_5) : tensor<i64>, tensor<78x78x31xf64>, tensor<63x63x16xf64> attributes {enzymexla.disable_min_cut}
     cond {
      %4 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    } do {
      %4 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      %5 = stablehlo.slice %iterArg_6 [7:70, 7:70, 7:23] : (tensor<78x78x31xf64>) -> tensor<63x63x16xf64>
      %6 = stablehlo.add %5, %iterArg_7 : tensor<63x63x16xf64>
      %7 = stablehlo.dynamic_update_slice %iterArg_6, %6, %c_3, %c_3, %c_3 : (tensor<78x78x31xf64>, tensor<63x63x16xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<78x78x31xf64>
      %8 = stablehlo.slice %7 [8:71, 6:69, 14:15] : (tensor<78x78x31xf64>) -> tensor<63x63x1xf64>
      %9 = stablehlo.slice %7 [8:71, 6:69, 7:8] : (tensor<78x78x31xf64>) -> tensor<63x63x1xf64>
      %10 = stablehlo.slice %7 [8:71, 6:69, 9:23] : (tensor<78x78x31xf64>) -> tensor<63x63x14xf64>
      %11 = stablehlo.concatenate %9, %8, %10, dim = 2 : (tensor<63x63x1xf64>, tensor<63x63x1xf64>, tensor<63x63x14xf64>) -> tensor<63x63x16xf64>
      %12 = stablehlo.transpose %7, dims = [2, 1, 0] : (tensor<78x78x31xf64>) -> tensor<31x78x78xf64>
      %13 = stablehlo.slice %12 [8:9, 7:70, 7:70] : (tensor<31x78x78xf64>) -> tensor<1x63x63xf64>
      %14 = stablehlo.reshape %13 : (tensor<1x63x63xf64>) -> tensor<63x63xf64>
      %15 = stablehlo.broadcast_in_dim %14, dims = [1, 0] : (tensor<63x63xf64>) -> tensor<63x63x16xf64>
      %16 = stablehlo.dynamic_update_slice %iterArg_6, %15, %c_3, %c_3, %c_3 : (tensor<78x78x31xf64>, tensor<63x63x16xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<78x78x31xf64>
      stablehlo.return %4, %16, %11 : tensor<i64>, tensor<78x78x31xf64>, tensor<63x63x16xf64>
    }
    %3 = stablehlo.reduce(%2#2 init: %cst) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<63x63x16xf64>, tensor<f64>) -> tensor<f64>
    return %3, %arg0 : tensor<f64>, tensor<63x63xf64>
  }
  func.func @main(%arg0: tensor<63x63xf64> {tf.aliasing_output = 1 : i32}) -> (tensor<63x63xf64>, tensor<63x63xf64>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<63x63xf64>
    %0:2 = enzyme.autodiff @"Const{typeof(estimate_tracer_error)}(Main.estimate_tracer_error)_autodiff"(%arg0, %cst, %cst_0) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>], strong_zero = true} : (tensor<63x63xf64>, tensor<f64>, tensor<63x63xf64>) -> (tensor<63x63xf64>, tensor<63x63xf64>)
    return %0#1, %0#0 : tensor<63x63xf64>, tensor<63x63xf64>
  }
}

// REVERSE:  func.func private @"diffeConst{typeof(estimate_tracer_error)}(Main.estimate_tracer_error)_autodiff"(%arg0: tensor<63x63xf64>, %arg1: tensor<f64>, %arg2: tensor<63x63xf64>) -> (tensor<63x63xf64>, tensor<63x63xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// REVERSE-NEXT:    %c = stablehlo.constant dense<2> : tensor<i64>
// REVERSE-NEXT:    %c_0 = stablehlo.constant dense<7> : tensor<i32>
// REVERSE-NEXT:    %c_1 = stablehlo.constant dense<14> : tensor<i32>
// REVERSE-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<i64>
// REVERSE-NEXT:    %c_3 = stablehlo.constant dense<3> : tensor<i64>
// REVERSE-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<i64>
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// REVERSE-NEXT:    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<63x63x16xf64>
// REVERSE-NEXT:    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<78x78x31xf64>
// REVERSE-NEXT:    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<63x63x16xf64>
// REVERSE-NEXT:    %1:4 = stablehlo.while(%iterArg = %c_4, %iterArg_7 = %cst_6, %iterArg_8 = %0, %iterArg_9 = %c) : tensor<i64>, tensor<78x78x31xf64>, tensor<63x63x16xf64>, tensor<i64>
// REVERSE-NEXT:     cond {
// REVERSE-NEXT:      %8 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:      stablehlo.return %8 : tensor<i1>
// REVERSE-NEXT:    } do {
// REVERSE-NEXT:      %8 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// REVERSE-NEXT:      %9 = stablehlo.dynamic_update_slice %iterArg_7, %cst_5, %c_0, %c_0, %c_0 : (tensor<78x78x31xf64>, tensor<63x63x16xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<78x78x31xf64>
// REVERSE-NEXT:      %10 = stablehlo.dynamic_slice %iterArg_7, %c_0, %c_0, %c_0, sizes = [63, 63, 16] : (tensor<78x78x31xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<63x63x16xf64>
// REVERSE-NEXT:      %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<63x63x16xf64>, tensor<f64>) -> tensor<63x63xf64>
// REVERSE-NEXT:      %12 = stablehlo.reshape %11 : (tensor<63x63xf64>) -> tensor<63x63x1xf64>
// REVERSE-NEXT:      %13 = stablehlo.transpose %12, dims = [1, 0, 2] : (tensor<63x63x1xf64>) -> tensor<63x63x1xf64>
// REVERSE-NEXT:      %14 = stablehlo.reshape %13 : (tensor<63x63x1xf64>) -> tensor<63x63xf64>
// REVERSE-NEXT:      %15 = stablehlo.reshape %14 : (tensor<63x63xf64>) -> tensor<1x63x63xf64>
// REVERSE-NEXT:      %16 = stablehlo.pad %15, %cst, low = [8, 7, 7], high = [22, 8, 8], interior = [0, 0, 0] : (tensor<1x63x63xf64>, tensor<f64>) -> tensor<31x78x78xf64>
// REVERSE-NEXT:      %17 = stablehlo.transpose %16, dims = [2, 1, 0] : (tensor<31x78x78xf64>) -> tensor<78x78x31xf64>
// REVERSE-NEXT:      %18 = stablehlo.slice %iterArg_8 [0:63, 0:63, 0:1] : (tensor<63x63x16xf64>) -> tensor<63x63x1xf64>
// REVERSE-NEXT:      %19 = stablehlo.reshape %18 : (tensor<63x63x1xf64>) -> tensor<63x63x1xf64>
// REVERSE-NEXT:      %20 = stablehlo.slice %iterArg_8 [0:63, 0:63, 1:2] : (tensor<63x63x16xf64>) -> tensor<63x63x1xf64>
// REVERSE-NEXT:      %21 = stablehlo.reshape %20 : (tensor<63x63x1xf64>) -> tensor<63x63x1xf64>
// REVERSE-NEXT:      %22 = stablehlo.slice %iterArg_8 [0:63, 0:63, 2:16] : (tensor<63x63x16xf64>) -> tensor<63x63x14xf64>
// REVERSE-NEXT:      %23 = stablehlo.reshape %22 : (tensor<63x63x14xf64>) -> tensor<63x63x14xf64>
// REVERSE-NEXT:      %24 = stablehlo.pad %23, %cst, low = [8, 6, 9], high = [7, 9, 8], interior = [0, 0, 0] : (tensor<63x63x14xf64>, tensor<f64>) -> tensor<78x78x31xf64>
// REVERSE-NEXT:      %25 = stablehlo.add %17, %24 : tensor<78x78x31xf64>
// REVERSE-NEXT:      %26 = stablehlo.pad %19, %cst, low = [8, 6, 7], high = [7, 9, 23], interior = [0, 0, 0] : (tensor<63x63x1xf64>, tensor<f64>) -> tensor<78x78x31xf64>
// REVERSE-NEXT:      %27 = stablehlo.add %25, %26 : tensor<78x78x31xf64>
// REVERSE-NEXT:      %28 = stablehlo.pad %21, %cst, low = [8, 6, 14], high = [7, 9, 16], interior = [0, 0, 0] : (tensor<63x63x1xf64>, tensor<f64>) -> tensor<78x78x31xf64>
// REVERSE-NEXT:      %29 = stablehlo.add %27, %28 : tensor<78x78x31xf64>
// REVERSE-NEXT:      %30 = stablehlo.dynamic_update_slice %29, %cst_5, %c_0, %c_0, %c_0 : (tensor<78x78x31xf64>, tensor<63x63x16xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<78x78x31xf64>
// REVERSE-NEXT:      %31 = stablehlo.add %9, %30 : tensor<78x78x31xf64>
// REVERSE-NEXT:      %32 = stablehlo.dynamic_slice %29, %c_0, %c_0, %c_0, sizes = [63, 63, 16] : (tensor<78x78x31xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<63x63x16xf64>
// REVERSE-NEXT:      %33 = stablehlo.pad %32, %cst, low = [7, 7, 7], high = [8, 8, 8], interior = [0, 0, 0] : (tensor<63x63x16xf64>, tensor<f64>) -> tensor<78x78x31xf64>
// REVERSE-NEXT:      %34 = stablehlo.add %31, %33 : tensor<78x78x31xf64>
// REVERSE-NEXT:      %35 = stablehlo.subtract %iterArg_9, %c_2 : tensor<i64>
// REVERSE-NEXT:      stablehlo.return %8, %34, %32, %35 : tensor<i64>, tensor<78x78x31xf64>, tensor<63x63x16xf64>, tensor<i64>
// REVERSE-NEXT:    }
// REVERSE-NEXT:    %2 = stablehlo.dynamic_slice %1#1, %c_0, %c_0, %c_1, sizes = [63, 63, 1] : (tensor<78x78x31xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<63x63x1xf64>
// REVERSE-NEXT:    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<63x63x1xf64>, tensor<f64>) -> tensor<63x63xf64>
// REVERSE-NEXT:    %4 = stablehlo.reshape %3 : (tensor<63x63xf64>) -> tensor<63x63x1xf64>
// REVERSE-NEXT:    %5 = stablehlo.transpose %4, dims = [1, 0, 2] : (tensor<63x63x1xf64>) -> tensor<63x63x1xf64>
// REVERSE-NEXT:    %6 = stablehlo.reshape %5 : (tensor<63x63x1xf64>) -> tensor<63x63xf64>
// REVERSE-NEXT:    %7 = stablehlo.add %arg2, %6 : tensor<63x63xf64>
// REVERSE-NEXT:    return %arg0, %7 : tensor<63x63xf64>, tensor<63x63xf64>
// REVERSE-NEXT:  }
