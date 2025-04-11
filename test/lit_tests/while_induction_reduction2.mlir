// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=while_induction_reduction" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s | FileCheck %s

module {
  func.func @"myloop!"(%arg0: tensor<6xf64>, %arg1: tensor<6xf64>, %arg2: tensor<6xf64>, %arg3: tensor<6xf64>, %arg4: tensor<6xf64>, %arg5: tensor<6xf64>, %arg6: tensor<6xf64>, %arg7: tensor<6xf64>, %arg8: tensor<6x6x7xf64>, %arg9: tensor<i64>) -> (tensor<i64>, tensor<i64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6x6x7xf64>) {
    %c = stablehlo.constant dense<2> : tensor<i64>
    %c_0 = stablehlo.constant dense<4> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_3 = %arg8) : tensor<i64>, tensor<6x6x7xf64>
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %arg9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.slice %iterArg_3 [2:4, 2:3, 2:5] : (tensor<6x6x7xf64>) -> tensor<2x1x3xf64>
      %2 = stablehlo.reshape %1 : (tensor<2x1x3xf64>) -> tensor<2x3xf64>
      %3 = stablehlo.slice %iterArg_3 [2:4, 3:4, 2:5] : (tensor<6x6x7xf64>) -> tensor<2x1x3xf64>
      %4 = stablehlo.reshape %3 : (tensor<2x1x3xf64>) -> tensor<2x3xf64>
      %5 = stablehlo.broadcast_in_dim %2, dims = [0, 2] : (tensor<2x3xf64>) -> tensor<2x1x3xf64>
      %6 = stablehlo.dynamic_update_slice %iterArg_3, %5, %c, %c_1, %c : (tensor<6x6x7xf64>, tensor<2x1x3xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<6x6x7xf64>
      %7 = stablehlo.broadcast_in_dim %4, dims = [0, 2] : (tensor<2x3xf64>) -> tensor<2x1x3xf64>
      %8 = stablehlo.dynamic_update_slice %6, %7, %c, %c_0, %c : (tensor<6x6x7xf64>, tensor<2x1x3xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<6x6x7xf64>
      %9 = stablehlo.slice %8 [0:6, 0:6, 2:3] : (tensor<6x6x7xf64>) -> tensor<6x6x1xf64>
      %10 = stablehlo.reshape %9 : (tensor<6x6x1xf64>) -> tensor<6x6xf64>
      %11 = stablehlo.slice %8 [0:6, 0:6, 3:4] : (tensor<6x6x7xf64>) -> tensor<6x6x1xf64>
      %12 = stablehlo.reshape %11 : (tensor<6x6x1xf64>) -> tensor<6x6xf64>
      %13 = stablehlo.slice %8 [0:6, 0:6, 4:5] : (tensor<6x6x7xf64>) -> tensor<6x6x1xf64>
      %14 = stablehlo.reshape %13 : (tensor<6x6x1xf64>) -> tensor<6x6xf64>
      %15 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<6x6xf64>) -> tensor<6x6x1xf64>
      %16 = stablehlo.slice %8 [0:6, 0:6, 1:7] : (tensor<6x6x7xf64>) -> tensor<6x6x6xf64>
      %17 = stablehlo.concatenate %15, %16, dim = 2 : (tensor<6x6x1xf64>, tensor<6x6x6xf64>) -> tensor<6x6x7xf64>
      %18 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<6x6xf64>) -> tensor<6x6x1xf64>
      %19 = stablehlo.slice %17 [0:6, 0:6, 0:1] : (tensor<6x6x7xf64>) -> tensor<6x6x1xf64>
      %20 = stablehlo.slice %17 [0:6, 0:6, 2:7] : (tensor<6x6x7xf64>) -> tensor<6x6x5xf64>
      %21 = stablehlo.concatenate %19, %18, %20, dim = 2 : (tensor<6x6x1xf64>, tensor<6x6x1xf64>, tensor<6x6x5xf64>) -> tensor<6x6x7xf64>
      %22 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<6x6xf64>) -> tensor<6x6x1xf64>
      %23 = stablehlo.slice %21 [0:6, 0:6, 0:5] : (tensor<6x6x7xf64>) -> tensor<6x6x5xf64>
      %24 = stablehlo.slice %21 [0:6, 0:6, 6:7] : (tensor<6x6x7xf64>) -> tensor<6x6x1xf64>
      %25 = stablehlo.concatenate %23, %22, %24, dim = 2 : (tensor<6x6x5xf64>, tensor<6x6x1xf64>, tensor<6x6x1xf64>) -> tensor<6x6x7xf64>
      %26 = stablehlo.slice %25 [0:6, 0:6, 0:6] : (tensor<6x6x7xf64>) -> tensor<6x6x6xf64>
      %27 = stablehlo.concatenate %26, %15, dim = 2 : (tensor<6x6x6xf64>, tensor<6x6x1xf64>) -> tensor<6x6x7xf64>
      %28 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      stablehlo.return %28, %27 : tensor<i64>, tensor<6x6x7xf64>
    }
    return %0#0, %arg9, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %0#1 : tensor<i64>, tensor<i64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6x6x7xf64>
  }
}

// WhileInductionReduction does not apply since we do not propagate into the return

// CHECK:  func.func @"myloop!"(%arg0: tensor<6xf64>, %arg1: tensor<6xf64>, %arg2: tensor<6xf64>, %arg3: tensor<6xf64>, %arg4: tensor<6xf64>, %arg5: tensor<6xf64>, %arg6: tensor<6xf64>, %arg7: tensor<6xf64>, %arg8: tensor<6x6x7xf64>, %arg9: tensor<i64>) -> (tensor<i64>, tensor<i64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6x6x7xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_3 = %arg8) : tensor<i64>, tensor<6x6x7xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %arg9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %1 = stablehlo.slice %iterArg_3 [2:4, 2:3, 2:5] : (tensor<6x6x7xf64>) -> tensor<2x1x3xf64>
// CHECK-NEXT:      %2 = stablehlo.reshape %1 : (tensor<2x1x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:      %3 = stablehlo.slice %iterArg_3 [2:4, 3:4, 2:5] : (tensor<6x6x7xf64>) -> tensor<2x1x3xf64>
// CHECK-NEXT:      %4 = stablehlo.reshape %3 : (tensor<2x1x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:      %5 = stablehlo.broadcast_in_dim %2, dims = [0, 2] : (tensor<2x3xf64>) -> tensor<2x1x3xf64>
// CHECK-NEXT:      %6 = stablehlo.dynamic_update_slice %iterArg_3, %5, %c, %c_1, %c : (tensor<6x6x7xf64>, tensor<2x1x3xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<6x6x7xf64>
// CHECK-NEXT:      %7 = stablehlo.broadcast_in_dim %4, dims = [0, 2] : (tensor<2x3xf64>) -> tensor<2x1x3xf64>
// CHECK-NEXT:      %8 = stablehlo.dynamic_update_slice %6, %7, %c, %c_0, %c : (tensor<6x6x7xf64>, tensor<2x1x3xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<6x6x7xf64>
// CHECK-NEXT:      %9 = stablehlo.slice %8 [0:6, 0:6, 2:3] : (tensor<6x6x7xf64>) -> tensor<6x6x1xf64>
// CHECK-NEXT:      %10 = stablehlo.reshape %9 : (tensor<6x6x1xf64>) -> tensor<6x6xf64>
// CHECK-NEXT:      %11 = stablehlo.slice %8 [0:6, 0:6, 3:4] : (tensor<6x6x7xf64>) -> tensor<6x6x1xf64>
// CHECK-NEXT:      %12 = stablehlo.reshape %11 : (tensor<6x6x1xf64>) -> tensor<6x6xf64>
// CHECK-NEXT:      %13 = stablehlo.slice %8 [0:6, 0:6, 4:5] : (tensor<6x6x7xf64>) -> tensor<6x6x1xf64>
// CHECK-NEXT:      %14 = stablehlo.reshape %13 : (tensor<6x6x1xf64>) -> tensor<6x6xf64>
// CHECK-NEXT:      %15 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<6x6xf64>) -> tensor<6x6x1xf64>
// CHECK-NEXT:      %16 = stablehlo.slice %8 [0:6, 0:6, 1:7] : (tensor<6x6x7xf64>) -> tensor<6x6x6xf64>
// CHECK-NEXT:      %17 = stablehlo.concatenate %15, %16, dim = 2 : (tensor<6x6x1xf64>, tensor<6x6x6xf64>) -> tensor<6x6x7xf64>
// CHECK-NEXT:      %18 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<6x6xf64>) -> tensor<6x6x1xf64>
// CHECK-NEXT:      %19 = stablehlo.slice %17 [0:6, 0:6, 0:1] : (tensor<6x6x7xf64>) -> tensor<6x6x1xf64>
// CHECK-NEXT:      %20 = stablehlo.slice %17 [0:6, 0:6, 2:7] : (tensor<6x6x7xf64>) -> tensor<6x6x5xf64>
// CHECK-NEXT:      %21 = stablehlo.concatenate %19, %18, %20, dim = 2 : (tensor<6x6x1xf64>, tensor<6x6x1xf64>, tensor<6x6x5xf64>) -> tensor<6x6x7xf64>
// CHECK-NEXT:      %22 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<6x6xf64>) -> tensor<6x6x1xf64>
// CHECK-NEXT:      %23 = stablehlo.slice %21 [0:6, 0:6, 0:5] : (tensor<6x6x7xf64>) -> tensor<6x6x5xf64>
// CHECK-NEXT:      %24 = stablehlo.slice %21 [0:6, 0:6, 6:7] : (tensor<6x6x7xf64>) -> tensor<6x6x1xf64>
// CHECK-NEXT:      %25 = stablehlo.concatenate %23, %22, %24, dim = 2 : (tensor<6x6x5xf64>, tensor<6x6x1xf64>, tensor<6x6x1xf64>) -> tensor<6x6x7xf64>
// CHECK-NEXT:      %26 = stablehlo.slice %25 [0:6, 0:6, 0:6] : (tensor<6x6x7xf64>) -> tensor<6x6x6xf64>
// CHECK-NEXT:      %27 = stablehlo.concatenate %26, %15, dim = 2 : (tensor<6x6x6xf64>, tensor<6x6x1xf64>) -> tensor<6x6x7xf64>
// CHECK-NEXT:      %28 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %28, %27 : tensor<i64>, tensor<6x6x7xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#0, %arg9, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %0#1 : tensor<i64>, tensor<i64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6xf64>, tensor<6x6x7xf64>
// CHECK-NEXT:  }
