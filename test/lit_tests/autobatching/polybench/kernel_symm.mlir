// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<i64>) -> tensor<32x64xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x64xf32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<32> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.convert %arg4 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.subtract %0, %c_0 : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3 = stablehlo.dynamic_slice %arg3, %c, %1, sizes = [32, 1] : (tensor<32x64xf32>, tensor<i32>, tensor<i32>) -> tensor<32x1xf32>
    %4 = stablehlo.reshape %3 : (tensor<32x1xf32>) -> tensor<32xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<32xf32>
    %6 = stablehlo.dynamic_slice %arg2, %1, %1, sizes = [1, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 1] : (tensor<1x1xf32>) -> tensor<32x1x1xf32>
    %8 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<32xf32>) -> tensor<32x64x1xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<32x1xf32>
    %10 = stablehlo.multiply %9, %3 : tensor<32x1xf32>
    %11 = stablehlo.reshape %7 : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<32x1xf32>
    %13 = stablehlo.dynamic_slice %arg2, %c, %1, sizes = [64, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<64x1xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [1, 2] : (tensor<64x1xf32>) -> tensor<32x64x1xf32>
    %15 = stablehlo.multiply %8, %14 : tensor<32x64x1xf32>
    %16 = stablehlo.reshape %arg3 : (tensor<32x64xf32>) -> tensor<32x64x1xf32>
    %17 = stablehlo.dot_general %16, %13, contracting_dims = [1, 2] x [0, 1] : (tensor<32x64x1xf32>, tensor<64x1xf32>) -> tensor<32xf32>
    %18 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
    %19 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<32x1x1xf32>
    %20 = stablehlo.dynamic_slice %15, %c_1, %18, %c_1, sizes = [32, 1, 1] : (tensor<32x64x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<32x1x1xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<32x1x1xf32>
    %22 = stablehlo.reshape %12 : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %23 = stablehlo.add %21, %22 : tensor<32x1x1xf32>
    %24 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<32x1x1xf32>
    %25 = stablehlo.reshape %17 : (tensor<32xf32>) -> tensor<32x1x1xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<32x1x1xf32>
    %27 = stablehlo.add %23, %26 : tensor<32x1x1xf32>
    %28:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %cst) : tensor<i64>, tensor<32x64xf32>
    cond {
      %29 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %29 : tensor<i1>
    } do {
      %29 = stablehlo.add %c_3, %iterArg {enzymexla.bounds = [[1, 32]]} : tensor<i64>
      %30 = stablehlo.convert %29 {enzymexla.bounds = [[1, 32]]} : (tensor<i64>) -> tensor<i32>
      %31 = stablehlo.subtract %30, %c_0 {enzymexla.bounds = [[0, 31]]} : tensor<i32>
      %32 = stablehlo.dynamic_slice %15, %iterArg, %c_1, %c_1, sizes = [1, 64, 1] : (tensor<32x64x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x64x1xf32>
      %33 = stablehlo.dynamic_slice %27, %iterArg, %c_1, %c_1, sizes = [1, 1, 1] : (tensor<32x1x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x1xf32>
      %34 = stablehlo.reshape %33 : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
      %35 = stablehlo.reshape %32 : (tensor<1x64x1xf32>) -> tensor<1x64xf32>
      %36 = stablehlo.dynamic_update_slice %iterArg_4, %35, %31, %c : (tensor<32x64xf32>, tensor<1x64xf32>, tensor<i32>, tensor<i32>) -> tensor<32x64xf32>
      %37 = stablehlo.dynamic_update_slice %36, %34, %31, %1 : (tensor<32x64xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<32x64xf32>
      stablehlo.return %29, %37 : tensor<i64>, tensor<32x64xf32>
    }
    return %28#1 : tensor<32x64xf32>
  }
}

// CHECK-NOT: stablehlo.while
// CHECK: %28 = stablehlo.reshape %15 : (tensor<32x64x1xf32>) -> tensor<32x1x64xf32>
// CHECK-NEXT: %29 = stablehlo.dynamic_update_slice %28, %27, %c, %c, %1 : (tensor<32x1x64xf32>, tensor<32x1x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x1x64xf32>
// CHECK-NEXT: %30 = stablehlo.reshape %29 : (tensor<32x1x64xf32>) -> tensor<32x64xf32>
// CHECK-NEXT: return %30 : tensor<32x64xf32>

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<i64>) -> tensor<32x64xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x64xf32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<32> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.convert %arg4 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %arg1 : (tensor<f32>) -> tensor<1x1xf32>
    %2 = stablehlo.subtract %0, %c_0 : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %4 = stablehlo.dynamic_slice %arg3, %c, %2, sizes = [32, 1] : (tensor<32x64xf32>, tensor<i32>, tensor<i32>) -> tensor<32x1xf32>
    %5 = stablehlo.reshape %4 : (tensor<32x1xf32>) -> tensor<32xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<32xf32>
    %7 = stablehlo.dynamic_slice %arg2, %2, %2, sizes = [1, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [2, 1] : (tensor<1x1xf32>) -> tensor<32x1x1xf32>
    %9 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<32xf32>) -> tensor<32x64x1xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<32x1xf32>
    %11 = stablehlo.multiply %10, %4 : tensor<32x1xf32>
    %12 = stablehlo.reshape %8 : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<32x1xf32>
    %14 = stablehlo.dynamic_slice %arg2, %c, %2, sizes = [64, 1] : (tensor<64x64xf32>, tensor<i32>, tensor<i32>) -> tensor<64x1xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<64x1xf32>) -> tensor<32x64x1xf32>
    %16 = stablehlo.multiply %9, %15 : tensor<32x64x1xf32>
    %17 = stablehlo.reshape %arg3 : (tensor<32x64xf32>) -> tensor<32x64x1xf32>
    %18 = stablehlo.dot_general %17, %14, contracting_dims = [1, 2] x [0, 1] : (tensor<32x64x1xf32>, tensor<64x1xf32>) -> tensor<32xf32>
    %19 = stablehlo.reshape %18 : (tensor<32xf32>) -> tensor<32x1xf32>
    %20 = stablehlo.multiply %10, %19 : tensor<32x1xf32>
    %21:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %cst) : tensor<i64>, tensor<32x64xf32>
    cond {
      %22 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %22 : tensor<i1>
    } do {
      %22 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %23 = stablehlo.convert %22 : (tensor<i64>) -> tensor<i32>
      %24 = stablehlo.subtract %23, %c_0 : tensor<i32>
      %25 = stablehlo.dynamic_slice %16, %iterArg, %c_1, %c_1, sizes = [1, 64, 1] : (tensor<32x64x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x64x1xf32>
      %26 = stablehlo.dynamic_slice %iterArg_4, %24, %c, sizes = [1, 64] : (tensor<32x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x64xf32>
      %27 = stablehlo.reshape %25 : (tensor<1x64x1xf32>) -> tensor<1x64xf32>
      %28 = stablehlo.add %26, %27 : tensor<1x64xf32>
      %29 = stablehlo.dynamic_update_slice %iterArg_4, %28, %24, %c : (tensor<32x64xf32>, tensor<1x64xf32>, tensor<i32>, tensor<i32>) -> tensor<32x64xf32>
      %30 = stablehlo.dynamic_slice %29, %24, %2, sizes = [1, 1] : (tensor<32x64xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
      %31 = stablehlo.multiply %1, %30 : tensor<1x1xf32>
      %32 = stablehlo.dynamic_slice %13, %iterArg, %c_1, sizes = [1, 1] : (tensor<32x1xf32>, tensor<i64>, tensor<i64>) -> tensor<1x1xf32>
      %33 = stablehlo.add %31, %32 : tensor<1x1xf32>
      %34 = stablehlo.dynamic_slice %20, %iterArg, %c_1, sizes = [1, 1] : (tensor<32x1xf32>, tensor<i64>, tensor<i64>) -> tensor<1x1xf32>
      %35 = stablehlo.add %33, %34 : tensor<1x1xf32>
      %36 = stablehlo.dynamic_update_slice %29, %35, %24, %2 : (tensor<32x64xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<32x64xf32>
      stablehlo.return %22, %36 : tensor<i64>, tensor<32x64xf32>
    }
    return %21#1 : tensor<32x64xf32>
  }
}

// CHECK-NOT: stablehlo.while
// CHECK: %20 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT: %21 = stablehlo.dynamic_slice %18, %c, %c, %20, sizes = [32, 1, 1] : (tensor<32x1x64xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<32x1x1xf32>
// CHECK-NEXT: %22 = stablehlo.multiply %19, %21 : tensor<32x1x1xf32>
// CHECK-NEXT: %23 = stablehlo.reshape %12 : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
// CHECK-NEXT: %24 = stablehlo.add %22, %23 : tensor<32x1x1xf32>
// CHECK-NEXT: %25 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<32x1x1xf32>
// CHECK-NEXT: %26 = stablehlo.reshape %17 : (tensor<32xf32>) -> tensor<32x1x1xf32>
// CHECK-NEXT: %27 = stablehlo.multiply %25, %26 : tensor<32x1x1xf32>
// CHECK-NEXT: %28 = stablehlo.add %24, %27 : tensor<32x1x1xf32>
// CHECK-NEXT: %29 = stablehlo.dynamic_update_slice %18, %28, %c_0, %c_0, %1 : (tensor<32x1x64xf32>, tensor<32x1x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x1x64xf32>
// CHECK-NEXT: %30 = stablehlo.reshape %29 : (tensor<32x1x64xf32>) -> tensor<32x64xf32>
// CHECK-NEXT: return %30 : tensor<32x64xf32>
