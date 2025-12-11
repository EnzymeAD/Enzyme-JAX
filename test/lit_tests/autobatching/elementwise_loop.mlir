// RUN: enzymexlamlir-opt --enzyme-hlo-opt --auto-batching --inline --enzyme-hlo-generate-td="patterns=reshape_dynamic_slice(1);reshape_licm(1);transpose_dynamic_slice;transpose_licm(1);while_is_copy_simplify;reshape_elementwise(1);elementwise_licm(1)" --transform-interpreter --enzyme-hlo-remove-transform --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf64>) -> tensor<10xf64> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<10> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %cst) : tensor<i64>, tensor<10xf64>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %3 = stablehlo.subtract %2, %c : tensor<i32>
      %4 = stablehlo.dynamic_slice %arg0, %3, sizes = [1] : (tensor<10xf64>, tensor<i32>) -> tensor<1xf64>

      %sin_res = stablehlo.sine %4 : tensor<1xf64>
      %neg_res = stablehlo.negate %sin_res : tensor<1xf64>
      %cos_res = stablehlo.cosine %4 : tensor<1xf64>
      %5 = stablehlo.add %neg_res, %cos_res : tensor<1xf64>

      %6 = stablehlo.remainder %iterArg, %c_1 : tensor<i64>
      %7 = stablehlo.add %6, %c_2 : tensor<i64>
      %8 = stablehlo.convert %7 : (tensor<i64>) -> tensor<i32>
      %9 = stablehlo.subtract %8, %c : tensor<i32>
      %10 = stablehlo.dynamic_update_slice %iterArg_3, %5, %9 : (tensor<10xf64>, tensor<1xf64>, tensor<i32>) -> tensor<10xf64>

      stablehlo.return %1, %10 : tensor<i64>, tensor<10xf64>
    }
    return %0#1 : tensor<10xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<10xf64>) -> tensor<10xf64> {
// CHECK-DAG:    [[COS:%.*]] = stablehlo.cosine %arg0 : tensor<10xf64>
// CHECK-DAG:    [[SIN:%.*]] = stablehlo.sine %arg0 : tensor<10xf64>
// CHECK:    [[RES:%.*]] = stablehlo.subtract [[COS]], [[SIN]] : tensor<10xf64>
// CHECK:    return [[RES]] : tensor<10xf64>
// CHECK:  }

module {
  func.func @main(%arg0: tensor<5x4x3xf32>, %arg1: tensor<5x3xf32>) -> tensor<4x5x3xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x5x3xf32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<4> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<5x4x3xf32>) -> tensor<3x4x5xf32>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<5x3xf32>) -> tensor<3x5xf32>
    %2:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %cst) : tensor<i64>, tensor<4x5x3xf32>
    cond {
      %3 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_0 : tensor<i32>
      %6 = stablehlo.dynamic_slice %0, %c, %5, %c, sizes = [3, 1, 5] : (tensor<3x4x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x1x5xf32>
      %7 = stablehlo.reshape %6 : (tensor<3x1x5xf32>) -> tensor<3x5xf32>
      %8 = stablehlo.add %7, %1 : tensor<3x5xf32>
      %9 = stablehlo.sine %8 : tensor<3x5xf32>
      %10 = stablehlo.broadcast_in_dim %9, dims = [2, 1] : (tensor<3x5xf32>) -> tensor<1x5x3xf32>
      %11 = stablehlo.dynamic_update_slice %iterArg_4, %10, %5, %c, %c : (tensor<4x5x3xf32>, tensor<1x5x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x5x3xf32>
      stablehlo.return %3, %11 : tensor<i64>, tensor<4x5x3xf32>
    }
    return %2#1 : tensor<4x5x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x4x3xf32>, %arg1: tensor<5x3xf32>) -> tensor<4x5x3xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 2, 0] : (tensor<5x4x3xf32>) -> tensor<4x3x5xf32>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %arg1, dims = [2, 1] : (tensor<5x3xf32>) -> tensor<4x3x5xf32>
// CHECK-NEXT:   %2 = stablehlo.add %0, %1 : tensor<4x3x5xf32>
// CHECK-NEXT:   %3 = stablehlo.sine %2 : tensor<4x3x5xf32>
// CHECK-NEXT:   %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<4x3x5xf32>) -> tensor<4x5x3xf32>
// CHECK-NEXT:   return %4 : tensor<4x5x3xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<10xf64>) -> tensor<10xf64> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<10> : tensor<i64>
    %c_2 = stablehlo.constant dense<3> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %cst) : tensor<i64>, tensor<10xf64>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %3 = stablehlo.subtract %2, %c : tensor<i32>
      %4 = stablehlo.dynamic_slice %arg0, %3, sizes = [1] : (tensor<10xf64>, tensor<i32>) -> tensor<1xf64>

      %sin_res = stablehlo.sine %4 : tensor<1xf64>
      %neg_res = stablehlo.negate %sin_res : tensor<1xf64>
      %cos_res = stablehlo.cosine %4 : tensor<1xf64>
      %5 = stablehlo.add %neg_res, %cos_res : tensor<1xf64>

      %6 = stablehlo.remainder %iterArg, %c_1 : tensor<i64>
      %7 = stablehlo.add %6, %c_2 : tensor<i64>
      %8 = stablehlo.convert %7 : (tensor<i64>) -> tensor<i32>
      %9 = stablehlo.subtract %8, %c : tensor<i32>
      %10 = stablehlo.dynamic_update_slice %iterArg_3, %5, %9 : (tensor<10xf64>, tensor<1xf64>, tensor<i32>) -> tensor<10xf64>

      stablehlo.return %1, %10 : tensor<i64>, tensor<10xf64>
    }
    return %0#1 : tensor<10xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<10xf64>) -> tensor<10xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.dynamic_slice %arg0, %c_3, sizes = [10] : (tensor<10xf64>, tensor<i32>) -> tensor<10xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %0 [0:10:3] : (tensor<10xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %2 = stablehlo.cosine %1 : tensor<4xf64>
// CHECK-NEXT:   %3 = stablehlo.dynamic_slice %arg0, %c_3, sizes = [10] : (tensor<10xf64>, tensor<i32>) -> tensor<10xf64>
// CHECK-NEXT:   %4 = stablehlo.slice %3 [0:10:3] : (tensor<10xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %5 = stablehlo.sine %4 : tensor<4xf64>
// CHECK-NEXT:   %6:2 = stablehlo.while(%iterArg = %c_0, %iterArg_4 = %cst) : tensor<i64>, tensor<10xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %7 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %7 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %7 = stablehlo.add %c_2, %iterArg : tensor<i64>
// CHECK-NEXT:     %8 = stablehlo.convert %7 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:     %9 = stablehlo.subtract %8, %c : tensor<i32>
// CHECK-NEXT:     %10 = stablehlo.divide %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:     %11 = stablehlo.dynamic_slice %5, %10, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %12 = stablehlo.dynamic_slice %2, %10, sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %13 = stablehlo.subtract %12, %11 : tensor<1xf64>
// CHECK-NEXT:     %14 = stablehlo.dynamic_update_slice %iterArg_4, %13, %9 : (tensor<10xf64>, tensor<1xf64>, tensor<i32>) -> tensor<10xf64>
// CHECK-NEXT:     stablehlo.return %7, %14 : tensor<i64>, tensor<10xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %6#1 : tensor<10xf64>
// CHECK-NEXT: }
