// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 --canonicalize | FileCheck %s

module {
  func.func private @f(%initP : memref<f64>, %arg : memref<28x12xf64>, %arg1 : memref<9x27x59xf64>) {
    %cst = arith.constant 1.600000e+00 : f64
    %cst_0 = arith.constant 6.000000e-01 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    affine.parallel (%arg7, %arg8) = (0, 0) to (20, 45) {
      %init = affine.load %initP[] : memref<f64>
      %17 = affine.for %arg9 = 0 to 9 iter_args(%arg11 = %init) -> (f64) {
        %18 = affine.load %arg[%arg7 + 8, %arg9 + 3] : memref<28x12xf64>
        %34 = arith.addf %arg11, %18 : f64
	affine.store %34, %arg1[%arg9, %arg7 + 7, %arg8 + 7] : memref<9x27x59xf64>
        affine.yield %34 : f64
      }
    }
    return
  }
  func.func private @f2(%initP : memref<f64>, %arg : memref<28x12xf64>, %arg1 : memref<9x27x59xf64>) {
    %cst = arith.constant 1.600000e+00 : f64
    %cst_0 = arith.constant 6.000000e-01 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    affine.parallel (%arg7, %arg8) = (0, 0) to (20, 45) {
      %init = affine.load %initP[] : memref<f64>
      %17 = affine.for %arg9 = 0 to 9 iter_args(%arg11 = %init) -> (f64) {
        %18 = affine.load %arg[%arg7 + 8, %arg9 + 3] : memref<28x12xf64>
        %34 = arith.addf %18, %arg11 : f64
	affine.store %34, %arg1[%arg9, %arg7 + 7, %arg8 + 7] : memref<9x27x59xf64>
        affine.yield %34 : f64
      }
    }
    return
  }
  func.func private @s(%initP : memref<f64>, %arg : memref<28x12xf64>, %arg1 : memref<9x27x59xf64>) {
    %cst = arith.constant 1.600000e+00 : f64
    %cst_0 = arith.constant 6.000000e-01 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    affine.parallel (%arg7, %arg8) = (0, 0) to (20, 45) {
      %init = affine.load %initP[] : memref<f64>
      %17 = affine.for %arg9 = 0 to 9 iter_args(%arg11 = %init) -> (f64) {
        %18 = affine.load %arg[%arg7 + 8, %arg9 + 3] : memref<28x12xf64>
        %34 = arith.subf %arg11, %18 : f64
	affine.store %34, %arg1[%arg9, %arg7 + 7, %arg8 + 7] : memref<9x27x59xf64>
        affine.yield %34 : f64
      }
    }
    return
  }
  func.func private @s2(%initP : memref<f64>, %arg : memref<28x12xf64>, %arg1 : memref<9x27x59xf64>) {
    %cst = arith.constant 1.600000e+00 : f64
    %cst_0 = arith.constant 6.000000e-01 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    affine.parallel (%arg7, %arg8) = (0, 0) to (20, 45) {
      %init = affine.load %initP[] : memref<f64>
      %17 = affine.for %arg9 = 0 to 9 iter_args(%arg11 = %init) -> (f64) {
        %18 = affine.load %arg[%arg7 + 8, %arg9 + 3] : memref<28x12xf64>
        %34 = arith.subf %18, %arg11 : f64
	affine.store %34, %arg1[%arg9, %arg7 + 7, %arg8 + 7] : memref<9x27x59xf64>
        affine.yield %34 : f64
      }
    }
    return
  }
}

// CHECK:  func.func private @s2_raised(%arg0: tensor<f64>, %arg1: tensor<28x12xf64>, %arg2: tensor<9x27x59xf64>) -> (tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<20x45xf64>
// CHECK-NEXT:    %1:3 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %0, %iterArg_6 = %arg2) : tensor<i64>, tensor<20x45xf64>, tensor<9x27x59xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      %3 = stablehlo.dynamic_slice %arg1, %c_1, %2, sizes = [20, 1] : (tensor<28x12xf64>, tensor<i64>, tensor<i64>) -> tensor<20x1xf64>
// CHECK-NEXT:      %4 = stablehlo.reshape %3 : (tensor<20x1xf64>) -> tensor<20xf64>
// CHECK-NEXT:      %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<20xf64>) -> tensor<20x45xf64>
// CHECK-NEXT:      %6 = arith.subf %5, %iterArg_5 : tensor<20x45xf64>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<20x45xf64>) -> tensor<1x20x45xf64>
// CHECK-NEXT:      %8 = stablehlo.dynamic_update_slice %iterArg_6, %7, %iterArg, %c, %c : (tensor<9x27x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<9x27x59xf64>
// CHECK-NEXT:      %9 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %9, %6, %8 : tensor<i64>, tensor<20x45xf64>, tensor<9x27x59xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %arg0, %arg1, %1#2 : tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>
// CHECK-NEXT:  }

// CHECK:  func.func private @s_raised(%arg0: tensor<f64>, %arg1: tensor<28x12xf64>, %arg2: tensor<9x27x59xf64>) -> (tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [8:28, 3:12] : (tensor<28x12xf64>) -> tensor<20x9xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<20x9xf64>
// CHECK-NEXT{LITERAL}:    %2 = "stablehlo.reduce_window"(%0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 9>, window_strides = array<i64: 1, 1>}> ({
// CHECK-NEXT:    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:      %6 = stablehlo.add %arg3, %arg4 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %6 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<20x9xf64>, tensor<f64>) -> tensor<20x9xf64>
// CHECK-NEXT:    %3 = stablehlo.subtract %1, %2 : tensor<20x9xf64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 0] : (tensor<20x9xf64>) -> tensor<9x20x45xf64>
// CHECK-NEXT:    %5 = stablehlo.dynamic_update_slice %arg2, %4, %c_0, %c, %c : (tensor<9x27x59xf64>, tensor<9x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<9x27x59xf64>
// CHECK-NEXT:    return %arg0, %arg1, %5 : tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>
// CHECK-NEXT:  }

// CHECK:  func.func private @f2_raised(%arg0: tensor<f64>, %arg1: tensor<28x12xf64>, %arg2: tensor<9x27x59xf64>) -> (tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [8:28, 3:12] : (tensor<28x12xf64>) -> tensor<20x9xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<20x9xf64>
// CHECK-NEXT{LITERAL}:    %2 = "stablehlo.reduce_window"(%0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 9>, window_strides = array<i64: 1, 1>}> ({
// CHECK-NEXT:    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:      %6 = stablehlo.add %arg3, %arg4 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %6 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<20x9xf64>, tensor<f64>) -> tensor<20x9xf64>
// CHECK-NEXT:    %3 = stablehlo.add %2, %1 : tensor<20x9xf64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 0] : (tensor<20x9xf64>) -> tensor<9x20x45xf64>
// CHECK-NEXT:    %5 = stablehlo.dynamic_update_slice %arg2, %4, %c_0, %c, %c : (tensor<9x27x59xf64>, tensor<9x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<9x27x59xf64>
// CHECK-NEXT:    return %arg0, %arg1, %5 : tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>
// CHECK-NEXT:  }

// CHECK:  func.func private @f_raised(%arg0: tensor<f64>, %arg1: tensor<28x12xf64>, %arg2: tensor<9x27x59xf64>) -> (tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [8:28, 3:12] : (tensor<28x12xf64>) -> tensor<20x9xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<20x9xf64>
// CHECK-NEXT{LITERAL}:    %2 = "stablehlo.reduce_window"(%0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 9>, window_strides = array<i64: 1, 1>}> ({
// CHECK-NEXT:    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:      %6 = stablehlo.add %arg3, %arg4 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %6 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<20x9xf64>, tensor<f64>) -> tensor<20x9xf64>
// CHECK-NEXT:    %3 = stablehlo.add %2, %1 : tensor<20x9xf64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 0] : (tensor<20x9xf64>) -> tensor<9x20x45xf64>
// CHECK-NEXT:    %5 = stablehlo.dynamic_update_slice %arg2, %4, %c_0, %c, %c : (tensor<9x27x59xf64>, tensor<9x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<9x27x59xf64>
// CHECK-NEXT:    return %arg0, %arg1, %5 : tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>
// CHECK-NEXT:  }
