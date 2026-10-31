// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=add_terms_to_reduce" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// The last stage of a sum-factorized apply unrolled into one term per k:
// sum_k x[:, 5 i + k] * b[5 k + j]. The k = 0 term reads its coefficient
// row from b itself, the others from b viewed as [1, 25]: the broadcasts
// differ only in the unit dimension, and the coefficients are one strided
// view of b, so the terms are one reduction.
func.func @unit_dims(%69: tensor<8x1x125xf64>, %5: tensor<25xf64>) -> tensor<8x5x25xf64> {  %11 = stablehlo.slice %5 [0:25:5] : (tensor<25xf64>) -> tensor<5xf64>
  %85 = stablehlo.reshape %5 : (tensor<25xf64>) -> tensor<1x25xf64>
  %86 = stablehlo.slice %69 [0:8, 0:1, 0:125:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
  %87 = stablehlo.reshape %86 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
  %88 = stablehlo.broadcast_in_dim %87, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
  %89 = stablehlo.reshape %88 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %90 = stablehlo.broadcast_in_dim %11, dims = [3] : (tensor<5xf64>) -> tensor<8x5x5x5xf64>
  %93 = stablehlo.reshape %90 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %94 = stablehlo.multiply %89, %93 : tensor<8x5x25xf64>
  %111 = stablehlo.slice %69 [0:8, 0:1, 1:122:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
  %112 = stablehlo.slice %85 [0:1, 1:22:5] : (tensor<1x25xf64>) -> tensor<1x5xf64>
  %113 = stablehlo.reshape %111 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
  %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
  %115 = stablehlo.reshape %114 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %116 = stablehlo.broadcast_in_dim %112, dims = [1, 3] : (tensor<1x5xf64>) -> tensor<8x5x5x5xf64>
  %117 = stablehlo.reshape %116 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %118 = stablehlo.multiply %115, %117 : tensor<8x5x25xf64>
  %119 = stablehlo.add %94, %118 : tensor<8x5x25xf64>
  %137 = stablehlo.slice %69 [0:8, 0:1, 2:123:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
  %138 = stablehlo.slice %85 [0:1, 2:23:5] : (tensor<1x25xf64>) -> tensor<1x5xf64>
  %139 = stablehlo.reshape %137 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
  %140 = stablehlo.broadcast_in_dim %139, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
  %141 = stablehlo.reshape %140 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %142 = stablehlo.broadcast_in_dim %138, dims = [1, 3] : (tensor<1x5xf64>) -> tensor<8x5x5x5xf64>
  %143 = stablehlo.reshape %142 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %144 = stablehlo.multiply %141, %143 : tensor<8x5x25xf64>
  %145 = stablehlo.add %119, %144 : tensor<8x5x25xf64>
  %163 = stablehlo.slice %69 [0:8, 0:1, 3:124:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
  %164 = stablehlo.slice %85 [0:1, 3:24:5] : (tensor<1x25xf64>) -> tensor<1x5xf64>
  %165 = stablehlo.reshape %163 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
  %166 = stablehlo.broadcast_in_dim %165, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
  %167 = stablehlo.reshape %166 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %168 = stablehlo.broadcast_in_dim %164, dims = [1, 3] : (tensor<1x5xf64>) -> tensor<8x5x5x5xf64>
  %169 = stablehlo.reshape %168 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %170 = stablehlo.multiply %167, %169 : tensor<8x5x25xf64>
  %171 = stablehlo.add %145, %170 : tensor<8x5x25xf64>
  %189 = stablehlo.slice %69 [0:8, 0:1, 4:125:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
  %190 = stablehlo.slice %85 [0:1, 4:25:5] : (tensor<1x25xf64>) -> tensor<1x5xf64>
  %191 = stablehlo.reshape %189 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
  %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
  %193 = stablehlo.reshape %192 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %194 = stablehlo.broadcast_in_dim %190, dims = [1, 3] : (tensor<1x5xf64>) -> tensor<8x5x5x5xf64>
  %195 = stablehlo.reshape %194 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %196 = stablehlo.multiply %193, %195 : tensor<8x5x25xf64>
  %197 = stablehlo.add %171, %196 : tensor<8x5x25xf64>
  return %197 : tensor<8x5x25xf64>
}

// CHECK:  func.func @unit_dims(%arg0: tensor<8x1x125xf64>, %arg1: tensor<25xf64>) -> tensor<8x5x25xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<8x1x125xf64>) -> tensor<8x1x25x5xf64>
// CHECK-NEXT:    %1 = stablehlo.transpose %0, dims = [3, 0, 1, 2] : (tensor<8x1x25x5xf64>) -> tensor<5x8x1x25xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<5x8x1x25xf64>) -> tensor<5x8x5x5xf64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3] : (tensor<5x8x5x5xf64>) -> tensor<5x8x5x5x5xf64>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<5x8x5x5x5xf64>) -> tensor<5x8x5x25xf64>
// CHECK-NEXT:    %5 = stablehlo.reshape %arg1 : (tensor<25xf64>) -> tensor<5x5xf64>
// CHECK-NEXT:    %6 = stablehlo.transpose %5, dims = [1, 0] : (tensor<5x5xf64>) -> tensor<5x5xf64>
// CHECK-NEXT:    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 4] : (tensor<5x5xf64>) -> tensor<5x8x5x5x5xf64>
// CHECK-NEXT:    %8 = stablehlo.reshape %7 : (tensor<5x8x5x5x5xf64>) -> tensor<5x8x5x25xf64>
// CHECK-NEXT:    %9 = stablehlo.multiply %4, %8 : tensor<5x8x5x25xf64>
// CHECK-NEXT:    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<5x8x5x25xf64>, tensor<f64>) -> tensor<8x5x25xf64>
// CHECK-NEXT:    return %10 : tensor<8x5x25xf64>
// CHECK-NEXT:  }

// -----

// Coefficients of two different tensors are not one view: left alone.
func.func @other_tensor(%69: tensor<8x1x125xf64>, %5: tensor<25xf64>, %6: tensor<25xf64>) -> tensor<8x5x25xf64> {  %11 = stablehlo.slice %5 [0:25:5] : (tensor<25xf64>) -> tensor<5xf64>
  %85 = stablehlo.reshape %6 : (tensor<25xf64>) -> tensor<1x25xf64>
  %86 = stablehlo.slice %69 [0:8, 0:1, 0:125:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
  %87 = stablehlo.reshape %86 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
  %88 = stablehlo.broadcast_in_dim %87, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
  %89 = stablehlo.reshape %88 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %90 = stablehlo.broadcast_in_dim %11, dims = [3] : (tensor<5xf64>) -> tensor<8x5x5x5xf64>
  %93 = stablehlo.reshape %90 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %94 = stablehlo.multiply %89, %93 : tensor<8x5x25xf64>
  %111 = stablehlo.slice %69 [0:8, 0:1, 1:122:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
  %112 = stablehlo.slice %85 [0:1, 1:22:5] : (tensor<1x25xf64>) -> tensor<1x5xf64>
  %113 = stablehlo.reshape %111 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
  %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
  %115 = stablehlo.reshape %114 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %116 = stablehlo.broadcast_in_dim %112, dims = [1, 3] : (tensor<1x5xf64>) -> tensor<8x5x5x5xf64>
  %117 = stablehlo.reshape %116 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %118 = stablehlo.multiply %115, %117 : tensor<8x5x25xf64>
  %119 = stablehlo.add %94, %118 : tensor<8x5x25xf64>
  %137 = stablehlo.slice %69 [0:8, 0:1, 2:123:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
  %138 = stablehlo.slice %85 [0:1, 2:23:5] : (tensor<1x25xf64>) -> tensor<1x5xf64>
  %139 = stablehlo.reshape %137 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
  %140 = stablehlo.broadcast_in_dim %139, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
  %141 = stablehlo.reshape %140 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %142 = stablehlo.broadcast_in_dim %138, dims = [1, 3] : (tensor<1x5xf64>) -> tensor<8x5x5x5xf64>
  %143 = stablehlo.reshape %142 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %144 = stablehlo.multiply %141, %143 : tensor<8x5x25xf64>
  %145 = stablehlo.add %119, %144 : tensor<8x5x25xf64>
  %163 = stablehlo.slice %69 [0:8, 0:1, 3:124:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
  %164 = stablehlo.slice %85 [0:1, 3:24:5] : (tensor<1x25xf64>) -> tensor<1x5xf64>
  %165 = stablehlo.reshape %163 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
  %166 = stablehlo.broadcast_in_dim %165, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
  %167 = stablehlo.reshape %166 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %168 = stablehlo.broadcast_in_dim %164, dims = [1, 3] : (tensor<1x5xf64>) -> tensor<8x5x5x5xf64>
  %169 = stablehlo.reshape %168 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %170 = stablehlo.multiply %167, %169 : tensor<8x5x25xf64>
  %171 = stablehlo.add %145, %170 : tensor<8x5x25xf64>
  %189 = stablehlo.slice %69 [0:8, 0:1, 4:125:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
  %190 = stablehlo.slice %85 [0:1, 4:25:5] : (tensor<1x25xf64>) -> tensor<1x5xf64>
  %191 = stablehlo.reshape %189 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
  %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
  %193 = stablehlo.reshape %192 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %194 = stablehlo.broadcast_in_dim %190, dims = [1, 3] : (tensor<1x5xf64>) -> tensor<8x5x5x5xf64>
  %195 = stablehlo.reshape %194 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
  %196 = stablehlo.multiply %193, %195 : tensor<8x5x25xf64>
  %197 = stablehlo.add %171, %196 : tensor<8x5x25xf64>
  return %197 : tensor<8x5x25xf64>
}

// CHECK:  func.func @other_tensor(%arg0: tensor<8x1x125xf64>, %arg1: tensor<25xf64>, %arg2: tensor<25xf64>) -> tensor<8x5x25xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:25:5] : (tensor<25xf64>) -> tensor<5xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg2 : (tensor<25xf64>) -> tensor<1x25xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [0:8, 0:1, 0:125:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
// CHECK-NEXT:    %5 = stablehlo.reshape %4 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<5xf64>) -> tensor<8x5x5x5xf64>
// CHECK-NEXT:    %7 = stablehlo.reshape %6 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
// CHECK-NEXT:    %8 = stablehlo.multiply %5, %7 : tensor<8x5x25xf64>
// CHECK-NEXT:    %9 = stablehlo.slice %arg0 [0:8, 0:1, 1:122:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
// CHECK-NEXT:    %10 = stablehlo.slice %1 [0:1, 1:22:5] : (tensor<1x25xf64>) -> tensor<1x5xf64>
// CHECK-NEXT:    %11 = stablehlo.reshape %9 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
// CHECK-NEXT:    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
// CHECK-NEXT:    %13 = stablehlo.reshape %12 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
// CHECK-NEXT:    %14 = stablehlo.broadcast_in_dim %10, dims = [1, 3] : (tensor<1x5xf64>) -> tensor<8x5x5x5xf64>
// CHECK-NEXT:    %15 = stablehlo.reshape %14 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
// CHECK-NEXT:    %16 = stablehlo.multiply %13, %15 : tensor<8x5x25xf64>
// CHECK-NEXT:    %17 = stablehlo.add %8, %16 : tensor<8x5x25xf64>
// CHECK-NEXT:    %18 = stablehlo.slice %arg0 [0:8, 0:1, 2:123:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
// CHECK-NEXT:    %19 = stablehlo.slice %1 [0:1, 2:23:5] : (tensor<1x25xf64>) -> tensor<1x5xf64>
// CHECK-NEXT:    %20 = stablehlo.reshape %18 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
// CHECK-NEXT:    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
// CHECK-NEXT:    %22 = stablehlo.reshape %21 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
// CHECK-NEXT:    %23 = stablehlo.broadcast_in_dim %19, dims = [1, 3] : (tensor<1x5xf64>) -> tensor<8x5x5x5xf64>
// CHECK-NEXT:    %24 = stablehlo.reshape %23 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
// CHECK-NEXT:    %25 = stablehlo.multiply %22, %24 : tensor<8x5x25xf64>
// CHECK-NEXT:    %26 = stablehlo.add %17, %25 : tensor<8x5x25xf64>
// CHECK-NEXT:    %27 = stablehlo.slice %arg0 [0:8, 0:1, 3:124:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
// CHECK-NEXT:    %28 = stablehlo.slice %1 [0:1, 3:24:5] : (tensor<1x25xf64>) -> tensor<1x5xf64>
// CHECK-NEXT:    %29 = stablehlo.reshape %27 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
// CHECK-NEXT:    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
// CHECK-NEXT:    %31 = stablehlo.reshape %30 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
// CHECK-NEXT:    %32 = stablehlo.broadcast_in_dim %28, dims = [1, 3] : (tensor<1x5xf64>) -> tensor<8x5x5x5xf64>
// CHECK-NEXT:    %33 = stablehlo.reshape %32 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
// CHECK-NEXT:    %34 = stablehlo.multiply %31, %33 : tensor<8x5x25xf64>
// CHECK-NEXT:    %35 = stablehlo.add %26, %34 : tensor<8x5x25xf64>
// CHECK-NEXT:    %36 = stablehlo.slice %arg0 [0:8, 0:1, 4:125:5] : (tensor<8x1x125xf64>) -> tensor<8x1x25xf64>
// CHECK-NEXT:    %37 = stablehlo.slice %1 [0:1, 4:25:5] : (tensor<1x25xf64>) -> tensor<1x5xf64>
// CHECK-NEXT:    %38 = stablehlo.reshape %36 : (tensor<8x1x25xf64>) -> tensor<8x5x5xf64>
// CHECK-NEXT:    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2] : (tensor<8x5x5xf64>) -> tensor<8x5x5x5xf64>
// CHECK-NEXT:    %40 = stablehlo.reshape %39 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
// CHECK-NEXT:    %41 = stablehlo.broadcast_in_dim %37, dims = [1, 3] : (tensor<1x5xf64>) -> tensor<8x5x5x5xf64>
// CHECK-NEXT:    %42 = stablehlo.reshape %41 : (tensor<8x5x5x5xf64>) -> tensor<8x5x25xf64>
// CHECK-NEXT:    %43 = stablehlo.multiply %40, %42 : tensor<8x5x25xf64>
// CHECK-NEXT:    %44 = stablehlo.add %35, %43 : tensor<8x5x25xf64>
// CHECK-NEXT:    return %44 : tensor<8x5x25xf64>
// CHECK-NEXT:  }
