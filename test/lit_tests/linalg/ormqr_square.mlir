// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = enzymexla.lapack.ormqr %arg0, %arg1, %arg2 {side = #enzymexla.LapackSide<"left">} : (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0: tensor<64x64xf32>
  }
}

module {
  func.func @main(%arg0: tensor<64x64xf64>, %arg1: tensor<64xf64>, %arg2: tensor<64x64xf64>) -> tensor<64x64xf64> {
    %0 = enzymexla.lapack.ormqr %arg0, %arg1, %arg2 {side = #enzymexla.LapackSide<"left">} : (tensor<64x64xf64>, tensor<64xf64>, tensor<64x64xf64>) -> tensor<64x64xf64>
    return %0: tensor<64x64xf64>
  }
}

module {
  func.func @main(%arg0: tensor<64x64xcomplex<f32>>, %arg1: tensor<64xcomplex<f32>>, %arg2: tensor<64x64xcomplex<f32>>) -> tensor<64x64xcomplex<f32>> {
    %0 = enzymexla.lapack.ormqr %arg0, %arg1, %arg2 {side = #enzymexla.LapackSide<"left">} : (tensor<64x64xcomplex<f32>>, tensor<64xcomplex<f32>>, tensor<64x64xcomplex<f32>>) -> tensor<64x64xcomplex<f32>>
    return %0: tensor<64x64xcomplex<f32>>
  }
}

module {
  func.func @main(%arg0: tensor<64x64xcomplex<f64>>, %arg1: tensor<64xcomplex<f64>>, %arg2: tensor<64x64xcomplex<f64>>) -> tensor<64x64xcomplex<f64>> {
    %0 = enzymexla.lapack.ormqr %arg0, %arg1, %arg2 {side = #enzymexla.LapackSide<"left">} : (tensor<64x64xcomplex<f64>>, tensor<64xcomplex<f64>>, tensor<64x64xcomplex<f64>>) -> tensor<64x64xcomplex<f64>>
    return %0: tensor<64x64xcomplex<f64>>
  }
}
