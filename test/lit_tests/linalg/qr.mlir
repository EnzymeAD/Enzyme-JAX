// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main_default(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    // CPU: %0:2 = enzymexla.lapack.geqrf %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>)
    // CPU-NEXT: %1 = enzymexla.lapack.orgqr %0#0, %0#1 : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64x64xf32>
    %0:2 = enzymexla.linalg.qr %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>)
    return %0#0, %0#1 : tensor<64x64xf32>, tensor<64x64xf32>
  }
}

module {
  func.func @main_geqrf(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    // CPU: %0:2 = enzymexla.lapack.geqrf %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>)
    // CPU-NEXT: %1 = enzymexla.lapack.orgqr %0#0, %0#1 : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64x64xf32>
    %0:2 = enzymexla.linalg.qr %arg0 {algorithm = geqrf} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>)
    return %0#0, %0#1 : tensor<64x64xf32>, tensor<64x64xf32>
  }
}

// TODO test geqrt
