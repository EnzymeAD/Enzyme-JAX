// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA

module {
    func.func @main([[INPUT:%[a-z0-9]+]]: tensor<64x64xf32>, [[TAU:%[a-z0-9]+]]: tensor<64xf32>) -> tensor<64x64xf32> {
        %0 = enzymexla.lapack.orgqr [[INPUT]], [[TAU]] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64x64xf32>
        return %0 : tensor<64x64xf32>
    }
}

// CPU:

// CUDA:

module {
  func.func @main([[INPUT:%[a-z0-9]+]]: tensor<64x64xf64>, [[TAU:%[a-z0-9]+]]: tensor<64xf64>) -> tensor<64x64xf64> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_dorgqr_[[WRAPPER_ID:[0-9]+]]
    %0:3 = enzymexla.lapack.geqrf %arg0 : (tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xf64>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<64x64xf64>, tensor<64xf64>, tensor<i64>
  }
}

module {
  func.func @main([[INPUT:%[a-z0-9]+]]: tensor<64x64xcomplex<f32>>, [[TAU:%[a-z0-9]+]]: tensor<64xcomplex<f32>>) -> tensor<64x64xcomplex<f32>> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_cungqr_[[WRAPPER_ID:[0-9]+]]
    %0:3 = enzymexla.lapack.geqrf %arg0 : (tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xcomplex<f32>>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<64x64xcomplex<f32>>, tensor<64xcomplex<f32>>, tensor<i64>
  }
}

module {
  func.func @main([[INPUT:%[a-z0-9]+]]: tensor<64x64xcomplex<f64>>, [[TAU:%[a-z0-9]+]]: tensor<64xcomplex<f64>>) -> tensor<64x64xcomplex<f64>> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_zungqr_[[WRAPPER_ID:[0-9]+]]
    %0:3 = enzymexla.lapack.geqrf %arg0 : (tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xcomplex<f64>>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<64x64xcomplex<f64>>, tensor<64xcomplex<f64>>, tensor<i64>
  }
}
