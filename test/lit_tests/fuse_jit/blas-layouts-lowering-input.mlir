// RUN: enzymexlamlir-opt --lower-enzymexla-blas={backend=cpu,blas_int_width=64} --enzyme-hlo-opt %s | FileCheck %s
// To print the lowered IR, run this command without the trailing `| FileCheck %s`.

module {
  func.func @blas_chain(
      %a: tensor<4x4xf64>, %b: tensor<4x4xf64>, %c: tensor<4x4xf64>,
      %alpha: tensor<f64>, %beta: tensor<f64>) -> tensor<4x4xf64> {
    %0 = enzymexla.blas.symm %a, %b, %c, %alpha, %beta <{side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>}> : (tensor<4x4xf64>, tensor<4x4xf64>, tensor<4x4xf64>, tensor<f64>, tensor<f64>) -> tensor<4x4xf64>
    %1 = enzymexla.blas.syrk %a, %0, %alpha, %beta <{uplo = #enzymexla.uplo<U>, output_uplo = #enzymexla.uplo<U>}> : (tensor<4x4xf64>, tensor<4x4xf64>, tensor<f64>, tensor<f64>) -> tensor<4x4xf64>
    return %1 : tensor<4x4xf64>
  }
}

// CHECK-COUNT-2: enzymexla.jit_call
