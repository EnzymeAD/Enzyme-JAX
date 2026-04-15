// RUN: enzymexlamlir-opt --lower-enzymexla-linalg="backend=cpu" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --lower-enzymexla-linalg="backend=cuda" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --lower-enzymexla-linalg="backend=tpu" --enzyme-hlo-opt %s | FileCheck %s --check-prefix=TPU


func.func @trisolve(%arg0: tensor<63xf32>, %arg1: tensor<64xf32>, %arg2: tensor<63xf32>, %arg3: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = enzymexla.linalg.tridiagonal_solve %arg0, %arg1, %arg2, %arg3: (tensor<63xf32>, tensor<64xf32>, tensor<63xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
}

// CHECK: enzymexla.linalg.tridiagonal_solve
