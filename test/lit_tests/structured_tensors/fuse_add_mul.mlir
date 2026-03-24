// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dot_general_to_symm;fuse_add_into_symm;fuse_mul_into_symm" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @test1(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %alpha = stablehlo.constant dense<2.0> : tensor<f32>
    %beta = stablehlo.constant dense<3.0> : tensor<f32>
    %scalar = stablehlo.constant dense<3.0> : tensor<f32>
    %scalar_bcast = stablehlo.broadcast_in_dim %scalar, dims = []: (tensor<f32>) -> tensor<64x64xf32>
    %0 = enzymexla.blas.symm %arg0, %arg1, %arg2, %alpha, %beta {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>} : (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
    %1 = stablehlo.multiply %scalar_bcast, %0 : tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
}


// CHECK:  func.func @test1(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CHECK-DAG:    %[[cst2:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK-DAG:    %[[cst3:.+]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.multiply %[[cst3]], %[[cst3]] : tensor<f32>
// CHECK-NEXT:    %1 = stablehlo.multiply %[[cst2]], %[[cst3]] : tensor<f32>
// CHECK-NEXT:    %2 = enzymexla.blas.symm %arg0, %arg1, %arg2, %1, %0 {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>} : (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:    return %2 : tensor<64x64xf32>
// CHECK-NEXT:  }

func.func @test2(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %alpha = stablehlo.constant dense<2.0> : tensor<f32>
    %beta = stablehlo.constant dense<3.0> : tensor<f32>
    %const = stablehlo.constant dense<4.0> : tensor<f32>
    %const_bcast = stablehlo.broadcast_in_dim %const, dims = []: (tensor<f32>) -> tensor<64x64xf32>
    %0 = enzymexla.blas.symm %arg0, %arg1, %arg2, %alpha, %beta {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>} : (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
    %1 = stablehlo.add %const_bcast, %0 : tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
}

// CHECK:  func.func @test2(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CHECK-DAG:    %[[cst:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:    %[[cst2:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK-DAG:    %[[cst3:.+]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK-DAG:    %[[cst4:.+]] = stablehlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %[[cst4]], dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %[[cst3]], dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:    %2 = stablehlo.multiply %arg2, %1 : tensor<64x64xf32>
// CHECK-NEXT:    %3 = stablehlo.add %2, %0 : tensor<64x64xf32>
// CHECK-NEXT:    %4 = enzymexla.blas.symm %arg0, %arg1, %3, %[[cst2]], %[[cst]] {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>} : (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:    return %4 : tensor<64x64xf32>
// CHECK-NEXT:  }

func.func @test3(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %t = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %A_symm = stablehlo.add %arg0, %t : tensor<64x64xf32>
    %alpha = stablehlo.constant dense<2.0> : tensor<f32>
    %alpha_bcast = stablehlo.broadcast_in_dim %alpha, dims = []: (tensor<f32>) -> tensor<64x64xf32>
    %A_B = stablehlo.dot_general %A_symm, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %A_B_alpha = stablehlo.multiply %A_B, %alpha_bcast : tensor<64x64xf32>

    %beta = stablehlo.constant dense<3.0> : tensor<f32>
    %beta_bcast = stablehlo.broadcast_in_dim %beta, dims = []: (tensor<f32>) -> tensor<64x64xf32>
    %C_beta = stablehlo.multiply %arg2, %beta_bcast : tensor<64x64xf32>

    %res = stablehlo.add %A_B_alpha, %C_beta : tensor<64x64xf32>
    return %res : tensor<64x64xf32>
}

// CHECK: func.func @test3(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CHECK-DAG:   %[[cstC:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-DAG:   %[[cst1:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:   %[[cstm0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:   %[[cst3:.+]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK-DAG:   %[[cst2:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %1 = stablehlo.add %arg0, %0 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<64x64xf32>
// CHECK-NEXT:   %2 = stablehlo.multiply %[[cstm0]], %[[cst2]] : tensor<f32>
// CHECK-NEXT:   %3 = stablehlo.multiply %[[cst1]], %[[cst2]] : tensor<f32>
// CHECK-NEXT:   %4 = stablehlo.broadcast_in_dim %[[cst3]], dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %5 = stablehlo.multiply %arg2, %4 : tensor<64x64xf32>
// CHECK-NEXT:   %6 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   %7 = stablehlo.multiply %[[cstC]], %6 : tensor<64x64xf32>
// CHECK-NEXT:   %8 = stablehlo.add %7, %5 : tensor<64x64xf32>
// CHECK-NEXT:   %9 = enzymexla.blas.symm %1, %arg1, %8, %3, %[[cst1]] {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<F>} : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>) -> tensor<64x64xf32>
// CHECK-NEXT:   return %9 : tensor<64x64xf32>
// CHECK-NEXT: }
// CHECK-NEXT: }
