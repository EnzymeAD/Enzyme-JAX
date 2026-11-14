// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Issue #1355: remove unnecessary intermediate singleton dims in broadcast_in_dim before reshape
// CHECK-LABEL: func.func @reshape_bcast_remove_intermediate_ones
// CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 3, 4] : (tensor<256x4x128x32xf32>) -> tensor<256x4x16x128x32xf32>
// CHECK-NEXT: %1 = stablehlo.reshape %0 : (tensor<256x4x16x128x32xf32>) -> tensor<256x64x128x32xf32>
// CHECK-NEXT: return %1 : tensor<256x64x128x32xf32>
func.func @reshape_bcast_remove_intermediate_ones(%arg0: tensor<256x4x128x32xf32>) -> tensor<256x64x128x32xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 4, 6] : (tensor<256x4x128x32xf32>) -> tensor<256x1x4x16x128x1x32x1xf32>
  %1 = stablehlo.reshape %0 : (tensor<256x1x4x16x128x1x32x1xf32>) -> tensor<256x64x128x32xf32>
  return %1 : tensor<256x64x128x32xf32>
}

// Issue #1362: DUS(splat or broadcast, update) -> pad(update, 0-dim splat)
// CHECK-LABEL: func.func @dus_splat_to_pad
// CHECK-NEXT: %cst = stablehlo.constant dense<7.000000e+00> : tensor<f32>
// CHECK-NEXT: %0 = stablehlo.pad %arg0, %cst, low = [1, 2], high = [1, 0], interior = [0, 0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<4x5xf32>
// CHECK-NEXT: return %0 : tensor<4x5xf32>
func.func @dus_splat_to_pad(%u: tensor<2x3xf32>) -> tensor<4x5xf32> {
  %base = stablehlo.constant dense<7.0> : tensor<4x5xf32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %base, %u, %c1, %c2 : (tensor<4x5xf32>, tensor<2x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// CHECK-LABEL: func.func @dus_broadcast_op_clamped_to_pad
// CHECK-NEXT: %0 = stablehlo.pad %arg1, %arg0, low = [0, 2], high = [2, 0], interior = [0, 0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<4x5xf32>
// CHECK-NEXT: return %0 : tensor<4x5xf32>
func.func @dus_broadcast_op_clamped_to_pad(%s: tensor<f32>, %u: tensor<2x3xf32>) -> tensor<4x5xf32> {
  %base = stablehlo.broadcast %s, sizes = [4, 5] : (tensor<f32>) -> tensor<4x5xf32>
  %cm3 = stablehlo.constant dense<-3> : tensor<i32>
  %c99 = stablehlo.constant dense<99> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %base, %u, %cm3, %c99 : (tensor<4x5xf32>, tensor<2x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// CHECK-LABEL: func.func @dus_broadcast_in_dim_1d_to_pad
// CHECK-NEXT: %0 = stablehlo.reshape %arg0 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT: %1 = stablehlo.pad %arg1, %0, low = [1, 2], high = [1, 0], interior = [0, 0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<4x5xf32>
// CHECK-NEXT: return %1 : tensor<4x5xf32>
func.func @dus_broadcast_in_dim_1d_to_pad(%s: tensor<1xf32>, %u: tensor<2x3xf32>) -> tensor<4x5xf32> {
  %base = stablehlo.broadcast_in_dim %s, dims = [0] : (tensor<1xf32>) -> tensor<4x5xf32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %base, %u, %c1, %c2 : (tensor<4x5xf32>, tensor<2x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// Issue #1652: syrk output_uplo refinement for trmm, trsm, triangular_solve, cholesky
// CHECK-LABEL: func.func @syrk_user_triangular_solve
// CHECK: enzymexla.blas.syrk {{.*}}output_uplo = L
// CHECK-NEXT: "stablehlo.triangular_solve"
func.func @syrk_user_triangular_solve(%arg0: tensor<5x4xf32>, %b: tensor<4x3xf32>) -> tensor<4x3xf32> {
  %cst_0 = stablehlo.constant dense<0.0> : tensor<f32>
  %cst_1 = stablehlo.constant dense<1.0> : tensor<f32>
  %cst_c = stablehlo.constant dense<0.0> : tensor<4x4xf32>
  %0 = enzymexla.blas.syrk %arg0, %cst_c, %cst_1, %cst_0 <{output_uplo = #enzymexla.uplo<F>, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<F>}> : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
  %1 = "stablehlo.triangular_solve"(%0, %b) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  return %1 : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @syrk_user_trmm
// CHECK: enzymexla.blas.syrk {{.*}}output_uplo = U
// CHECK-NEXT: enzymexla.blas.trmm
func.func @syrk_user_trmm(%arg0: tensor<5x4xf32>, %b: tensor<4x3xf32>) -> tensor<4x3xf32> {
  %cst_0 = stablehlo.constant dense<0.0> : tensor<f32>
  %cst_1 = stablehlo.constant dense<1.0> : tensor<f32>
  %cst_c = stablehlo.constant dense<0.0> : tensor<4x4xf32>
  %0 = enzymexla.blas.syrk %arg0, %cst_c, %cst_1, %cst_0 <{output_uplo = #enzymexla.uplo<F>, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<F>}> : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
  %1 = enzymexla.blas.trmm %0, %b, %cst_1 <{side = #enzymexla.side<left>, transpose = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<U>}> : (tensor<4x4xf32>, tensor<4x3xf32>, tensor<f32>) -> tensor<4x3xf32>
  return %1 : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @syrk_user_trsm
// CHECK: enzymexla.blas.syrk {{.*}}output_uplo = L
// CHECK-NEXT: enzymexla.blas.trsm
func.func @syrk_user_trsm(%arg0: tensor<5x4xf32>, %b: tensor<4x3xf32>) -> tensor<4x3xf32> {
  %cst_0 = stablehlo.constant dense<0.0> : tensor<f32>
  %cst_1 = stablehlo.constant dense<1.0> : tensor<f32>
  %cst_c = stablehlo.constant dense<0.0> : tensor<4x4xf32>
  %0 = enzymexla.blas.syrk %arg0, %cst_c, %cst_1, %cst_0 <{output_uplo = #enzymexla.uplo<F>, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<F>}> : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
  %1 = enzymexla.blas.trsm %cst_1, %0, %b <{side = #enzymexla.side<left>, transa = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<L>}> : (tensor<f32>, tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  return %1 : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @syrk_user_cholesky
// CHECK: enzymexla.blas.syrk {{.*}}output_uplo = U
// CHECK-NEXT: stablehlo.cholesky
func.func @syrk_user_cholesky(%arg0: tensor<5x4xf32>) -> tensor<4x4xf32> {
  %cst_0 = stablehlo.constant dense<0.0> : tensor<f32>
  %cst_1 = stablehlo.constant dense<1.0> : tensor<f32>
  %cst_c = stablehlo.constant dense<0.0> : tensor<4x4xf32>
  %0 = enzymexla.blas.syrk %arg0, %cst_c, %cst_1, %cst_0 <{output_uplo = #enzymexla.uplo<F>, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<F>}> : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
  %1 = stablehlo.cholesky %0, lower = false : tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Conflicting U and L triangular users must keep output_uplo = F
// CHECK-LABEL: func.func @syrk_conflicting_trmm_trisolve
// CHECK: enzymexla.blas.syrk {{.*}}output_uplo = F
func.func @syrk_conflicting_trmm_trisolve(%arg0: tensor<5x4xf32>, %b: tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<4x3xf32>) {
  %cst_0 = stablehlo.constant dense<0.0> : tensor<f32>
  %cst_1 = stablehlo.constant dense<1.0> : tensor<f32>
  %cst_c = stablehlo.constant dense<0.0> : tensor<4x4xf32>
  %0 = enzymexla.blas.syrk %arg0, %cst_c, %cst_1, %cst_0 <{output_uplo = #enzymexla.uplo<F>, transpose = #enzymexla.transpose<transpose>, uplo = #enzymexla.uplo<F>}> : (tensor<5x4xf32>, tensor<4x4xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4xf32>
  %1 = enzymexla.blas.trmm %0, %b, %cst_1 <{side = #enzymexla.side<left>, transpose = #enzymexla.transpose<none>, uplo = #enzymexla.uplo<U>}> : (tensor<4x4xf32>, tensor<4x3xf32>, tensor<f32>) -> tensor<4x3xf32>
  %2 = "stablehlo.triangular_solve"(%0, %b) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  return %1, %2 : tensor<4x3xf32>, tensor<4x3xf32>
}
