// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Exact reproducer from Enzyme-JAX issue #1005:
// Pushing conj through multiply with a purely real constant cancels the inner conj,
// exposing CSE and leaving only a single chlo.conj.
// CHECK-LABEL: func.func @issue_1005_conj(
// CHECK: %[[FFT1:.*]] = stablehlo.fft {{.*}} type =  FFT
// CHECK-NOT: chlo.conj
// CHECK: %[[SCALED:.*]] = stablehlo.multiply %{{.*}}, %[[FFT1]]
// CHECK-NEXT: %[[DBL:.*]] = stablehlo.add %[[SCALED]], %[[SCALED]]
// CHECK-NEXT: %[[CONJ_ONLY:.*]] = chlo.conj %[[DBL]]
// CHECK-NOT: chlo.conj
// CHECK: stablehlo.fft %[[CONJ_ONLY]], type =  FFT
func.func @issue_1005_conj(%arg0: tensor<5x3x4xf32>) -> tensor<5x3x4xf32> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<4x5x3xf32>
  %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x5x3xf32>
  %cst_1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4x5x3xcomplex<f32>>
  %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<5x3x4xf32>) -> tensor<4x5x3xf32>
  %1 = stablehlo.complex %0, %cst_0 : tensor<4x5x3xcomplex<f32>>
  %2 = stablehlo.fft %1, type =  FFT, length = [5, 3] : (tensor<4x5x3xcomplex<f32>>) -> tensor<4x5x3xcomplex<f32>>
  %3 = chlo.conj %2 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %4 = stablehlo.complex %cst, %cst_0 : tensor<4x5x3xcomplex<f32>>
  %5 = chlo.conj %4 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %6 = stablehlo.add %cst_1, %5 : tensor<4x5x3xcomplex<f32>>
  %7 = chlo.conj %6 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %8 = stablehlo.multiply %7, %3 : tensor<4x5x3xcomplex<f32>>
  %9 = chlo.conj %8 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %10 = stablehlo.add %cst_1, %9 : tensor<4x5x3xcomplex<f32>>
  %11 = stablehlo.multiply %7, %2 : tensor<4x5x3xcomplex<f32>>
  %12 = chlo.conj %11 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %13 = stablehlo.add %cst_1, %12 : tensor<4x5x3xcomplex<f32>>
  %14 = chlo.conj %13 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %15 = chlo.conj %14 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %16 = chlo.conj %15 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %17 = stablehlo.add %10, %16 : tensor<4x5x3xcomplex<f32>>
  %18 = chlo.conj %17 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %19 = chlo.conj %18 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %20 = stablehlo.add %cst_1, %19 : tensor<4x5x3xcomplex<f32>>
  %21 = chlo.conj %20 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %22 = stablehlo.fft %21, type =  FFT, length = [5, 3] : (tensor<4x5x3xcomplex<f32>>) -> tensor<4x5x3xcomplex<f32>>
  %23 = chlo.conj %22 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %24 = stablehlo.add %cst_1, %23 : tensor<4x5x3xcomplex<f32>>
  %25 = chlo.conj %24 : tensor<4x5x3xcomplex<f32>> -> tensor<4x5x3xcomplex<f32>>
  %26 = stablehlo.real %25 : (tensor<4x5x3xcomplex<f32>>) -> tensor<4x5x3xf32>
  %27 = stablehlo.transpose %26, dims = [1, 2, 0] : (tensor<4x5x3xf32>) -> tensor<5x3x4xf32>
  return %27 : tensor<5x3x4xf32>
}

// Multi-op algebraic + analytic cone with DAG sharing and complex constant leaf.
// CHECK-LABEL: func.func @conj_cone_algebraic_analytic(
// CHECK-NOT: chlo.conj
// CHECK: %[[CST:.*]] = stablehlo.constant dense<(2.000000e+00,-3.000000e+00)> : tensor<4xcomplex<f32>>
// CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
// CHECK: %[[EXP:.*]] = stablehlo.exponential %[[ADD]]
// CHECK: %[[MUL:.*]] = stablehlo.multiply %[[EXP]], %[[CST]]
// CHECK: %[[SUB:.*]] = stablehlo.subtract %[[MUL]], %[[ADD]]
// CHECK: return %[[SUB]] : tensor<4xcomplex<f32>>
func.func @conj_cone_algebraic_analytic(%arg0: tensor<4xcomplex<f32>>, %arg1: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %cst = stablehlo.constant dense<(2.000000e+00,3.000000e+00)> : tensor<4xcomplex<f32>>
  %c0 = chlo.conj %arg0 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
  %c1 = chlo.conj %arg1 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
  %add = stablehlo.add %c0, %c1 : tensor<4xcomplex<f32>>
  %exp = stablehlo.exponential %add : tensor<4xcomplex<f32>>
  %mul = stablehlo.multiply %exp, %cst : tensor<4xcomplex<f32>>
  %sub = stablehlo.subtract %mul, %add : tensor<4xcomplex<f32>>
  %res = chlo.conj %sub : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
  return %res : tensor<4xcomplex<f32>>
}

// Structural ops (select) and dot_general with real convert leaf.
// CHECK-LABEL: func.func @conj_structural_and_dot(
// CHECK-NOT: chlo.conj
// CHECK: %[[CONV:.*]] = stablehlo.convert %arg2
// CHECK: %[[DOT:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[SEL:.*]] = stablehlo.select %arg3, %[[DOT]], %[[CONV]]
// CHECK: return %[[SEL]]
func.func @conj_structural_and_dot(%arg0: tensor<4x3xcomplex<f32>>, %arg1: tensor<3x5xcomplex<f32>>, %arg2: tensor<4x5xf32>, %arg3: tensor<4x5xi1>) -> tensor<4x5xcomplex<f32>> {
  %ca = chlo.conj %arg0 : tensor<4x3xcomplex<f32>> -> tensor<4x3xcomplex<f32>>
  %cb = chlo.conj %arg1 : tensor<3x5xcomplex<f32>> -> tensor<3x5xcomplex<f32>>
  %conv = stablehlo.convert %arg2 : (tensor<4x5xf32>) -> tensor<4x5xcomplex<f32>>
  %dot = stablehlo.dot_general %ca, %cb, contracting_dims = [1] x [0] : (tensor<4x3xcomplex<f32>>, tensor<3x5xcomplex<f32>>) -> tensor<4x5xcomplex<f32>>
  %sel = stablehlo.select %arg3, %dot, %conv : tensor<4x5xi1>, tensor<4x5xcomplex<f32>>
  %res = chlo.conj %sel : tensor<4x5xcomplex<f32>> -> tensor<4x5xcomplex<f32>>
  return %res : tensor<4x5xcomplex<f32>>
}

// Negative test 1: branch-cut op (sqrt) must NOT commute with conj across complex operand!
// CHECK-LABEL: func.func @conj_sqrt_branch_cut_negative(
// CHECK: %[[C0:.*]] = chlo.conj %arg0
// CHECK: %[[SQ:.*]] = stablehlo.sqrt %[[C0]]
// CHECK: %[[RES:.*]] = chlo.conj %[[SQ]]
// CHECK: return %[[RES]]
func.func @conj_sqrt_branch_cut_negative(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %c0 = chlo.conj %arg0 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
  %sq = stablehlo.sqrt %c0 : tensor<4xcomplex<f32>>
  %res = chlo.conj %sq : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
  return %res : tensor<4xcomplex<f32>>
}

// Negative test 2: un-conjugated complex leaf prevents conj elimination.
// CHECK-LABEL: func.func @conj_unconjugated_leaf_negative(
// CHECK: %[[C0:.*]] = chlo.conj %arg0
// CHECK: %[[MUL:.*]] = stablehlo.multiply %[[C0]], %arg1
// CHECK: %[[RES:.*]] = chlo.conj %[[MUL]]
// CHECK: return %[[RES]]
func.func @conj_unconjugated_leaf_negative(%arg0: tensor<4xcomplex<f32>>, %arg1: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %c0 = chlo.conj %arg0 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
  %mul = stablehlo.multiply %c0, %arg1 : tensor<4xcomplex<f32>>
  %res = chlo.conj %mul : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
  return %res : tensor<4xcomplex<f32>>
}

// Negative test 3: interior op with external user must not be duplicated.
// CHECK-LABEL: func.func @conj_escaped_interior_negative(
// CHECK: %[[ADD:.*]] = stablehlo.add
// CHECK: %[[CONJ:.*]] = chlo.conj %[[ADD]]
// CHECK: return %[[CONJ]], %[[ADD]]
func.func @conj_escaped_interior_negative(%arg0: tensor<4xcomplex<f32>>, %arg1: tensor<4xcomplex<f32>>) -> (tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>) {
  %c0 = chlo.conj %arg0 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
  %c1 = chlo.conj %arg1 : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
  %add = stablehlo.add %c0, %c1 : tensor<4xcomplex<f32>>
  %res = chlo.conj %add : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
  return %res, %add : tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>
}
