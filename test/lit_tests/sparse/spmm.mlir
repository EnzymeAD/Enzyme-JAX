// RUN: enzymexlamlir-opt --lower-enzymexla-sparse %s | FileCheck %s

#csr = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 64, crdWidth = 64 }>

// Constant alpha/beta: fully fused into a single accumulating library call
// with C aliased to the output.
// CHECK-LABEL: func.func @fused
// CHECK-NOT: sparse_tensor
// CHECK: %[[Y:.+]] = stablehlo.custom_call @reactant_csr_matmul_acc(%arg0, %arg1, %arg2, %arg3, %arg4)
// CHECK-SAME: api_version = 4 : i32
// CHECK-SAME: backend_config = {alpha = 2.000000e+00 : f64, beta = 3.000000e+00 : f64, index_base = 0 : i64, m = 10 : i64, n = 8 : i64, transpose = 0 : i64}
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 4, operand_tuple_indices = []>]
// CHECK-SAME: (tensor<11xi64>, tensor<20xi64>, tensor<20xf64>, tensor<8xf64>, tensor<10xf64>) -> tensor<10xf64>
// CHECK-NOT: sparse_tensor
// CHECK: return %[[Y]]
func.func @fused(%rowptr: tensor<11xi64>, %colind: tensor<20xi64>, %nzval: tensor<20xf64>, %x: tensor<8xf64>, %C: tensor<10xf64>) -> tensor<10xf64> {
  %alpha = stablehlo.constant dense<2.0> : tensor<f64>
  %beta = stablehlo.constant dense<3.0> : tensor<f64>
  %A = sparse_tensor.assemble (%rowptr, %colind), %nzval : (tensor<11xi64>, tensor<20xi64>), tensor<20xf64> to tensor<10x8xf64, #csr>
  %y = enzymexla.sparse.spmm %alpha, %A, %x, %beta, %C : (tensor<f64>, tensor<10x8xf64, #csr>, tensor<8xf64>, tensor<f64>, tensor<10xf64>) -> tensor<10xf64>
  return %y : tensor<10xf64>
}

// Constant beta == 0: no accumulation, alpha folded into the plain call and
// C unused.
// CHECK-LABEL: func.func @beta_zero
// CHECK-NOT: sparse_tensor
// CHECK: %[[Y:.+]] = stablehlo.custom_call @reactant_csr_matmul(%arg0, %arg1, %arg2, %arg3)
// CHECK-SAME: backend_config = {alpha = 2.000000e+00 : f64, index_base = 0 : i64, m = 10 : i64, n = 8 : i64, transpose = 0 : i64}
// CHECK-SAME: (tensor<11xi64>, tensor<20xi64>, tensor<20xf64>, tensor<8x3xf64>) -> tensor<10x3xf64>
// CHECK-NOT: sparse_tensor
// CHECK: return %[[Y]]
func.func @beta_zero(%rowptr: tensor<11xi64>, %colind: tensor<20xi64>, %nzval: tensor<20xf64>, %B: tensor<8x3xf64>, %C: tensor<10x3xf64>) -> tensor<10x3xf64> {
  %alpha = stablehlo.constant dense<2.0> : tensor<f64>
  %beta = stablehlo.constant dense<0.0> : tensor<f64>
  %A = sparse_tensor.assemble (%rowptr, %colind), %nzval : (tensor<11xi64>, tensor<20xi64>), tensor<20xf64> to tensor<10x8xf64, #csr>
  %y = enzymexla.sparse.spmm %alpha, %A, %B, %beta, %C : (tensor<f64>, tensor<10x8xf64, #csr>, tensor<8x3xf64>, tensor<f64>, tensor<10x3xf64>) -> tensor<10x3xf64>
  return %y : tensor<10x3xf64>
}

// Runtime alpha/beta: plain library call plus explicit stablehlo scaling and
// accumulation.
// CHECK-LABEL: func.func @runtime_scalars
// CHECK-NOT: sparse_tensor
// CHECK: %[[CC:.+]] = stablehlo.custom_call @reactant_csr_matmul(%arg0, %arg1, %arg2, %arg3)
// CHECK-SAME: backend_config = {alpha = 1.000000e+00 : f64, index_base = 0 : i64, m = 10 : i64, n = 8 : i64, transpose = 0 : i64}
// CHECK: %[[AB:.+]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f64>) -> tensor<10xf64>
// CHECK: %[[SCALED:.+]] = stablehlo.multiply %[[CC]], %[[AB]]
// CHECK: %[[BB:.+]] = stablehlo.broadcast_in_dim %arg6, dims = [] : (tensor<f64>) -> tensor<10xf64>
// CHECK: %[[SC:.+]] = stablehlo.multiply %arg4, %[[BB]]
// CHECK: %[[Y:.+]] = stablehlo.add %[[SCALED]], %[[SC]]
// CHECK-NOT: sparse_tensor
// CHECK: return %[[Y]]
func.func @runtime_scalars(%rowptr: tensor<11xi64>, %colind: tensor<20xi64>, %nzval: tensor<20xf64>, %x: tensor<8xf64>, %C: tensor<10xf64>, %alpha: tensor<f64>, %beta: tensor<f64>) -> tensor<10xf64> {
  %A = sparse_tensor.assemble (%rowptr, %colind), %nzval : (tensor<11xi64>, tensor<20xi64>), tensor<20xf64> to tensor<10x8xf64, #csr>
  %y = enzymexla.sparse.spmm %alpha, %A, %x, %beta, %C : (tensor<f64>, tensor<10x8xf64, #csr>, tensor<8xf64>, tensor<f64>, tensor<10xf64>) -> tensor<10xf64>
  return %y : tensor<10xf64>
}

// Constant alpha with runtime beta: alpha still folds into the call, beta
// accumulates explicitly.
// CHECK-LABEL: func.func @runtime_beta
// CHECK-NOT: sparse_tensor
// CHECK: %[[CC:.+]] = stablehlo.custom_call @reactant_csr_matmul(%arg0, %arg1, %arg2, %arg3)
// CHECK-SAME: backend_config = {alpha = 2.000000e+00 : f64,
// CHECK: %[[BB:.+]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f64>) -> tensor<10xf64>
// CHECK: %[[SC:.+]] = stablehlo.multiply %arg4, %[[BB]]
// CHECK: %[[Y:.+]] = stablehlo.add %[[CC]], %[[SC]]
// CHECK-NOT: sparse_tensor
// CHECK: return %[[Y]]
func.func @runtime_beta(%rowptr: tensor<11xi64>, %colind: tensor<20xi64>, %nzval: tensor<20xf64>, %x: tensor<8xf64>, %C: tensor<10xf64>, %beta: tensor<f64>) -> tensor<10xf64> {
  %alpha = stablehlo.constant dense<2.0> : tensor<f64>
  %A = sparse_tensor.assemble (%rowptr, %colind), %nzval : (tensor<11xi64>, tensor<20xi64>), tensor<20xf64> to tensor<10x8xf64, #csr>
  %y = enzymexla.sparse.spmm %alpha, %A, %x, %beta, %C : (tensor<f64>, tensor<10x8xf64, #csr>, tensor<8xf64>, tensor<f64>, tensor<10xf64>) -> tensor<10xf64>
  return %y : tensor<10xf64>
}
