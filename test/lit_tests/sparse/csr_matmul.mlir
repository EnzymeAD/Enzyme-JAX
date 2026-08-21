// RUN: enzymexlamlir-opt --lower-enzymexla-sparse %s | FileCheck %s

#csr = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 64, crdWidth = 64 }>
#csr32 = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 32, crdWidth = 32 }>

// CHECK-LABEL: func.func @spmv
// CHECK-NOT: sparse_tensor
// CHECK: %[[Y:.+]] = stablehlo.custom_call @reactant_csr_matmul(%arg0, %arg1, %arg2, %arg3)
// CHECK-SAME: api_version = 4 : i32
// CHECK-SAME: backend_config = {alpha = 1.000000e+00 : f64, index_base = 0 : i64, m = 10 : i64, n = 8 : i64, transpose = 0 : i64}
// CHECK-SAME: (tensor<11xi64>, tensor<20xi64>, tensor<20xf64>, tensor<8xf64>) -> tensor<10xf64>
// CHECK-NOT: sparse_tensor
// CHECK: return %[[Y]]
func.func @spmv(%rowptr: tensor<11xi64>, %colind: tensor<20xi64>, %nzval: tensor<20xf64>, %x: tensor<8xf64>) -> tensor<10xf64> {
  %A = sparse_tensor.assemble (%rowptr, %colind), %nzval : (tensor<11xi64>, tensor<20xi64>), tensor<20xf64> to tensor<10x8xf64, #csr>
  %y = stablehlo.dot_general %A, %x, contracting_dims = [1] x [0] : (tensor<10x8xf64, #csr>, tensor<8xf64>) -> tensor<10xf64>
  return %y : tensor<10xf64>
}

// CHECK-LABEL: func.func @spmm
// CHECK-NOT: sparse_tensor
// CHECK: %[[Y:.+]] = stablehlo.custom_call @reactant_csr_matmul(%arg0, %arg1, %arg2, %arg3)
// CHECK-SAME: backend_config = {alpha = 1.000000e+00 : f64, index_base = 0 : i64, m = 10 : i64, n = 8 : i64, transpose = 0 : i64}
// CHECK-SAME: operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>]
// CHECK-SAME: result_layouts = [dense<[0, 1]> : tensor<2xindex>]
// CHECK-SAME: (tensor<11xi64>, tensor<20xi64>, tensor<20xf64>, tensor<8x3xf64>) -> tensor<10x3xf64>
// CHECK-NOT: sparse_tensor
// CHECK: return %[[Y]]
func.func @spmm(%rowptr: tensor<11xi64>, %colind: tensor<20xi64>, %nzval: tensor<20xf64>, %B: tensor<8x3xf64>) -> tensor<10x3xf64> {
  %A = sparse_tensor.assemble (%rowptr, %colind), %nzval : (tensor<11xi64>, tensor<20xi64>), tensor<20xf64> to tensor<10x8xf64, #csr>
  %y = stablehlo.dot_general %A, %B, contracting_dims = [1] x [0] : (tensor<10x8xf64, #csr>, tensor<8x3xf64>) -> tensor<10x3xf64>
  return %y : tensor<10x3xf64>
}

// CHECK-LABEL: func.func @spmv_i32
// CHECK-NOT: sparse_tensor
// CHECK: %[[Y:.+]] = stablehlo.custom_call @reactant_csr_matmul(%arg0, %arg1, %arg2, %arg3)
// CHECK-SAME: (tensor<11xi32>, tensor<20xi32>, tensor<20xf32>, tensor<8xf32>) -> tensor<10xf32>
// CHECK-NOT: sparse_tensor
// CHECK: return %[[Y]]
func.func @spmv_i32(%rowptr: tensor<11xi32>, %colind: tensor<20xi32>, %nzval: tensor<20xf32>, %x: tensor<8xf32>) -> tensor<10xf32> {
  %A = sparse_tensor.assemble (%rowptr, %colind), %nzval : (tensor<11xi32>, tensor<20xi32>), tensor<20xf32> to tensor<10x8xf32, #csr32>
  %y = stablehlo.dot_general %A, %x, contracting_dims = [1] x [0] : (tensor<10x8xf32, #csr32>, tensor<8xf32>) -> tensor<10xf32>
  return %y : tensor<10xf32>
}

// A single assemble feeding two dot_generals: both get lowered and the
// assemble is erased.
// CHECK-LABEL: func.func @shared_assemble
// CHECK-NOT: sparse_tensor
// CHECK: stablehlo.custom_call @reactant_csr_matmul
// CHECK: stablehlo.custom_call @reactant_csr_matmul
// CHECK-NOT: sparse_tensor
func.func @shared_assemble(%rowptr: tensor<11xi64>, %colind: tensor<20xi64>, %nzval: tensor<20xf64>, %x: tensor<8xf64>, %B: tensor<8x3xf64>) -> (tensor<10xf64>, tensor<10x3xf64>) {
  %A = sparse_tensor.assemble (%rowptr, %colind), %nzval : (tensor<11xi64>, tensor<20xi64>), tensor<20xf64> to tensor<10x8xf64, #csr>
  %y = stablehlo.dot_general %A, %x, contracting_dims = [1] x [0] : (tensor<10x8xf64, #csr>, tensor<8xf64>) -> tensor<10xf64>
  %Z = stablehlo.dot_general %A, %B, contracting_dims = [1] x [0] : (tensor<10x8xf64, #csr>, tensor<8x3xf64>) -> tensor<10x3xf64>
  return %y, %Z : tensor<10xf64>, tensor<10x3xf64>
}
