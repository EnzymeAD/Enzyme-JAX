// RUN: enzymexlamlir-opt --lower-sparse-csr --verify-diagnostics %s

#csr = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 64, crdWidth = 64 }>

// A sparse tensor that escapes as a function result cannot be lowered to a
// custom call and must be reported instead of being handed to XLA.
func.func @escaping_sparse(%rowptr: tensor<11xi64>, %colind: tensor<20xi64>, %nzval: tensor<20xf64>) -> tensor<10x8xf64, #csr> {
  // expected-error @below {{unsupported use of sparse tensors}}
  %A = sparse_tensor.assemble (%rowptr, %colind), %nzval : (tensor<11xi64>, tensor<20xi64>), tensor<20xf64> to tensor<10x8xf64, #csr>
  return %A : tensor<10x8xf64, #csr>
}
