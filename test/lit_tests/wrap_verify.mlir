// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

// Amounts within the operand's extent (no error expected).
func.func @within(%x: tensor<4xf64>) -> tensor<10xf64> {
  %w = "enzymexla.wrap"(%x) <{dimension = 0 : i64, lhs = 4 : i64, rhs = 2 : i64}> : (tensor<4xf64>) -> tensor<10xf64>
  return %w : tensor<10xf64>
}

// -----

func.func @beyond(%x: tensor<2xf64>) -> tensor<10xf64> {
  // expected-error @+1 {{amounts 3 and 5 must not exceed the operand's extent 2 along dimension 0}}
  %w = "enzymexla.wrap"(%x) <{dimension = 0 : i64, lhs = 3 : i64, rhs = 5 : i64}> : (tensor<2xf64>) -> tensor<10xf64>
  return %w : tensor<10xf64>
}
