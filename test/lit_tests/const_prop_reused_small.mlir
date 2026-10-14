// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=div_const_prop(4);rem_const_prop(4);neg_const_prop(4)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Index arithmetic on a constant that has other users: with the parameter, a
// result of at most 4 elements folds even though the constant stays.
func.func @small(%x: tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
  %c = stablehlo.constant dense<[0, 5, 7, 12]> : tensor<4xi64>
  %d = stablehlo.constant dense<6> : tensor<4xi64>
  %q = stablehlo.divide %c, %d : tensor<4xi64>
  %r = stablehlo.remainder %c, %d : tensor<4xi64>
  %u = stablehlo.add %c, %x : tensor<4xi64>
  return %q, %r, %u : tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
}

// CHECK-LABEL: func.func @small(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.constant dense<[0, 5, 7, 12]> : tensor<4xi64>
// CHECK-NEXT: %[[Q:[^ ,:)]+]] = stablehlo.constant dense<[0, 0, 1, 2]> : tensor<4xi64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.constant dense<[0, 5, 1, 0]> : tensor<4xi64>
// CHECK-NEXT: %[[U:[^ ,:)]+]] = stablehlo.add %[[C]], %[[X]] : tensor<4xi64>
// CHECK-NEXT: return %[[Q]], %[[R]], %[[U]] : tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
// CHECK-NEXT: }

// More than 4 elements: the constant stays, so these do not fold.
func.func @large(%x: tensor<8xi64>) -> (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>) {
  %c = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
  %d = stablehlo.constant dense<3> : tensor<8xi64>
  %q = stablehlo.divide %c, %d : tensor<8xi64>
  %n = stablehlo.negate %c : tensor<8xi64>
  %u = stablehlo.add %c, %x : tensor<8xi64>
  return %q, %n, %u : tensor<8xi64>, tensor<8xi64>, tensor<8xi64>
}

// CHECK-LABEL: func.func @large(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<8xi64>) -> (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>) {
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
// CHECK-NEXT: %[[D:[^ ,:)]+]] = stablehlo.constant dense<3> : tensor<8xi64>
// CHECK-NEXT: %[[Q:[^ ,:)]+]] = stablehlo.divide %[[C]], %[[D]] : tensor<8xi64>
// CHECK-NEXT: %[[N:[^ ,:)]+]] = stablehlo.negate %[[C]] : tensor<8xi64>
// CHECK-NEXT: %[[U:[^ ,:)]+]] = stablehlo.add %[[C]], %[[X]] : tensor<8xi64>
// CHECK-NEXT: return %[[Q]], %[[N]], %[[U]] : tensor<8xi64>, tensor<8xi64>, tensor<8xi64>
// CHECK-NEXT: }

// The unary op with the parameter: folds beside its constant.
func.func @unary(%x: tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
  %c = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi64>
  %n = stablehlo.negate %c : tensor<3xi64>
  %u = stablehlo.add %c, %x : tensor<3xi64>
  return %n, %u : tensor<3xi64>, tensor<3xi64>
}

// CHECK-LABEL: func.func @unary(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
// CHECK-NEXT: %[[C:[^ ,:)]+]] = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi64>
// CHECK-NEXT: %[[N:[^ ,:)]+]] = stablehlo.constant dense<[-1, -2, -3]> : tensor<3xi64>
// CHECK-NEXT: %[[U:[^ ,:)]+]] = stablehlo.add %[[C]], %[[X]] : tensor<3xi64>
// CHECK-NEXT: return %[[N]], %[[U]] : tensor<3xi64>, tensor<3xi64>
// CHECK-NEXT: }
