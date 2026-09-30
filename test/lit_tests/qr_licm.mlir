// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="licm_passes=true" --allow-unregistered-dialect | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=qr_licm(0)" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefix=TD

module {
  func.func @main(%X: tensor<8x2xf64>, %init: tensor<8x2xf64>, %n: tensor<i64>) -> tensor<8x2xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c0, %acc = %init) : tensor<i64>, tensor<8x2xf64>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %q, %r, %info = enzymexla.linalg.qr %X : (tensor<8x2xf64>) -> (tensor<8x2xf64>, tensor<8x2xf64>, tensor<i32>)
      %q2, %r2, %info2 = enzymexla.linalg.qr %acc : (tensor<8x2xf64>) -> (tensor<8x2xf64>, tensor<8x2xf64>, tensor<i32>)
      %sum = stablehlo.add %q, %q2 : tensor<8x2xf64>
      %i2 = stablehlo.add %iterArg, %c1 : tensor<i64>
      stablehlo.return %i2, %sum : tensor<i64>, tensor<8x2xf64>
    }
    return %0#1 : tensor<8x2xf64>
  }
}

// CHECK:  func.func @main(%[[X:.+]]: tensor<8x2xf64>, %[[INIT:.+]]: tensor<8x2xf64>, %[[N:.+]]: tensor<i64>) -> tensor<8x2xf64> {
// CHECK:    %[[QH:.+]], %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[X]] : (tensor<8x2xf64>) -> (tensor<8x2xf64>, tensor<8x2xf64>, tensor<i32>)
// CHECK-NEXT:    %{{.+}}:2 = stablehlo.while(%iterArg = %{{.+}}, %[[ACC:.+]] = %[[INIT]]) : tensor<i64>, tensor<8x2xf64>
// CHECK:    } do {
// CHECK-NEXT:      %[[QV:.+]], %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[ACC]] : (tensor<8x2xf64>) -> (tensor<8x2xf64>, tensor<8x2xf64>, tensor<i32>)
// CHECK-NEXT:      %{{.+}} = stablehlo.add %[[QH]], %[[QV]] : tensor<8x2xf64>

// TD:  func.func @main(%[[X:.+]]: tensor<8x2xf64>, %[[INIT:.+]]: tensor<8x2xf64>, %[[N:.+]]: tensor<i64>) -> tensor<8x2xf64> {
// TD:    %[[QH:.+]], %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[X]] : (tensor<8x2xf64>) -> (tensor<8x2xf64>, tensor<8x2xf64>, tensor<i32>)
// TD-NEXT:    %{{.+}}:2 = stablehlo.while(%iterArg = %{{.+}}, %[[ACC:.+]] = %[[INIT]]) : tensor<i64>, tensor<8x2xf64>
// TD:    } do {
// TD-NEXT:      %[[QV:.+]], %{{.+}}, %{{.+}} = enzymexla.linalg.qr %[[ACC]] : (tensor<8x2xf64>) -> (tensor<8x2xf64>, tensor<8x2xf64>, tensor<i32>)
// TD-NEXT:      %{{.+}} = stablehlo.add %[[QH]], %[[QV]] : tensor<8x2xf64>
