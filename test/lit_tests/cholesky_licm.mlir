// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="licm_passes=true" --allow-unregistered-dialect | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=cholesky_licm(0)" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%M: tensor<4x4xf64>, %init: tensor<4x4xf64>, %n: tensor<i64>) -> tensor<4x4xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c0, %acc = %init) : tensor<i64>, tensor<4x4xf64>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %f = stablehlo.cholesky %M, lower = true : tensor<4x4xf64>
      %g = stablehlo.cholesky %acc, lower = true : tensor<4x4xf64>
      %sum = stablehlo.add %f, %g : tensor<4x4xf64>
      %i2 = stablehlo.add %iterArg, %c1 : tensor<i64>
      stablehlo.return %i2, %sum : tensor<i64>, tensor<4x4xf64>
    }
    return %0#1 : tensor<4x4xf64>
  }
}

// CHECK:  func.func @main(%[[M:.+]]: tensor<4x4xf64>, %[[INIT:.+]]: tensor<4x4xf64>, %[[N:.+]]: tensor<i64>) -> tensor<4x4xf64> {
// CHECK-NEXT:    %[[C0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[C1:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[FHOIST:.+]] = stablehlo.cholesky %[[M]], lower = true : tensor<4x4xf64>
// CHECK-NEXT:    %[[W:.+]]:2 = stablehlo.while(%iterArg = %[[C0]], %iterArg_1 = %[[INIT]]) : tensor<i64>, tensor<4x4xf64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %[[CMP:.+]] = stablehlo.compare  LT, %iterArg, %[[N]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %[[CMP]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[GVARY:.+]] = stablehlo.cholesky %iterArg_1, lower = true : tensor<4x4xf64>
// CHECK-NEXT:      %[[SUM:.+]] = stablehlo.add %[[FHOIST]], %[[GVARY]] : tensor<4x4xf64>
// CHECK-NEXT:      %[[I2:.+]] = stablehlo.add %iterArg, %[[C1]] : tensor<i64>
// CHECK-NEXT:      stablehlo.return %[[I2]], %[[SUM]] : tensor<i64>, tensor<4x4xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[W]]#1 : tensor<4x4xf64>
// CHECK-NEXT:  }
