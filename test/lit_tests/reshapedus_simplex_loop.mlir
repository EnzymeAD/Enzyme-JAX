// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=merge_consecutive_reshapes<16>;reshape_dus},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s

// Hang regression: reshape_dus must not mutate the IR on a failed match.

module {
  func.func @simplex_reshape_dus_loop(%init: tensor<1x9xf64>, %col: tensor<3x1xf64>) -> tensor<1x9xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %n = stablehlo.constant dense<5> : tensor<i64>
    %r:2 = stablehlo.while(%it = %c0, %buf = %init) : tensor<i64>, tensor<1x9xf64>
      cond {
        %lt = stablehlo.compare LT, %it, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %lt : tensor<i1>
      } do {
        %a = stablehlo.reshape %buf : (tensor<1x9xf64>) -> tensor<3x3xf64>
        %b = stablehlo.reshape %a : (tensor<3x3xf64>) -> tensor<1x9xf64>
        %a2 = stablehlo.reshape %b : (tensor<1x9xf64>) -> tensor<3x3xf64>
        %dus = stablehlo.dynamic_update_slice %a2, %col, %c0, %c1 : (tensor<3x3xf64>, tensor<3x1xf64>, tensor<i64>, tensor<i64>) -> tensor<3x3xf64>
        %next = stablehlo.reshape %dus : (tensor<3x3xf64>) -> tensor<1x9xf64>
        %it2 = stablehlo.add %it, %c1 : tensor<i64>
        stablehlo.return %it2, %next : tensor<i64>, tensor<1x9xf64>
      }
    return %r#1 : tensor<1x9xf64>
  }
}

// CHECK:  func.func @simplex_reshape_dus_loop(%arg0: tensor<1x9xf64>, %arg1: tensor<3x1xf64>) -> tensor<1x9xf64> {
// CHECK-NEXT:    %[[C0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[C1:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[N:.+]] = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:    %[[R:.+]]:2 = stablehlo.while(%[[IT:.+]] = %[[C0]], %[[BUF:.+]] = %arg0) : tensor<i64>, tensor<1x9xf64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %[[LT:.+]] = stablehlo.compare LT, %[[IT]], %[[N]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %[[LT]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[A:.+]] = stablehlo.reshape %[[BUF]] : (tensor<1x9xf64>) -> tensor<3x3xf64>
// CHECK-NEXT:      %[[DUS:.+]] = stablehlo.dynamic_update_slice %[[A]], %arg1, %[[C0]], %[[C1]] : (tensor<3x3xf64>, tensor<3x1xf64>, tensor<i64>, tensor<i64>) -> tensor<3x3xf64>
// CHECK-NEXT:      %[[NEXT:.+]] = stablehlo.reshape %[[DUS]] : (tensor<3x3xf64>) -> tensor<1x9xf64>
// CHECK-NEXT:      %[[IT2:.+]] = stablehlo.add %[[IT]], %[[C1]] : tensor<i64>
// CHECK-NEXT:      stablehlo.return %[[IT2]], %[[NEXT]] : tensor<i64>, tensor<1x9xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[R]]#1 : tensor<1x9xf64>
// CHECK-NEXT:  }
