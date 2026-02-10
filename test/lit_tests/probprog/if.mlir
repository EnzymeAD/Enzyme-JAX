// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s

module {
  // CHECK:       func.func @test_simple_if(%[[ARG0:.+]]: tensor<i1>, %[[ARG1:.+]]: tensor<f64>, %[[ARG2:.+]]: tensor<f64>) -> tensor<f64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = "stablehlo.if"(%[[ARG0]]) ({
  // CHECK-NEXT:      stablehlo.return %[[ARG1]] : tensor<f64>
  // CHECK-NEXT:    }, {
  // CHECK-NEXT:      stablehlo.return %[[ARG2]] : tensor<f64>
  // CHECK-NEXT:    }) : (tensor<i1>) -> tensor<f64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<f64>
  // CHECK-NEXT:  }
  func.func @test_simple_if(%pred: tensor<i1>, %a: tensor<f64>, %b: tensor<f64>) -> tensor<f64> {
    %result = enzyme.if (%pred) ({
      enzyme.yield %a : tensor<f64>
    }, {
      enzyme.yield %b : tensor<f64>
    }) : (tensor<i1>) -> tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK:       func.func @test_if_with_computation(%[[ARG0:.+]]: tensor<i1>, %[[ARG1:.+]]: tensor<f64>, %[[ARG2:.+]]: tensor<f64>) -> tensor<f64> {
  // CHECK-NEXT:    %[[RESULT:.+]] = "stablehlo.if"(%[[ARG0]]) ({
  // CHECK-NEXT:      %[[ADD:.+]] = stablehlo.add %[[ARG1]], %[[ARG2]] : tensor<f64>
  // CHECK-NEXT:      stablehlo.return %[[ADD]] : tensor<f64>
  // CHECK-NEXT:    }, {
  // CHECK-NEXT:      %[[MUL:.+]] = stablehlo.multiply %[[ARG1]], %[[ARG2]] : tensor<f64>
  // CHECK-NEXT:      stablehlo.return %[[MUL]] : tensor<f64>
  // CHECK-NEXT:    }) : (tensor<i1>) -> tensor<f64>
  // CHECK-NEXT:    return %[[RESULT]] : tensor<f64>
  // CHECK-NEXT:  }
  func.func @test_if_with_computation(%pred: tensor<i1>, %x: tensor<f64>, %y: tensor<f64>) -> tensor<f64> {
    %result = enzyme.if (%pred) ({
      %add = stablehlo.add %x, %y : tensor<f64>
      enzyme.yield %add : tensor<f64>
    }, {
      %mul = stablehlo.multiply %x, %y : tensor<f64>
      enzyme.yield %mul : tensor<f64>
    }) : (tensor<i1>) -> tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK:       func.func @test_if_multiple_results(%[[ARG0:.+]]: tensor<i1>, %[[ARG1:.+]]: tensor<f64>, %[[ARG2:.+]]: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
  // CHECK-NEXT:    %[[RESULT:.+]]:2 = "stablehlo.if"(%[[ARG0]]) ({
  // CHECK-NEXT:      stablehlo.return %[[ARG1]], %[[ARG2]] : tensor<f64>, tensor<f64>
  // CHECK-NEXT:    }, {
  // CHECK-NEXT:      stablehlo.return %[[ARG2]], %[[ARG1]] : tensor<f64>, tensor<f64>
  // CHECK-NEXT:    }) : (tensor<i1>) -> (tensor<f64>, tensor<f64>)
  // CHECK-NEXT:    return %[[RESULT]]#0, %[[RESULT]]#1 : tensor<f64>, tensor<f64>
  // CHECK-NEXT:  }
  func.func @test_if_multiple_results(%pred: tensor<i1>, %a: tensor<f64>, %b: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %r0, %r1 = enzyme.if (%pred) ({
      enzyme.yield %a, %b : tensor<f64>, tensor<f64>
    }, {
      enzyme.yield %b, %a : tensor<f64>, tensor<f64>
    }) : (tensor<i1>) -> (tensor<f64>, tensor<f64>)
    return %r0, %r1 : tensor<f64>, tensor<f64>
  }

  // CHECK:       func.func @test_nested_if(%[[ARG0:.+]]: tensor<i1>, %[[ARG1:.+]]: tensor<i1>, %[[ARG2:.+]]: tensor<f64>) -> tensor<f64> {
  // CHECK-NEXT:    %[[CST:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  // CHECK-NEXT:    %[[CST_0:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<f64>
  // CHECK-NEXT:    %[[OUTER:.+]] = "stablehlo.if"(%[[ARG0]]) ({
  // CHECK-NEXT:      %[[INNER:.+]] = "stablehlo.if"(%[[ARG1]]) ({
  // CHECK-NEXT:        %[[ADD:.+]] = stablehlo.add %[[ARG2]], %[[CST]] : tensor<f64>
  // CHECK-NEXT:        stablehlo.return %[[ADD]] : tensor<f64>
  // CHECK-NEXT:      }, {
  // CHECK-NEXT:        %[[SUB:.+]] = stablehlo.subtract %[[ARG2]], %[[CST]] : tensor<f64>
  // CHECK-NEXT:        stablehlo.return %[[SUB]] : tensor<f64>
  // CHECK-NEXT:      }) : (tensor<i1>) -> tensor<f64>
  // CHECK-NEXT:      stablehlo.return %[[INNER]] : tensor<f64>
  // CHECK-NEXT:    }, {
  // CHECK-NEXT:      %[[MUL:.+]] = stablehlo.multiply %[[ARG2]], %[[CST_0]] : tensor<f64>
  // CHECK-NEXT:      stablehlo.return %[[MUL]] : tensor<f64>
  // CHECK-NEXT:    }) : (tensor<i1>) -> tensor<f64>
  // CHECK-NEXT:    return %[[OUTER]] : tensor<f64>
  // CHECK-NEXT:  }
  func.func @test_nested_if(%p1: tensor<i1>, %p2: tensor<i1>, %x: tensor<f64>) -> tensor<f64> {
    %c1 = stablehlo.constant dense<1.0> : tensor<f64>
    %c2 = stablehlo.constant dense<2.0> : tensor<f64>

    %result = enzyme.if (%p1) ({
      %inner = enzyme.if (%p2) ({
        %add = stablehlo.add %x, %c1 : tensor<f64>
        enzyme.yield %add : tensor<f64>
      }, {
        %sub = stablehlo.subtract %x, %c1 : tensor<f64>
        enzyme.yield %sub : tensor<f64>
      }) : (tensor<i1>) -> tensor<f64>
      enzyme.yield %inner : tensor<f64>
    }, {
      %mul = stablehlo.multiply %x, %c2 : tensor<f64>
      enzyme.yield %mul : tensor<f64>
    }) : (tensor<i1>) -> tensor<f64>

    return %result : tensor<f64>
  }
}
