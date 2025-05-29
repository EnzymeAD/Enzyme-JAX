// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --cse --enzyme-hlo-opt | FileCheck %s

func.func @square(%a : tensor<2xf32>) -> tensor<2xf32> {
  %c = stablehlo.multiply %a, %a : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %c : tensor<2xf32>
}

func.func @dsquare(%x: tensor<2xf32>, %dr: tensor<2xf32>) -> tensor<2xf32> {
  %r = enzyme.autodiff @square(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>], strong_zero=true } : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK:  func.func private @diffesquare(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %0 = stablehlo.compare  EQ, %arg1, %cst : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:    %1 = stablehlo.multiply %arg1, %arg0 : tensor<2xf32>
// CHECK-NEXT:    %2 = stablehlo.select %0, %cst, %1 : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT:    %3 = arith.addf %2, %2 : tensor<2xf32>
// CHECK-NEXT:    return %3 : tensor<2xf32>
// CHECK-NEXT:  }
