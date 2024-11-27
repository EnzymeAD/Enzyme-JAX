// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @f(%arg3:tensor<10xi32>, %arg4:tensor<10xi32>) -> tensor<10xi32> {
  %c0 = stablehlo.constant dense<true> : tensor<10xi1>
  %7 = stablehlo.select %c0, %arg3, %arg4 : tensor<10xi1>, tensor<10xi32>
  return %7 : tensor<10xi32>
}

func.func @g() -> tensor<3xi32> {
  %c0 = stablehlo.constant dense<[true, false, false]> : tensor<3xi1>
    %c1 = stablehlo.constant dense<[42, 61, 76]> : tensor<3xi32>
  %c2 = stablehlo.constant dense<[5, 36, 23]> : tensor<3xi32>

  %7 = stablehlo.select %c0, %c1, %c2 : tensor<3xi1>, tensor<3xi32>
  return %7 : tensor<3xi32>
}

CHECK:  func.func @f(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32>
CHECK-NEXT   return %arg0 : tensor<10xi32>

CHECK: func.func @g() -> tensor<3xi32> 
CHECK-NEXT: %c = stablehlo.constant dense<[42, 36, 23]> : tensor<3xi32>
CHECK-NEXT    return %c : tensor<3xi32>

