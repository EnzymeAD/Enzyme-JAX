// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<2xf32>, %b : tensor<1xf32>, %x : tensor<2xf32>, %y : tensor<1xf32>) -> tensor<3xf32> {
    %u = stablehlo.add %a, %x : tensor<2xf32>
    %v = stablehlo.add %b, %y : tensor<1xf32>
    %concat = stablehlo.concatenate %u, %v, dim=0 : (tensor<2xf32>, tensor<1xf32>) -> tensor<3xf32>
    return %concat : tensor<3xf32>
  }
  func.func @main2(%a : tensor<2xf32>, %b : tensor<1xf32>, %x : tensor<2xf32>, %y : tensor<1xf32>) -> tensor<3xf64> {
    %u = stablehlo.multiply %a, %x : tensor<2xf32>
    %uc = stablehlo.convert %u : (tensor<2xf32>) -> tensor<2xf64>
    %v = stablehlo.multiply %b, %y : tensor<1xf32>
    %vc = stablehlo.convert %v : (tensor<1xf32>) -> tensor<1xf64>
    %concat = stablehlo.concatenate %uc, %vc, dim=0 : (tensor<2xf64>, tensor<1xf64>) -> tensor<3xf64>
    return %concat : tensor<3xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<1xf32>, %arg2: tensor<2xf32>, %arg3: tensor<1xf32>) -> tensor<3xf32> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<2xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK-NEXT:    %1 = stablehlo.concatenate %arg2, %arg3, dim = 0 : (tensor<2xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK-NEXT:    %2 = stablehlo.add %0, %1 : tensor<3xf32>
// CHECK-NEXT:    return %2 : tensor<3xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @main2(%arg0: tensor<2xf32>, %arg1: tensor<1xf32>, %arg2: tensor<2xf32>, %arg3: tensor<1xf32>) -> tensor<3xf64> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<2xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK-NEXT:    %1 = stablehlo.concatenate %arg2, %arg3, dim = 0 : (tensor<2xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK-NEXT:    %2 = stablehlo.multiply %0, %1 : tensor<3xf32>
// CHECK-NEXT:    %3 = stablehlo.convert %2 : (tensor<3xf32>) -> tensor<3xf64>
// CHECK-NEXT:    return %3 : tensor<3xf64>
// CHECK-NEXT:  }
