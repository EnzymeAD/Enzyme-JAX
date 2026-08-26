// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a: tensor<2x3x50xf32>) ->tensor<4x1x25x15x2x3xf32>{
    %bc = stablehlo.broadcast_in_dim %a, dims = [4, 5, 2] : (tensor<2x3x50xf32>)->tensor<7x11x50x15x2x3xf32>
    %add = stablehlo.slice %bc [0:4, 7:9:2, 0:50:2, 0:15, 0:2, 0:3] : (tensor<7x11x50x15x2x3xf32>) ->tensor<4x1x25x15x2x3xf32>
    return %add : tensor<4x1x25x15x2x3xf32>
  }

  func.func @main2(%1909 : tensor<1x8x3x1024x1xf32>) -> tensor<1x8x3x1024x1024xf32> {
    %2388 = stablehlo.broadcast_in_dim %1909, dims = [0, 1, 2, 3, 4] : (tensor<1x8x3x1024x1xf32>) -> tensor<1x8x3x1024x2048xf32> 
    %2389 = stablehlo.slice %2388 [0:1, 0:8, 0:3, 0:1024, 1024:2048] : (tensor<1x8x3x1024x2048xf32>) -> tensor<1x8x3x1024x1024xf32>
    return %2389 : tensor<1x8x3x1024x1024xf32>
  }
  
  func.func @main3(%20: tensor<f64>) -> (tensor<3056x6128xf64>, tensor<3055x6128xf64>) {
    %28 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f64>) -> tensor<3056x6128xf64>
    %31 = stablehlo.slice %28 [1:3056, 0:6128] : (tensor<3056x6128xf64>) -> tensor<3055x6128xf64> 
    return %28, %31 : tensor<3056x6128xf64>, tensor<3055x6128xf64>
  }

  func.func @main4(%12: tensor<4xf64>) ->  (tensor<1515x4xf64>, tensor<1520x4xf64>) {
    %43 = stablehlo.broadcast_in_dim %12, dims = [1] : (tensor<4xf64>) -> tensor<1520x4xf64>
    %284 = stablehlo.slice %43 [3:1518, 0:4] : (tensor<1520x4xf64>) -> tensor<1515x4xf64>
    return %284, %43 : tensor<1515x4xf64>, tensor<1520x4xf64>
  }

}

// CHECK:  func.func @main(%arg0: tensor<2x3x50xf32>) -> tensor<4x1x25x15x2x3xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:2, 0:3, 0:50:2] : (tensor<2x3x50xf32>) -> tensor<2x3x25xf32>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [4, 5, 2] : (tensor<2x3x25xf32>) -> tensor<4x1x25x15x2x3xf32>
// CHECK-NEXT:    return %1 : tensor<4x1x25x15x2x3xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @main2(%arg0: tensor<1x8x3x1024x1xf32>) -> tensor<1x8x3x1024x1024xf32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3, 4] : (tensor<1x8x3x1024x1xf32>) -> tensor<1x8x3x1024x1024xf32>
// CHECK-NEXT:    return %0 : tensor<1x8x3x1024x1024xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @main3(%arg0: tensor<f64>) -> (tensor<3056x6128xf64>, tensor<3055x6128xf64>) {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<3056x6128xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<3055x6128xf64>
// CHECK-NEXT:    return %0, %1 : tensor<3056x6128xf64>, tensor<3055x6128xf64>
// CHECK-NEXT:  }

// CHECK:  func.func @main4(%arg0: tensor<4xf64>) -> (tensor<1515x4xf64>, tensor<1520x4xf64>) {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xf64>) -> tensor<1520x4xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xf64>) -> tensor<1515x4xf64>
// CHECK-NEXT:    return %1, %0 : tensor<1515x4xf64>, tensor<1520x4xf64>
// CHECK-NEXT:  }

