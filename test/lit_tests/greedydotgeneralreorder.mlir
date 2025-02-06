// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// module {
//   func.func @main(%arg0: tensor<1x10xf64>, %arg1: tensor<10x1xf64>, %arg2: tensor<10x10xf64>) -> tensor<10x10xf64> {
//     %0 = stablehlo.reshape %arg0 : (tensor<1x10xf64>) -> tensor<10x1xf64>
//     %1 = stablehlo.reshape %arg1 : (tensor<10x1xf64>) -> tensor<1x10xf64>
//     %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] : (tensor<10x1xf64>, tensor<1x10xf64>) -> tensor<10x10xf64>
//     %3 = stablehlo.dot_general %arg2, %2, contracting_dims = [1] x [1] : (tensor<10x10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
//     return %3 : tensor<10x10xf64>
//   }
// }

module {
  func.func @inefficient_dot_chain(
    %arg0: tensor<10x50xf32>,
    %arg1: tensor<50x100xf32>,
    %arg2: tensor<100x20xf32>,
    %arg3: tensor<20x30xf32>
  ) -> tensor<10x30xf32> {
    // Original order: ((A × B) × C) × D
    // Shapes: (10×50) × (50×100) × (100×20) × (20×30)
    // FLOPs: (10×50×100) + (10×100×20) + (10×20×30)
    //      = 50,000 + 20,000 + 6,000 = 76,000 FLOPs
    
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<10x50xf32>, tensor<50x100xf32>) -> tensor<10x100xf32>
    %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] : (tensor<10x100xf32>, tensor<100x20xf32>) -> tensor<10x20xf32>
    %2 = stablehlo.dot_general %1, %arg3, contracting_dims = [1] x [0] : (tensor<10x20xf32>, tensor<20x30xf32>) -> tensor<10x30xf32>
    return %2 : tensor<10x30xf32>
  }
}
