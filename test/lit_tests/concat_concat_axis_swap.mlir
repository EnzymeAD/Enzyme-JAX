// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=548863})" %s | FileCheck %s --check-prefix=ROTATE

module @"reactant_loop!" {
  func.func @main(%arg21: tensor<20x24x96xf64>, %6537 : tensor<4x7x80xf64>, %cst_286 : tensor<f64>) -> (tensor<4x8x80xf64>) {
      %11824 = stablehlo.slice %arg21 [8:12, 6:8, 8:87] : (tensor<20x24x96xf64>) -> tensor<4x2x79xf64>

      %11825 = stablehlo.slice %6537 [0:4, 0:5, 0:79] : (tensor<4x7x80xf64>) -> tensor<4x5x79xf64>

      %11826 = stablehlo.pad %11825, %cst_286, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x5x79xf64>, tensor<f64>) -> tensor<4x6x79xf64>
      %11827 = stablehlo.concatenate %11824, %11826, dim = 1 : (tensor<4x2x79xf64>, tensor<4x6x79xf64>) -> tensor<4x8x79xf64>


      %11828 = stablehlo.slice %arg21 [8:12, 6:8, 87:88] : (tensor<20x24x96xf64>) -> tensor<4x2x1xf64>

      %11829 = stablehlo.slice %6537 [0:4, 0:5, 79:80] : (tensor<4x7x80xf64>) -> tensor<4x5x1xf64>

      %11830 = stablehlo.pad %11829, %cst_286, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x5x1xf64>, tensor<f64>) -> tensor<4x6x1xf64>

      %11831 = stablehlo.concatenate %11828, %11830, dim = 1 : (tensor<4x2x1xf64>, tensor<4x6x1xf64>) -> tensor<4x8x1xf64>

      %11832 = stablehlo.concatenate %11831, %11827, dim = 2 : (tensor<4x8x1xf64>, tensor<4x8x79xf64>) -> tensor<4x8x80xf64>

      stablehlo.return %11832 : tensor<4x8x80xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<20x24x96xf64>, %arg1: tensor<4x7x80xf64>, %arg2: tensor<f64>) -> tensor<4x8x80xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 6:8, 8:87] : (tensor<20x24x96xf64>) -> tensor<4x2x79xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:4, 0:5, 0:79] : (tensor<4x7x80xf64>) -> tensor<4x5x79xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [8:12, 6:8, 87:88] : (tensor<20x24x96xf64>) -> tensor<4x2x1xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %arg1 [0:4, 0:5, 79:80] : (tensor<4x7x80xf64>) -> tensor<4x5x1xf64>
// CHECK-NEXT:    %4 = stablehlo.concatenate %2, %0, dim = 2 : (tensor<4x2x1xf64>, tensor<4x2x79xf64>) -> tensor<4x2x80xf64>
// CHECK-NEXT:    %5 = stablehlo.concatenate %3, %1, dim = 2 : (tensor<4x5x1xf64>, tensor<4x5x79xf64>) -> tensor<4x5x80xf64>
// CHECK-NEXT:    %6 = stablehlo.pad %5, %arg2, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x5x80xf64>, tensor<f64>) -> tensor<4x6x80xf64>
// CHECK-NEXT:    %7 = stablehlo.concatenate %4, %6, dim = 1 : (tensor<4x2x80xf64>, tensor<4x6x80xf64>) -> tensor<4x8x80xf64>
// CHECK-NEXT:    stablehlo.return %7 : tensor<4x8x80xf64>
// CHECK-NEXT:  }

// ROTATE:  func.func @main(%arg0: tensor<20x24x96xf64>, %arg1: tensor<4x7x80xf64>, %arg2: tensor<f64>) -> tensor<4x8x80xf64> {
// ROTATE-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 6:8, 8:88] : (tensor<20x24x96xf64>) -> tensor<4x2x80xf64>
// ROTATE-NEXT:    %1 = "enzymexla.rotate"(%0) <{amount = 79 : si32, dimension = 2 : si32}> : (tensor<4x2x80xf64>) -> tensor<4x2x80xf64>
// ROTATE-NEXT:    %2 = stablehlo.slice %arg1 [0:4, 0:5, 0:80] : (tensor<4x7x80xf64>) -> tensor<4x5x80xf64>
// ROTATE-NEXT:    %3 = "enzymexla.rotate"(%2) <{amount = 79 : si32, dimension = 2 : si32}> : (tensor<4x5x80xf64>) -> tensor<4x5x80xf64>
// ROTATE-NEXT:    %4 = stablehlo.pad %3, %arg2, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x5x80xf64>, tensor<f64>) -> tensor<4x6x80xf64>
// ROTATE-NEXT:    %5 = stablehlo.concatenate %1, %4, dim = 1 : (tensor<4x2x80xf64>, tensor<4x6x80xf64>) -> tensor<4x8x80xf64>
// ROTATE-NEXT:    stablehlo.return %5 : tensor<4x8x80xf64>
// ROTATE-NEXT:  }