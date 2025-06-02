// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @test1(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
    %0 = stablehlo.sign %arg0 : tensor<2x2xf64>
    %1 = stablehlo.abs %arg0 : tensor<2x2xf64>
    %2 = stablehlo.multiply %0, %1 : tensor<2x2xf64>
    return %2 : tensor<2x2xf64>
}

// CHECK-LABEL: func.func @test1(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
// CHECK-NEXT:    return %arg0 : tensor<2x2xf64>
// CHECK-NEXT:  }

func.func @test2(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
    %0 = stablehlo.sign %arg0 : tensor<2x2xf64>
    %1 = stablehlo.abs %arg0 : tensor<2x2xf64>
    %2 = stablehlo.multiply %0, %1 : tensor<2x2xf64>
    return %2 : tensor<2x2xf64>
}

// CHECK-LABEL: func.func @test2(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
// CHECK-NEXT:    return %arg0 : tensor<2x2xf64>
// CHECK-NEXT:  }
