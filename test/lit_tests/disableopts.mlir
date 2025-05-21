// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @simpleopt(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %c = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
    %0 = stablehlo.add %arg0, %c : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

func.func @simpleopt2(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {enzymexla.disable_hlo_opts} {
    %c = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
    %0 = stablehlo.add %arg0, %c : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

func.func @simpleopt3(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = stablehlo.add %0, %0 : tensor<2x2xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
}

func.func @simpleopt4(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %1 = stablehlo.add %0, %0 : tensor<?x?xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
}

// CHECK: func.func @simpleopt(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:     return %arg0 : tensor<2x2xf32>
// CHECK-NEXT: }

// CHECK: func.func @simpleopt2(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {enzymexla.disable_hlo_opts} {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %cst : tensor<2x2xf32>
// CHECK-NEXT:     return %0 : tensor<2x2xf32>
// CHECK-NEXT: }

// CHECK: func.func @simpleopt3(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %arg0 : tensor<2x2xf32>
// CHECK-NEXT:     return %0 : tensor<2x2xf32>
// CHECK-NEXT: }

// CHECK: func.func @simpleopt4(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:     %1 = stablehlo.add %0, %0 : tensor<?x?xf32>
// CHECK-NEXT:     %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:     return %2 : tensor<?x?xf32>
// CHECK-NEXT: }
