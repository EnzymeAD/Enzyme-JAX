// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=factor_scalars_in_dot_general" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @main1(%arg0: tensor<10x10xf64> {enzymexla.memory_effects = []}) -> tensor<10x10xf64> attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.constant dense<4.0> : tensor<10x10xf64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<10x10xf64>
    %2 = stablehlo.dot_general %1, %arg0, contracting_dims = [0] x [0] : (tensor<10x10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
    return %2 : tensor<10x10xf64>
}

// CHECK: func.func @main1(%arg0: tensor<10x10xf64> {enzymexla.memory_effects = []}) -> tensor<10x10xf64> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %cst = stablehlo.constant dense<4.000000e+00> : tensor<10x10xf64>
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [0] x [0] : (tensor<10x10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
// CHECK-NEXT:   %1 = stablehlo.multiply %cst, %0 : tensor<10x10xf64>
// CHECK-NEXT:   return %1 : tensor<10x10xf64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<10x10xf64> {enzymexla.memory_effects = []}) -> tensor<10x10xf64> attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.constant dense<4.0> : tensor<10x10xf64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<10x10xf64>
    %2 = stablehlo.constant dense<2.0> : tensor<10x10xf64>
    %3 = stablehlo.multiply %2, %arg0 : tensor<10x10xf64>
    %4 = stablehlo.dot_general %1, %3, contracting_dims = [1] x [0] : (tensor<10x10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
    return %4 : tensor<10x10xf64>
}

// CHECK: func.func @main2(%arg0: tensor<10x10xf64> {enzymexla.memory_effects = []}) -> tensor<10x10xf64> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<10x10xf64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<4.000000e+00> : tensor<10x10xf64>
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [1] x [0] : (tensor<10x10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
// CHECK-NEXT:     %1 = stablehlo.multiply %cst_0, %cst : tensor<10x10xf64>
// CHECK-NEXT:     %2 = stablehlo.multiply %1, %0 : tensor<10x10xf64>
// CHECK-NEXT:     return %2 : tensor<10x10xf64>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<10x10xf64> {enzymexla.memory_effects = []}) -> tensor<10x10xf64> attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.constant dense<4.0> : tensor<10x10xf64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<10x10xf64>
    %2 = stablehlo.constant dense<2.0> : tensor<10x10xf64>
    %3 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<10x10xf64>) -> tensor<10x10xf64>
    %4 = stablehlo.multiply %2, %3 : tensor<10x10xf64>
    %5 = stablehlo.dot_general %1, %4, contracting_dims = [1] x [0] : (tensor<10x10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
    return %5 : tensor<10x10xf64>
}

//CHECK: func.func @main3(%arg0: tensor<10x10xf64> {enzymexla.memory_effects = []}) -> tensor<10x10xf64> attributes {enzymexla.memory_effects = []} {
//CHECK-NEXT:   %cst = stablehlo.constant dense<2.000000e+00> : tensor<10x10xf64>
//CHECK-NEXT:   %cst_0 = stablehlo.constant dense<4.000000e+00> : tensor<10x10xf64>
//CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<10x10xf64>) -> tensor<10x10xf64>
//CHECK-NEXT:   %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<10x10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
//CHECK-NEXT:   %2 = stablehlo.multiply %cst_0, %cst : tensor<10x10xf64>
//CHECK-NEXT:   %3 = stablehlo.multiply %2, %1 : tensor<10x10xf64>
//CHECK-NEXT:   return %3 : tensor<10x10xf64>
//CHECK-NEXT: }

func.func @main4(%arg0: tensor<10x10xf64> {enzymexla.memory_effects = []}) -> tensor<10x10xf64> attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.constant dense<4.0> : tensor<10x10xf64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<10x10xf64>
    %2 = stablehlo.dot_general %1, %arg0, contracting_dims = [0] x [0] : (tensor<10x10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
    %3 = stablehlo.add %2, %1 : tensor<10x10xf64>
    return %3 : tensor<10x10xf64>
}

// CHECK:  func.func @main4(%arg0: tensor<10x10xf64> {enzymexla.memory_effects = []}) -> tensor<10x10xf64> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:    %cst = stablehlo.constant dense<4.000000e+00> : tensor<10x10xf64>
// CHECK-NEXT:    %0 = stablehlo.multiply %cst, %arg0 : tensor<10x10xf64>
// CHECK-NEXT:    %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [0] x [0] : (tensor<10x10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
// CHECK-NEXT:    %2 = stablehlo.add %1, %0 : tensor<10x10xf64>
// CHECK-NEXT:    return %2 : tensor<10x10xf64>
// CHECK-NEXT:  }
