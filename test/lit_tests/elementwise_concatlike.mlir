// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @rotate_add(%arg0: tensor<4x1520x3056xf64>, %arg1: tensor<4x1520x3056xf64>) -> tensor<4x1520x3056xf64> {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 2 : si32, dimension = 2 : si32}> : (tensor<4x1520x3056xf64>) -> tensor<4x1520x3056xf64>
    %1 = "enzymexla.rotate"(%arg1) <{amount = 2 : si32, dimension = 2 : si32}> : (tensor<4x1520x3056xf64>) -> tensor<4x1520x3056xf64>
    %2 = stablehlo.add %0, %1 : tensor<4x1520x3056xf64>
    return %2 : tensor<4x1520x3056xf64>
}

// CHECK: func.func @rotate_add(%arg0: tensor<4x1520x3056xf64>, %arg1: tensor<4x1520x3056xf64>) -> tensor<4x1520x3056xf64> {
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %arg1 : tensor<4x1520x3056xf64>
// CHECK-NEXT:     %1 = "enzymexla.rotate"(%0) <{amount = 2 : si32, dimension = 2 : si32}> : (tensor<4x1520x3056xf64>) -> tensor<4x1520x3056xf64>
// CHECK-NEXT:     return %1 : tensor<4x1520x3056xf64>
// CHECK-NEXT: }

func.func @rotate_wrap(%arg0: tensor<4x1520x3056xf64>, %arg1: tensor<4x1520x3056xf64>) -> tensor<4x1536x3056xf64> {
    %0 = "enzymexla.wrap"(%arg0) <{dimension = 1 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<4x1536x3056xf64>
    %1 = "enzymexla.wrap"(%arg1) <{dimension = 1 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<4x1536x3056xf64>
    %2 = stablehlo.add %0, %1 : tensor<4x1536x3056xf64>
    return %2 : tensor<4x1536x3056xf64>
}

// CHECK: func.func @rotate_wrap(%arg0: tensor<4x1520x3056xf64>, %arg1: tensor<4x1520x3056xf64>) -> tensor<4x1536x3056xf64> {
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %arg1 : tensor<4x1520x3056xf64>
// CHECK-NEXT:     %1 = "enzymexla.wrap"(%0) <{dimension = 1 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<4x1536x3056xf64>
// CHECK-NEXT:     return %1 : tensor<4x1536x3056xf64>
// CHECK-NEXT: }

func.func @rotate_extend(%arg0: tensor<4x1520x3056xf64>, %arg1: tensor<4x1520x3056xf64>) -> tensor<4x1536x3056xf64> {
    %0 = "enzymexla.extend"(%arg0) <{dimension = 1 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<4x1536x3056xf64>
    %1 = "enzymexla.extend"(%arg1) <{dimension = 1 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<4x1536x3056xf64>
    %2 = stablehlo.add %0, %1 : tensor<4x1536x3056xf64>
    return %2 : tensor<4x1536x3056xf64>
}

// CHECK: func.func @rotate_extend(%arg0: tensor<4x1520x3056xf64>, %arg1: tensor<4x1520x3056xf64>) -> tensor<4x1536x3056xf64> {
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %arg1 : tensor<4x1520x3056xf64>
// CHECK-NEXT:     %1 = "enzymexla.extend"(%0) <{dimension = 1 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<4x1536x3056xf64>
// CHECK-NEXT:     return %1 : tensor<4x1536x3056xf64>
// CHECK-NEXT: }
