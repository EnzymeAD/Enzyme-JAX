// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns=select_simplify --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

func.func @main1(%pred : tensor<i1>) -> tensor<32xi1> {
    %trues = stablehlo.constant dense<true> : tensor<32xi1>
    %falses = stablehlo.constant dense<false> : tensor<32xi1>
    %0 = stablehlo.select %pred, %trues, %falses : tensor<i1>, tensor<32xi1>
    return %0 : tensor<32xi1>
}


// CHECK: func.func @main1(%arg0: tensor<i1>) -> tensor<32xi1> {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<i1>) -> tensor<32xi1>
// CHECK-NEXT:     return %0 : tensor<32xi1>
// CHECK-NEXT: }

func.func @main2(%pred : tensor<32xi1>) -> tensor<32xi1> {
    %trues = stablehlo.constant dense<true> : tensor<32xi1>
    %falses = stablehlo.constant dense<false> : tensor<32xi1>
    %0 = stablehlo.select %pred, %trues, %falses : tensor<32xi1>, tensor<32xi1>
    return %0 : tensor<32xi1>
}

// CHECK: func.func @main2(%arg0: tensor<32xi1>) -> tensor<32xi1> {
// CHECK-NEXT:     return %arg0 : tensor<32xi1>
// CHECK-NEXT: }

func.func @main3(%pred : tensor<i1>) -> tensor<32xi1> {
    %trues = stablehlo.constant dense<true> : tensor<32xi1>
    %falses = stablehlo.constant dense<false> : tensor<32xi1>
    %0 = stablehlo.select %pred, %falses, %trues : tensor<i1>, tensor<32xi1>
    return %0 : tensor<32xi1>
}

// CHECK: func.func @main3(%arg0: tensor<i1>) -> tensor<32xi1> {
// CHECK-NEXT:     %0 = stablehlo.not %arg0 : tensor<i1>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i1>) -> tensor<32xi1>
// CHECK-NEXT:     return %1 : tensor<32xi1>
// CHECK-NEXT: }

func.func @main4(%pred : tensor<32xi1>) -> tensor<32xi1> {
    %trues = stablehlo.constant dense<true> : tensor<32xi1>
    %falses = stablehlo.constant dense<false> : tensor<32xi1>
    %0 = stablehlo.select %pred, %falses, %trues : tensor<32xi1>, tensor<32xi1>
    return %0 : tensor<32xi1>
}

// CHECK: func.func @main4(%arg0: tensor<32xi1>) -> tensor<32xi1> {
// CHECK-NEXT:     %0 = stablehlo.not %arg0 : tensor<32xi1>
// CHECK-NEXT:     return %0 : tensor<32xi1>
// CHECK-NEXT: }
