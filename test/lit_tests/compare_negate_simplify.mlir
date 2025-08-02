// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns=compare_negate_const_simplify --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefix=PARTIAL
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s --check-prefix=FULL

func.func @compare_neg_iota_const_ge() -> tensor<1520xi1> {
    %c = stablehlo.constant dense<-1518> : tensor<1520xi64>
    %0 = stablehlo.iota dim = 0 : tensor<1520xi64>
    %1 = stablehlo.negate %0 : tensor<1520xi64>
    %2 = stablehlo.compare GE, %1, %c : (tensor<1520xi64>, tensor<1520xi64>) -> tensor<1520xi1>
    return %2 : tensor<1520xi1>
}

// PARTIAL: func.func @compare_neg_iota_const_ge() -> tensor<1520xi1> {
// PARTIAL-NEXT:     %c = stablehlo.constant dense<-1518> : tensor<1520xi64>
// PARTIAL-NEXT:     %0 = stablehlo.iota dim = 0 : tensor<1520xi64>
// PARTIAL-NEXT:     %1 = stablehlo.negate %c : tensor<1520xi64>
// PARTIAL-NEXT:     %2 = stablehlo.compare  LE, %0, %1 : (tensor<1520xi64>, tensor<1520xi64>) -> tensor<1520xi1>
// PARTIAL-NEXT:     return %2 : tensor<1520xi1>
// PARTIAL-NEXT: }

// FULL: func.func @compare_neg_iota_const_ge() -> tensor<1520xi1> {
// FULL-NEXT:     %c = stablehlo.constant dense<false> : tensor<1xi1>
// FULL-NEXT:     %c_0 = stablehlo.constant dense<true> : tensor<i1>
// FULL-NEXT:     %0 = stablehlo.pad %c, %c_0, low = [1519], high = [0], interior = [0] : (tensor<1xi1>, tensor<i1>) -> tensor<1520xi1>
// FULL-NEXT:     return %0 : tensor<1520xi1>
// FULL-NEXT: }

func.func @compare_neg_neg() -> tensor<1520xi1> {
    %0 = stablehlo.iota dim = 0 : tensor<1520xi64>
    %1 = stablehlo.negate %0 : tensor<1520xi64>
    %2 = stablehlo.iota dim = 0 : tensor<1520xi64>
    %3 = stablehlo.negate %2 : tensor<1520xi64>
    %4 = stablehlo.compare GE, %1, %3 : (tensor<1520xi64>, tensor<1520xi64>) -> tensor<1520xi1>
    return %4 : tensor<1520xi1>
}

// PARTIAL: func.func @compare_neg_neg() -> tensor<1520xi1> {
// PARTIAL-NEXT:     %0 = stablehlo.iota dim = 0 : tensor<1520xi64>
// PARTIAL-NEXT:     %1 = stablehlo.iota dim = 0 : tensor<1520xi64>
// PARTIAL-NEXT:     %2 = stablehlo.compare  GE, %1, %0 : (tensor<1520xi64>, tensor<1520xi64>) -> tensor<1520xi1>
// PARTIAL-NEXT:     return %2 : tensor<1520xi1>
// PARTIAL-NEXT: }

// FULL: func.func @compare_neg_neg() -> tensor<1520xi1> {
// FULL-NEXT:     %0 = stablehlo.iota dim = 0 : tensor<1520xi64>
// FULL-NEXT:     %1 = stablehlo.compare  GE, %0, %0 : (tensor<1520xi64>, tensor<1520xi64>) -> tensor<1520xi1>
// FULL-NEXT:     return %1 : tensor<1520xi1>
// FULL-NEXT: }
