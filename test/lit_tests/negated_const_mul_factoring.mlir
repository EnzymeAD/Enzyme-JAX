// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Test 1: (a * cst) + (b * -cst) -> (a - b) * cst
func.func @add_mul_cst_mul_negcst(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
    %cst = stablehlo.constant dense<3.0> : tensor<10xf64>
    %negcst = stablehlo.constant dense<-3.0> : tensor<10xf64>
    %0 = stablehlo.multiply %arg0, %cst : tensor<10xf64>
    %1 = stablehlo.multiply %arg1, %negcst : tensor<10xf64>
    %2 = stablehlo.add %0, %1 : tensor<10xf64>
    return %2 : tensor<10xf64>
}

// CHECK-LABEL: func.func @add_mul_cst_mul_negcst
// CHECK-SAME: (%arg0: tensor<10xf64>, %arg1: tensor<10xf64>)
// CHECK:      %cst = stablehlo.constant dense<3.0{{.*}}>
// CHECK-NEXT: %0 = stablehlo.subtract %arg0, %arg1
// CHECK-NEXT: %1 = stablehlo.multiply %0, %cst
// CHECK-NEXT: return %1

// Test 2: (cst * a) + (b * -cst) -> (a - b) * cst
func.func @add_cstmul_mul_negcst(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
    %cst = stablehlo.constant dense<3.0> : tensor<10xf64>
    %negcst = stablehlo.constant dense<-3.0> : tensor<10xf64>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<10xf64>
    %1 = stablehlo.multiply %arg1, %negcst : tensor<10xf64>
    %2 = stablehlo.add %0, %1 : tensor<10xf64>
    return %2 : tensor<10xf64>
}

// CHECK-LABEL: func.func @add_cstmul_mul_negcst
// CHECK-SAME: (%arg0: tensor<10xf64>, %arg1: tensor<10xf64>)
// CHECK:      %cst = stablehlo.constant dense<3.0{{.*}}>
// CHECK-NEXT: %0 = stablehlo.subtract %arg0, %arg1
// CHECK-NEXT: %1 = stablehlo.multiply %0, %cst
// CHECK-NEXT: return %1

// Test 3: (a * cst) + (-cst * b) -> (a - b) * cst
func.func @add_mul_cst_negcstmul(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
    %cst = stablehlo.constant dense<3.0> : tensor<10xf64>
    %negcst = stablehlo.constant dense<-3.0> : tensor<10xf64>
    %0 = stablehlo.multiply %arg0, %cst : tensor<10xf64>
    %1 = stablehlo.multiply %negcst, %arg1 : tensor<10xf64>
    %2 = stablehlo.add %0, %1 : tensor<10xf64>
    return %2 : tensor<10xf64>
}

// CHECK-LABEL: func.func @add_mul_cst_negcstmul
// CHECK-SAME: (%arg0: tensor<10xf64>, %arg1: tensor<10xf64>)
// CHECK:      %cst = stablehlo.constant dense<3.0{{.*}}>
// CHECK-NEXT: %0 = stablehlo.subtract %arg0, %arg1
// CHECK-NEXT: %1 = stablehlo.multiply %0, %cst
// CHECK-NEXT: return %1

// Test 4: (cst * a) + (-cst * b) -> (a - b) * cst
func.func @add_cstmul_negcstmul(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
    %cst = stablehlo.constant dense<3.0> : tensor<10xf64>
    %negcst = stablehlo.constant dense<-3.0> : tensor<10xf64>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<10xf64>
    %1 = stablehlo.multiply %negcst, %arg1 : tensor<10xf64>
    %2 = stablehlo.add %0, %1 : tensor<10xf64>
    return %2 : tensor<10xf64>
}

// CHECK-LABEL: func.func @add_cstmul_negcstmul
// CHECK-SAME: (%arg0: tensor<10xf64>, %arg1: tensor<10xf64>)
// CHECK:      %cst = stablehlo.constant dense<3.0{{.*}}>
// CHECK-NEXT: %0 = stablehlo.subtract %arg0, %arg1
// CHECK-NEXT: %1 = stablehlo.multiply %0, %cst
// CHECK-NEXT: return %1

// Test 5: (a * -cst) + (b * cst) -> (b - a) * cst
// Here the negative constant is on the lhs, so we get (b - a) * cst
func.func @add_mul_negcst_mul_cst(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
    %cst = stablehlo.constant dense<3.0> : tensor<10xf64>
    %negcst = stablehlo.constant dense<-3.0> : tensor<10xf64>
    %0 = stablehlo.multiply %arg0, %negcst : tensor<10xf64>
    %1 = stablehlo.multiply %arg1, %cst : tensor<10xf64>
    %2 = stablehlo.add %0, %1 : tensor<10xf64>
    return %2 : tensor<10xf64>
}

// CHECK-LABEL: func.func @add_mul_negcst_mul_cst
// CHECK-SAME: (%arg0: tensor<10xf64>, %arg1: tensor<10xf64>)
// CHECK:      %cst = stablehlo.constant dense<-3.0{{.*}}>
// CHECK-NEXT: %0 = stablehlo.subtract %arg0, %arg1
// CHECK-NEXT: %1 = stablehlo.multiply %0, %cst
// CHECK-NEXT: return %1

// Test 6: (a * cst) - (b * -cst) -> (a + b) * cst
func.func @sub_mul_cst_mul_negcst(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
    %cst = stablehlo.constant dense<3.0> : tensor<10xf64>
    %negcst = stablehlo.constant dense<-3.0> : tensor<10xf64>
    %0 = stablehlo.multiply %arg0, %cst : tensor<10xf64>
    %1 = stablehlo.multiply %arg1, %negcst : tensor<10xf64>
    %2 = stablehlo.subtract %0, %1 : tensor<10xf64>
    return %2 : tensor<10xf64>
}

// CHECK-LABEL: func.func @sub_mul_cst_mul_negcst
// CHECK-SAME: (%arg0: tensor<10xf64>, %arg1: tensor<10xf64>)
// CHECK:      %cst = stablehlo.constant dense<3.0{{.*}}>
// CHECK-NEXT: %0 = stablehlo.add %arg0, %arg1
// CHECK-NEXT: %1 = stablehlo.multiply %0, %cst
// CHECK-NEXT: return %1

// Test 7: (cst * a) - (-cst * b) -> (a + b) * cst
func.func @sub_cstmul_negcstmul(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
    %cst = stablehlo.constant dense<3.0> : tensor<10xf64>
    %negcst = stablehlo.constant dense<-3.0> : tensor<10xf64>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<10xf64>
    %1 = stablehlo.multiply %negcst, %arg1 : tensor<10xf64>
    %2 = stablehlo.subtract %0, %1 : tensor<10xf64>
    return %2 : tensor<10xf64>
}

// CHECK-LABEL: func.func @sub_cstmul_negcstmul
// CHECK-SAME: (%arg0: tensor<10xf64>, %arg1: tensor<10xf64>)
// CHECK:      %cst = stablehlo.constant dense<3.0{{.*}}>
// CHECK-NEXT: %0 = stablehlo.add %arg0, %arg1
// CHECK-NEXT: %1 = stablehlo.multiply %0, %cst
// CHECK-NEXT: return %1

// Test 8: (a * -cst) - (b * cst) -> (a + b) * (-cst)
// Here the negative constant is on the lhs
func.func @sub_mul_negcst_mul_cst(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
    %cst = stablehlo.constant dense<3.0> : tensor<10xf64>
    %negcst = stablehlo.constant dense<-3.0> : tensor<10xf64>
    %0 = stablehlo.multiply %arg0, %negcst : tensor<10xf64>
    %1 = stablehlo.multiply %arg1, %cst : tensor<10xf64>
    %2 = stablehlo.subtract %0, %1 : tensor<10xf64>
    return %2 : tensor<10xf64>
}

// CHECK-LABEL: func.func @sub_mul_negcst_mul_cst
// CHECK-SAME: (%arg0: tensor<10xf64>, %arg1: tensor<10xf64>)
// CHECK:      %cst = stablehlo.constant dense<-3.0{{.*}}>
// CHECK-NEXT: %0 = stablehlo.add %arg0, %arg1
// CHECK-NEXT: %1 = stablehlo.multiply %0, %cst
// CHECK-NEXT: return %1

// Test 9: Integer types - (a * cst) + (b * -cst) -> (a - b) * cst
func.func @add_mul_cst_mul_negcst_int(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32> {
    %cst = stablehlo.constant dense<7> : tensor<10xi32>
    %negcst = stablehlo.constant dense<-7> : tensor<10xi32>
    %0 = stablehlo.multiply %arg0, %cst : tensor<10xi32>
    %1 = stablehlo.multiply %arg1, %negcst : tensor<10xi32>
    %2 = stablehlo.add %0, %1 : tensor<10xi32>
    return %2 : tensor<10xi32>
}

// CHECK-LABEL: func.func @add_mul_cst_mul_negcst_int
// CHECK-SAME: (%arg0: tensor<10xi32>, %arg1: tensor<10xi32>)
// CHECK:      %c = stablehlo.constant dense<7>
// CHECK-NEXT: %0 = stablehlo.subtract %arg0, %arg1
// CHECK-NEXT: %1 = stablehlo.multiply %0, %c
// CHECK-NEXT: return %1

// Test 10: Integer types - (a * cst) - (b * -cst) -> (a + b) * cst
func.func @sub_mul_cst_mul_negcst_int(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32> {
    %cst = stablehlo.constant dense<7> : tensor<10xi32>
    %negcst = stablehlo.constant dense<-7> : tensor<10xi32>
    %0 = stablehlo.multiply %arg0, %cst : tensor<10xi32>
    %1 = stablehlo.multiply %arg1, %negcst : tensor<10xi32>
    %2 = stablehlo.subtract %0, %1 : tensor<10xi32>
    return %2 : tensor<10xi32>
}

// CHECK-LABEL: func.func @sub_mul_cst_mul_negcst_int
// CHECK-SAME: (%arg0: tensor<10xi32>, %arg1: tensor<10xi32>)
// CHECK:      %c = stablehlo.constant dense<7>
// CHECK-NEXT: %0 = stablehlo.add %arg0, %arg1
// CHECK-NEXT: %1 = stablehlo.multiply %0, %c
// CHECK-NEXT: return %1

// Test 11: 2D tensor - (a * cst) + (b * -cst) -> (a - b) * cst
func.func @add_mul_cst_mul_negcst_2d(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
    %cst = stablehlo.constant dense<2.5> : tensor<4x5xf32>
    %negcst = stablehlo.constant dense<-2.5> : tensor<4x5xf32>
    %0 = stablehlo.multiply %arg0, %cst : tensor<4x5xf32>
    %1 = stablehlo.multiply %arg1, %negcst : tensor<4x5xf32>
    %2 = stablehlo.add %0, %1 : tensor<4x5xf32>
    return %2 : tensor<4x5xf32>
}

// CHECK-LABEL: func.func @add_mul_cst_mul_negcst_2d
// CHECK-SAME: (%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>)
// CHECK:      %cst = stablehlo.constant dense<2.5{{.*}}>
// CHECK-NEXT: %0 = stablehlo.subtract %arg0, %arg1
// CHECK-NEXT: %1 = stablehlo.multiply %0, %cst
// CHECK-NEXT: return %1
