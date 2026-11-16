// RUN: enzymexlamlir-opt %s --enzyme-refine-arguments="refined-types=tensor<4xi8>,tensor<i32>" | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-refine-arguments="refined-types=tensor<4xi8>,tensor<i32>" --stablehlo-refine-shapes | FileCheck %s --check-prefix=REFINED

// A byte argument carried through a loop is still dynamically shaped when
// the arguments become static; the result type follows the operand until
// shape refinement reaches the loop.
module {
  func.func @main(%arg0: tensor<?xi8>, %arg1: tensor<i32>) -> (tensor<?xi8>, tensor<i32>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<i64>
    %1:2 = stablehlo.while(%iterArg = %c, %iterArg_1 = %arg0) : tensor<i64>, tensor<?xi8>
    cond {
      %2 = stablehlo.compare LT, %iterArg, %0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %iterArg, %c_0 : tensor<i64>
      stablehlo.return %2, %iterArg_1 : tensor<i64>, tensor<?xi8>
    }
    return %1#1, %arg1 : tensor<?xi8>, tensor<i32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4xi8>, %arg1: tensor<i32>) -> (tensor<?xi8>, tensor<i32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<4> : tensor<1xi64>
// CHECK-NEXT:    %0 = stablehlo.custom_call @stablehlo.shape_refinement_operand_wrapper(%arg0, %c) {indices_of_shape_operands = dense<1> : tensor<1xi64>} : (tensor<4xi8>, tensor<1xi64>) -> tensor<?xi8>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %1 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:    %2:2 = stablehlo.while(%iterArg = %c_0, %iterArg_2 = %0) : tensor<i64>, tensor<?xi8>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %3 = stablehlo.compare LT, %iterArg, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %3 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %3 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %3, %iterArg_2 : tensor<i64>, tensor<?xi8>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %2#1, %arg1 : tensor<?xi8>, tensor<i32>
// CHECK-NEXT:  }

// REFINED:  func.func @main(%arg0: tensor<4xi8>, %arg1: tensor<i32>) -> (tensor<4xi8>, tensor<i32>) {
// REFINED-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// REFINED-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// REFINED-NEXT:    %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<i64>
// REFINED-NEXT:    %1:2 = stablehlo.while(%iterArg = %c_0, %iterArg_1 = %arg0) : tensor<i64>, tensor<4xi8>
// REFINED-NEXT:    cond {
// REFINED-NEXT:      %2 = stablehlo.compare LT, %iterArg, %0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REFINED-NEXT:      stablehlo.return %2 : tensor<i1>
// REFINED-NEXT:    } do {
// REFINED-NEXT:      %2 = stablehlo.add %iterArg, %c : tensor<i64>
// REFINED-NEXT:      stablehlo.return %2, %iterArg_1 : tensor<i64>, tensor<4xi8>
// REFINED-NEXT:    }
// REFINED-NEXT:    return %1#1, %arg1 : tensor<4xi8>, tensor<i32>
// REFINED-NEXT:  }
