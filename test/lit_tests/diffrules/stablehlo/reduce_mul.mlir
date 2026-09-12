// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%a: tensor<9xf32>, %init: tensor<f32>) -> tensor<f32> {
    %6 = "stablehlo.reduce"(%a, %init) <{dimensions = array<i64: 0>}> ({
        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        %8 = "stablehlo.multiply"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%8) : (tensor<f32>) -> ()
        }) : (tensor<9xf32>, tensor<f32>) -> tensor<f32>
    return %6 : tensor<f32>
  }
}



// CHECK:  func.func @main(%arg0: tensor<9xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<9xf32>, tensor<f32>) {
// CHECK-DAG:     %[[ONES:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<9xf32>
// CHECK-DAG:     %[[ONE:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-DAG:     %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-DAG:     %[[ZEROS:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<9xf32>
// CHECK:         %[[IS_ZERO:.*]] = stablehlo.compare EQ, %arg0, %[[ZEROS]] : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
// CHECK-NEXT:    %[[ZERO_INDICATORS:.*]] = stablehlo.convert %[[IS_ZERO]] : (tensor<9xi1>) -> tensor<9xi32>
// CHECK-NEXT:    %[[ZERO_COUNT:.*]] = stablehlo.reduce(%[[ZERO_INDICATORS]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0]
// CHECK:         %[[NO_ZEROS:.*]] = stablehlo.compare EQ, %[[ZERO_COUNT]], %[[ZERO]]
// CHECK-NEXT:    %[[ONE_ZERO:.*]] = stablehlo.compare EQ, %[[ZERO_COUNT]], %[[ONE]]
// CHECK:         %[[NO_ZEROS_BCAST:.*]] = stablehlo.broadcast_in_dim %[[NO_ZEROS]], dims = []
// CHECK-NEXT:    %[[ONE_ZERO_BCAST:.*]] = stablehlo.broadcast_in_dim %[[ONE_ZERO]], dims = []
// CHECK-NEXT:    %[[IS_SOLE_ZERO:.*]] = stablehlo.and %[[ONE_ZERO_BCAST]], %[[IS_ZERO]]
// CHECK-NEXT:    %[[HAS_DERIVATIVE:.*]] = stablehlo.or %[[NO_ZEROS_BCAST]], %[[IS_SOLE_ZERO]]
// CHECK-NEXT:    %[[SAFE_VALUES:.*]] = stablehlo.select %[[IS_ZERO]], %[[ONES]], %arg0
// CHECK-NEXT:    %[[NONZERO_PRODUCT:.*]] = stablehlo.reduce(%[[SAFE_VALUES]] init: %arg1) applies stablehlo.multiply across dimensions = [0]
// CHECK-NEXT:    %[[PRODUCT_BCAST:.*]] = stablehlo.broadcast_in_dim %[[NONZERO_PRODUCT]], dims = []
// CHECK-NEXT:    %[[QUOTIENT:.*]] = stablehlo.divide %[[PRODUCT_BCAST]], %[[SAFE_VALUES]]
// CHECK-NEXT:    %[[VALUE_DERIVATIVE:.*]] = stablehlo.select %[[HAS_DERIVATIVE]], %[[QUOTIENT]], %[[ZEROS]]
// CHECK-NEXT:    %[[VALUE_DIFFE:.*]] = stablehlo.multiply {{.*}}, %[[VALUE_DERIVATIVE]]
// CHECK-NEXT:    %[[INPUT_PRODUCT:.*]] = stablehlo.reduce(%arg0 init: {{.*}}) applies stablehlo.multiply across dimensions = [0]
// CHECK-NEXT:    %[[INIT_DIFFE:.*]] = stablehlo.multiply %[[INPUT_PRODUCT]], %arg2
// CHECK-NEXT:    return %[[VALUE_DIFFE]], %[[INIT_DIFFE]] : tensor<9xf32>, tensor<f32>
// CHECK-NEXT:  }
