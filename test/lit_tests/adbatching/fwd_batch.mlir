// RUN: enzymexlamlir-opt --split-input-file --enzyme-diff-batch %s | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --enzyme-diff-batch --enzyme-batch-to-tensor %s | FileCheck %s --check-prefix=LEGAL
// 1. Scalar test
module {
  func.func @square(%x : tensor<f64>) -> tensor<f64>{
    %y = stablehlo.multiply %x, %x : tensor<f64>
    return %y : tensor<f64>
  }
  func.func @test1(%x : tensor<f64>, %dx1 : tensor<f64>, %dx2 : tensor<f64>) -> (tensor<f64>,tensor<f64>) {
    %r1 = enzyme.fwddiff @square(%x, %dx1) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>]} : (tensor<f64>, tensor<f64>) -> (tensor<f64>)
    %r2 = enzyme.fwddiff @square(%x, %dx2) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (tensor<f64>, tensor<f64>) -> (tensor<f64>)
    return %r1,%r2 : tensor<f64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @test1
// CHECK-SAME: (%[[PRIMAL:.*]]: tensor<f64>, %[[DIFF1:.*]]: tensor<f64>, %[[DIFF2:.*]]: tensor<f64>) -> (tensor<f64>, tensor<f64>)
// CHECK:         %[[CONCAT:.*]] = enzyme.concat(%[[DIFF1]], %[[DIFF2]]) : (tensor<f64>, tensor<f64>) -> tensor<2xf64>
// CHECK:         %[[BATCHED_RES:.*]] = enzyme.fwddiff @square(%[[PRIMAL]], %[[CONCAT]]) {{.*}} width = 2 {{.*}} : (tensor<f64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[RES0:.*]] = enzyme.extract %[[BATCHED_RES]][%[[C0]]] : (tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[RES1:.*]] = enzyme.extract %[[BATCHED_RES]][%[[C1]]] : (tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:    return %[[RES0]], %[[RES1]]

// LEGAL-LABEL: func.func @test1
// LEGAL-SAME: (%[[PRIMAL:.*]]: tensor<f64>, %[[DIFF1:.*]]: tensor<f64>, %[[DIFF2:.*]]: tensor<f64>) -> (tensor<f64>, tensor<f64>)
// LEGAL:         %[[CONCAT:.*]] = tensor.from_elements %[[DIFF1]], %[[DIFF2]] : tensor<2xf64>
// LEGAL:         %[[BATCHED_RES:.*]] = enzyme.fwddiff @square(%[[PRIMAL]], %[[CONCAT]]) {{.*}} width = 2 {{.*}} : (tensor<f64>, tensor<2xf64>) -> tensor<2xf64>
// LEGAL:         %[[C0:.*]] = arith.constant 0 : index
// LEGAL-NEXT:    %[[RES0:.*]] = tensor.extract %[[BATCHED_RES]][%[[C0]]] : tensor<2xf64>
// LEGAL-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// LEGAL-NEXT:    %[[RES1:.*]] = tensor.extract %[[BATCHED_RES]][%[[C1]]] : tensor<2xf64>
// LEGAL-NEXT:    return %[[RES0]], %[[RES1]]

// -----

// 2. Tensor test
module {
  func.func @square(%x : tensor<10xf64>) -> tensor<10xf64>{
    %y = stablehlo.multiply %x, %x : tensor<10xf64>
    return %y : tensor<10xf64>
  }
  func.func @test2(%x : tensor<10xf64>, %dx : tensor<10xf64>, %dx2 : tensor<10xf64>) -> (tensor<10xf64>,tensor<10xf64>) {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>]} : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>)
    %r2 = enzyme.fwddiff @square(%x, %dx2) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>)
    return %r,%r2 : tensor<10xf64>,tensor<10xf64>
  }
}

// CHECK-LABEL: func.func @test2
// CHECK-SAME: (%[[PRIMAL:.*]]: tensor<10xf64>, %[[DIFF1:.*]]: tensor<10xf64>, %[[DIFF2:.*]]: tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
// CHECK:         %[[CONCAT:.*]] = enzyme.concat(%[[DIFF1]], %[[DIFF2]]) : (tensor<10xf64>, tensor<10xf64>) -> tensor<2x10xf64>
// CHECK:         %[[BATCHED_RES:.*]] = enzyme.fwddiff @square(%[[PRIMAL]], %[[CONCAT]]) {{.*}} width = 2 {{.*}} : (tensor<10xf64>, tensor<2x10xf64>) -> tensor<2x10xf64>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[RES0:.*]] = enzyme.extract %[[BATCHED_RES]][%[[C0]]] : (tensor<2x10xf64>) -> tensor<10xf64>
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[RES1:.*]] = enzyme.extract %[[BATCHED_RES]][%[[C1]]] : (tensor<2x10xf64>) -> tensor<10xf64>
// CHECK-NEXT:    return %[[RES0]], %[[RES1]]

// LEGAL-LABEL: func.func @test2
// LEGAL-SAME: (%[[PRIMAL:.*]]: tensor<10xf64>, %[[DIFF1:.*]]: tensor<10xf64>, %[[DIFF2:.*]]: tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
// LEGAL:         %[[EDIFF1:.*]] = tensor.expand_shape %[[DIFF1]] {{\[\[0, 1\]\]}} output_shape [1, 10] : tensor<10xf64> into tensor<1x10xf64>
// LEGAL:         %[[EDIFF2:.*]] = tensor.expand_shape %[[DIFF2]] {{\[\[0, 1\]\]}} output_shape [1, 10] : tensor<10xf64> into tensor<1x10xf64>
// LEGAL:         %[[CONCAT:.*]] = tensor.concat dim(0) %[[EDIFF1]], %[[EDIFF2]] : (tensor<1x10xf64>, tensor<1x10xf64>) -> tensor<2x10xf64>
// LEGAL:         %[[BATCHED_RES:.*]] = enzyme.fwddiff @square(%[[PRIMAL]], %[[CONCAT]]) {{.*}} width = 2 {{.*}} : (tensor<10xf64>, tensor<2x10xf64>) -> tensor<2x10xf64>
// LEGAL:         %[[C0:.*]] = arith.constant 0 : index
// LEGAL-NEXT:    %[[RES0:.*]] = tensor.extract_slice %[[BATCHED_RES]][%[[C0]], 0] [1, 10] [1, 1] : tensor<2x10xf64> to tensor<10xf64>
// LEGAL-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// LEGAL-NEXT:    %[[RES1:.*]] = tensor.extract_slice %[[BATCHED_RES]][%[[C1]], 0] [1, 10] [1, 1] : tensor<2x10xf64> to tensor<10xf64>
// LEGAL-NEXT:    return %[[RES0]], %[[RES1]]
