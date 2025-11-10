// RUN: enzymexlamlir-opt --split-input-file --enzyme-diff-batch %s | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --enzyme-diff-batch --enzyme-batch-to-tensor %s | FileCheck %s --check-prefix=LEGAL

//1. Scalar test
module {
  func.func @square(%x : tensor<f64>) -> tensor<f64>{
    %y = stablehlo.multiply %x, %x : tensor<f64>
    return %y : tensor<f64>
  }
  func.func @test1(%x : tensor<f64>, %dr1 : tensor<f64>, %dr2 : tensor<f64>) -> (tensor<f64>,tensor<f64>) {
    %r, %dx1 = enzyme.autodiff @square(%x, %dr1) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>] } : (tensor<f64>, tensor<f64>) -> (tensor<f64>,tensor<f64>) 
    %r2, %dx2 = enzyme.autodiff @square(%x, %dr2) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>] } : (tensor<f64>, tensor<f64>) -> (tensor<f64>,tensor<f64>) 
    return %dx1,%dx2 : tensor<f64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @test1
// CHECK-SAME: (%[[PRIMAL:.*]]: tensor<f64>, %[[DIFF1:.*]]: tensor<f64>, %[[DIFF2:.*]]: tensor<f64>) -> (tensor<f64>, tensor<f64>)
// CHECK:         %[[CONCAT:.*]] = enzyme.concat(%[[DIFF1]], %[[DIFF2]]) : (tensor<f64>, tensor<f64>) -> tensor<2xf64>
// CHECK:         %[[BATCHED_RES_BASE:.*]]:2 = enzyme.autodiff @square(%[[PRIMAL]], %[[CONCAT]]) {{.*}} width = 2 {{.*}} : (tensor<f64>, tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
// CHECK:         %[[RES0:.*]] = enzyme.extract %[[BATCHED_RES_BASE]]#1[0] : (tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[RES1:.*]] = enzyme.extract %[[BATCHED_RES_BASE]]#1[1] : (tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:    return %[[RES0]], %[[RES1]]

// LEGAL-LABEL: func.func @test1
// LEGAL-SAME: (%[[PRIMAL:.*]]: tensor<f64>, %[[DIFF1:.*]]: tensor<f64>, %[[DIFF2:.*]]: tensor<f64>) -> (tensor<f64>, tensor<f64>)
// LEGAL:         %[[EDIFF1:.*]] = stablehlo.reshape %[[DIFF1]] : (tensor<f64>) -> tensor<1xf64>
// LEGAL:         %[[EDIFF2:.*]] = stablehlo.reshape %[[DIFF2]] : (tensor<f64>) -> tensor<1xf64>
// LEGAL:         %[[CONCAT:.*]] = stablehlo.concatenate %[[EDIFF1]], %[[EDIFF2]], dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
// LEGAL:         %[[BATCHED_RES_BASE:.*]]:2 = enzyme.autodiff @square(%[[PRIMAL]], %[[CONCAT]]) {{.*}} width = 2 {{.*}} : (tensor<f64>, tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
// LEGAL:         %[[R0:.*]] = stablehlo.slice %[[BATCHED_RES_BASE]]#1 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// LEGAL-NEXT:    %[[RES0:.*]] = stablehlo.reshape %[[R0]] : (tensor<1xf64>) -> tensor<f64>
// LEGAL-NEXT:    %[[R1:.*]] = stablehlo.slice %[[BATCHED_RES_BASE]]#1 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// LEGAL-NEXT:    %[[RES1:.*]] = stablehlo.reshape %[[R1]] : (tensor<1xf64>) -> tensor<f64>
// LEGAL-NEXT:    return %[[RES0]], %[[RES1]]

// -----

//2. Tensor test
module {
  func.func @square(%x : tensor<10xf64>) -> tensor<10xf64>{
    %y = stablehlo.multiply %x, %x : tensor<10xf64>
    return %y : tensor<10xf64>
  }
  func.func @test2(%x : tensor<10xf64>, %dr1 : tensor<10xf64>, %dr2 : tensor<10xf64>) -> (tensor<10xf64>,tensor<10xf64>) {
    %r, %dx1 = enzyme.autodiff @square(%x, %dr1) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>]} : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
    %r2, %dx2 = enzyme.autodiff @square(%x, %dr2) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>]} : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
    return %dx1,%dx2 : tensor<10xf64>,tensor<10xf64>
  }
}


// CHECK-LABEL: func.func @test2
// CHECK-SAME: (%[[PRIMAL:.*]]: tensor<10xf64>, %[[DIFF1:.*]]: tensor<10xf64>, %[[DIFF2:.*]]: tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
// CHECK:         %[[CONCAT:.*]] = enzyme.concat(%[[DIFF1]], %[[DIFF2]]) : (tensor<10xf64>, tensor<10xf64>) -> tensor<2x10xf64>
// CHECK:         %[[BATCHED_RES_BASE:.*]]:2 = enzyme.autodiff @square(%[[PRIMAL]], %[[CONCAT]]) {{.*}} width = 2 {{.*}} : (tensor<10xf64>, tensor<2x10xf64>) -> (tensor<10xf64>, tensor<2x10xf64>)
// CHECK-NEXT:    %[[RES0:.*]] = enzyme.extract %[[BATCHED_RES_BASE]]#1[0] : (tensor<2x10xf64>) -> tensor<10xf64>
// CHECK-NEXT:    %[[RES1:.*]] = enzyme.extract %[[BATCHED_RES_BASE]]#1[1] : (tensor<2x10xf64>) -> tensor<10xf64>
// CHECK-NEXT:    return %[[RES0]], %[[RES1]]

// LEGAL-LABEL: func.func @test2
// LEGAL-SAME: (%[[PRIMAL:.*]]: tensor<10xf64>, %[[DIFF1:.*]]: tensor<10xf64>, %[[DIFF2:.*]]: tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
// LEGAL:         %[[EDIFF1:.*]] = stablehlo.reshape %[[DIFF1]] : (tensor<10xf64>) -> tensor<1x10xf64>
// LEGAL:         %[[EDIFF2:.*]] = stablehlo.reshape %[[DIFF2]] : (tensor<10xf64>) -> tensor<1x10xf64>
// LEGAL:         %[[CONCAT:.*]] = stablehlo.concatenate %[[EDIFF1]], %[[EDIFF2]], dim = 0 : (tensor<1x10xf64>, tensor<1x10xf64>) -> tensor<2x10xf64>
// LEGAL:         %[[BATCHED_RES_BASE:.*]]:2 = enzyme.autodiff @square(%[[PRIMAL]], %[[CONCAT]]) {{.*}} width = 2 {{.*}} : (tensor<10xf64>, tensor<2x10xf64>) -> (tensor<10xf64>, tensor<2x10xf64>)
// LEGAL:         %[[R0:.*]] = stablehlo.slice %[[BATCHED_RES_BASE]]#1 [0:1, 0:10] : (tensor<2x10xf64>) -> tensor<1x10xf64>
// LEGAL-NEXT:    %[[RES0:.*]] = stablehlo.reshape %[[R0]] : (tensor<1x10xf64>) -> tensor<10xf64>
// LEGAL-NEXT:    %[[R1:.*]] = stablehlo.slice %[[BATCHED_RES_BASE]]#1 [1:2, 0:10] : (tensor<2x10xf64>) -> tensor<1x10xf64>
// LEGAL-NEXT:    %[[RES1:.*]] = stablehlo.reshape %[[R1]] : (tensor<1x10xf64>) -> tensor<10xf64>
// LEGAL-NEXT:    return %[[RES0]], %[[RES1]]
