// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

module {
  func.func @dus_concat_test(
    %A: tensor<1x1024x8xf64>,   // Input A
    %B: tensor<1x1024x1008xf64>, // Input B (target for DUS)
    %C: tensor<1x1024x8xf64>,   // Input C
    %arg3: tensor<1x1010x1008xf64>   // Update tensor
  ) -> tensor<1x1024x1024xf64>  {

  %c_158 = stablehlo.constant dense<0> : tensor<i64>
  %c_95 = stablehlo.constant dense<7> : tensor<i64>
  %c_107 = stablehlo.constant dense<8> : tensor<i64>

  %42 = stablehlo.concatenate %A, %B, %C, dim = 2 : (tensor<1x1024x8xf64>, tensor<1x1024x1008xf64>, tensor<1x1024x8xf64>) -> tensor<1x1024x1024xf64>

  %dus = stablehlo.dynamic_update_slice %42, %arg3, %c_158, %c_95, %c_107 : (tensor<1x1024x1024xf64>, tensor<1x1010x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1024x1024xf64> 

    func.return %dus : tensor<1x1024x1024xf64> 
  }
}

// CHECK:  func.func @dus_concat_test(%arg0: tensor<1x1024x8xf64>, %arg1: tensor<1x1024x1008xf64>, %arg2: tensor<1x1024x8xf64>, %arg3: tensor<1x1010x1008xf64>) -> tensor<1x1024x1024xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:1, 0:7, 0:1008] : (tensor<1x1024x1008xf64>) -> tensor<1x7x1008xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:1, 1017:1024, 0:1008] : (tensor<1x1024x1008xf64>) -> tensor<1x7x1008xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %arg3, %1, dim = 1 : (tensor<1x7x1008xf64>, tensor<1x1010x1008xf64>, tensor<1x7x1008xf64>) -> tensor<1x1024x1008xf64>
// CHECK-NEXT:    %3 = stablehlo.concatenate %arg0, %2, %arg2, dim = 2 : (tensor<1x1024x8xf64>, tensor<1x1024x1008xf64>, tensor<1x1024x8xf64>) -> tensor<1x1024x1024xf64>
// CHECK-NEXT:    return %3 : tensor<1x1024x1024xf64>
// CHECK-NEXT:  }