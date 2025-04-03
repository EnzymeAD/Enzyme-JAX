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
  func.func @dus_concat_test2(
    %A: tensor<144x1024x8xf64>,   // Input A
    %B: tensor<144x1024x1008xf64>, // Input B (target for DUS)
    %C: tensor<144x1024x8xf64>,   // Input C
    %arg3: tensor<128x1008x1008xf64>,   // Update tensor
    %arg4: tensor<128x1008x1008xf64>   // Update tensor
  ) -> (tensor<144x1024x1024xf64>, tensor<144x1024x1024xf64>)  {


    %c_31 = stablehlo.constant dense<8> : tensor<i64>
    
      %39 = stablehlo.concatenate %A, %B, %C, dim = 2 : (tensor<144x1024x8xf64>, tensor<144x1024x1008xf64>, tensor<144x1024x8xf64>) -> tensor<144x1024x1024xf64>

      %dus = stablehlo.dynamic_update_slice %39, %arg3, %c_31, %c_31, %c_31 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1024xf64>

      %dus2 = stablehlo.dynamic_update_slice %39, %arg4, %c_31, %c_31, %c_31 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1024xf64>

    func.return %dus, %dus2 : tensor<144x1024x1024xf64>, tensor<144x1024x1024xf64>
  }
}

// CHECK:  func.func @dus_concat_test(%arg0: tensor<1x1024x8xf64>, %arg1: tensor<1x1024x1008xf64>, %arg2: tensor<1x1024x8xf64>, %arg3: tensor<1x1010x1008xf64>) -> tensor<1x1024x1024xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:1, 0:7, 0:1008] : (tensor<1x1024x1008xf64>) -> tensor<1x7x1008xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:1, 1017:1024, 0:1008] : (tensor<1x1024x1008xf64>) -> tensor<1x7x1008xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %arg3, %1, dim = 1 : (tensor<1x7x1008xf64>, tensor<1x1010x1008xf64>, tensor<1x7x1008xf64>) -> tensor<1x1024x1008xf64>
// CHECK-NEXT:    %3 = stablehlo.concatenate %arg0, %2, %arg2, dim = 2 : (tensor<1x1024x8xf64>, tensor<1x1024x1008xf64>, tensor<1x1024x8xf64>) -> tensor<1x1024x1024xf64>
// CHECK-NEXT:    return %3 : tensor<1x1024x1024xf64>
// CHECK-NEXT:  }

// CHECK-NEXT:  func.func @dus_concat_test2(%arg0: tensor<144x1024x8xf64>, %arg1: tensor<144x1024x1008xf64>, %arg2: tensor<144x1024x8xf64>, %arg3: tensor<128x1008x1008xf64>, %arg4: tensor<128x1008x1008xf64>) -> (tensor<144x1024x1024xf64>, tensor<144x1024x1024xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.dynamic_update_slice %arg1, %arg3, %c, %c, %c_0 : (tensor<144x1024x1008xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
// CHECK-NEXT:    %1 = stablehlo.concatenate %arg0, %0, %arg2, dim = 2 : (tensor<144x1024x8xf64>, tensor<144x1024x1008xf64>, tensor<144x1024x8xf64>) -> tensor<144x1024x1024xf64>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %arg1, %arg4, %c, %c, %c_0 : (tensor<144x1024x1008xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
// CHECK-NEXT:    %3 = stablehlo.concatenate %arg0, %2, %arg2, dim = 2 : (tensor<144x1024x8xf64>, tensor<144x1024x1008xf64>, tensor<144x1024x8xf64>) -> tensor<144x1024x1024xf64>
// CHECK-NEXT:    return %1, %3 : tensor<144x1024x1024xf64>, tensor<144x1024x1024xf64>
// CHECK-NEXT:  }