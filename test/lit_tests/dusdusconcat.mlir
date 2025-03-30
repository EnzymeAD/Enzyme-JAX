// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

module {
  func.func @fuse(%op0: tensor<24x34x59xf64>, %up0: tensor<1x22x47xf64>, %up1: tensor<1x22x47xf64>, %up2: tensor<1x22x47xf64>, %i2: tensor<i64>, %i3: tensor<i64>) -> tensor<24x34x59xf64> {  

    %c_221 = stablehlo.constant dense<9> : tensor<i64>
    %c_215 = stablehlo.constant dense<8> : tensor<i64>
    %c_214 = stablehlo.constant dense<7> : tensor<i64>

      %op1 = stablehlo.dynamic_update_slice %op0, %up0, %c_221, %i2, %i3 : (tensor<24x34x59xf64>, tensor<1x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>


      %op2 = stablehlo.dynamic_update_slice %op1, %up1, %c_215, %i2, %i3 : (tensor<24x34x59xf64>, tensor<1x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>

      %op3 = stablehlo.dynamic_update_slice %op2, %up2, %c_214, %i2, %i3 : (tensor<24x34x59xf64>, tensor<1x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>

      func.return %op3 : tensor<24x34x59xf64>
  }
  func.func @fuse2(%op0: tensor<24x34x59xf64>, %up0: tensor<1x22x47xf64>, %up1: tensor<1x22x47xf64>, %up2: tensor<1x22x47xf64>, %i2: tensor<i64>, %i3: tensor<i64>) -> tensor<24x34x59xf64> {  

    %c9 = stablehlo.constant dense<9> : tensor<i64>
    %c8 = stablehlo.constant dense<8> : tensor<i64>
    %c7 = stablehlo.constant dense<7> : tensor<i64>

      %op1 = stablehlo.dynamic_update_slice %op0, %up0, %c7, %i2, %i3 : (tensor<24x34x59xf64>, tensor<1x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>


      %op2 = stablehlo.dynamic_update_slice %op1, %up1, %c8, %i2, %i3 : (tensor<24x34x59xf64>, tensor<1x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>

      %op3 = stablehlo.dynamic_update_slice %op2, %up2, %c9, %i2, %i3 : (tensor<24x34x59xf64>, tensor<1x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
      func.return %op3 : tensor<24x34x59xf64>
  }
     
  func.func @fuse3(%op0: tensor<24x34x59xf64>, %up0: tensor<1x22x47xf64>, %up1: tensor<1x22x47xf64>, %up2: tensor<1x22x47xf64>, %i2: tensor<i64>, %i3: tensor<i64>) -> tensor<24x34x59xf64> {  

    %c10 = stablehlo.constant dense<10> : tensor<i64>
    %c8 = stablehlo.constant dense<8> : tensor<i64>
    %c7 = stablehlo.constant dense<7> : tensor<i64>

      %op1 = stablehlo.dynamic_update_slice %op0, %up0, %c7, %i2, %i3 : (tensor<24x34x59xf64>, tensor<1x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>


      %op2 = stablehlo.dynamic_update_slice %op1, %up1, %c8, %i2, %i3 : (tensor<24x34x59xf64>, tensor<1x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>

      %op3 = stablehlo.dynamic_update_slice %op2, %up2, %c10, %i2, %i3 : (tensor<24x34x59xf64>, tensor<1x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>

      func.return %op3 : tensor<24x34x59xf64>
  }

  func.func @fuse4(%51: tensor<1x34x59xf64>, %243: tensor<1x20x45xf64>, %282: tensor<1x22x45xf64>) -> tensor<1x34x59xf64> {  
    %c_221 = stablehlo.constant dense<0> : tensor<i64>
    %c_211 = stablehlo.constant dense<7> : tensor<i64>
    %c_214 = stablehlo.constant dense<6> : tensor<i64>


        %244 = stablehlo.dynamic_update_slice %51, %243, %c_221, %c_211, %c_211 : (tensor<1x34x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x34x59xf64>

      %283 = stablehlo.dynamic_update_slice %244, %282, %c_221, %c_214, %c_211 : (tensor<1x34x59xf64>, tensor<1x22x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x34x59xf64>

      func.return %283 : tensor<1x34x59xf64>
  }

  func.func @fuse5(%292: tensor<24x34x59xf64>, %300 : tensor<10x20x45xf64>, %312 : tensor<1x20x45xf64>) -> tensor<24x34x59xf64> {  
      %c_211 = stablehlo.constant dense<7> : tensor<i64>

      %301 = stablehlo.dynamic_update_slice %292, %300, %c_211, %c_211, %c_211 : (tensor<24x34x59xf64>, tensor<10x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>

      %313 = stablehlo.dynamic_update_slice %301, %312, %c_211, %c_211, %c_211 : (tensor<24x34x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>

      func.return %313 : tensor<24x34x59xf64>
  }
}


// CHECK:  func.func @fuse(%arg0: tensor<24x34x59xf64>, %arg1: tensor<1x22x47xf64>, %arg2: tensor<1x22x47xf64>, %arg3: tensor<1x22x47xf64>, %arg4: tensor<i64>, %arg5: tensor<i64>) -> tensor<24x34x59xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg3, %arg2, %arg1, dim = 0 : (tensor<1x22x47xf64>, tensor<1x22x47xf64>, tensor<1x22x47xf64>) -> tensor<3x22x47xf64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %arg0, %0, %c, %arg4, %arg5 : (tensor<24x34x59xf64>, tensor<3x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
// CHECK-NEXT:    return %1 : tensor<24x34x59xf64>
// CHECK-NEXT:  }

// CHECK:  func.func @fuse2(%arg0: tensor<24x34x59xf64>, %arg1: tensor<1x22x47xf64>, %arg2: tensor<1x22x47xf64>, %arg3: tensor<1x22x47xf64>, %arg4: tensor<i64>, %arg5: tensor<i64>) -> tensor<24x34x59xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg1, %arg2, %arg3, dim = 0 : (tensor<1x22x47xf64>, tensor<1x22x47xf64>, tensor<1x22x47xf64>) -> tensor<3x22x47xf64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %arg0, %0, %c, %arg4, %arg5 : (tensor<24x34x59xf64>, tensor<3x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
// CHECK-NEXT:    return %1 : tensor<24x34x59xf64>
// CHECK-NEXT:  }

// CHECK:  func.func @fuse3(%arg0: tensor<24x34x59xf64>, %arg1: tensor<1x22x47xf64>, %arg2: tensor<1x22x47xf64>, %arg3: tensor<1x22x47xf64>, %arg4: tensor<i64>, %arg5: tensor<i64>) -> tensor<24x34x59xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg1, %arg2, dim = 0 : (tensor<1x22x47xf64>, tensor<1x22x47xf64>) -> tensor<2x22x47xf64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %arg0, %0, %c_0, %arg4, %arg5 : (tensor<24x34x59xf64>, tensor<2x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %1, %arg3, %c, %arg4, %arg5 : (tensor<24x34x59xf64>, tensor<1x22x47xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
// CHECK-NEXT:    return %2 : tensor<24x34x59xf64>
// CHECK-NEXT:  }

// CHECK:  func.func @fuse4(%arg0: tensor<1x34x59xf64>, %arg1: tensor<1x20x45xf64>, %arg2: tensor<1x22x45xf64>) -> tensor<1x34x59xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.dynamic_update_slice %arg0, %arg2, %c, %c_1, %c_0 : (tensor<1x34x59xf64>, tensor<1x22x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x34x59xf64>
// CHECK-NEXT:    return %0 : tensor<1x34x59xf64>
// CHECK-NEXT:  }

// CHECK:  func.func @fuse5(%arg0: tensor<24x34x59xf64>, %arg1: tensor<10x20x45xf64>, %arg2: tensor<1x20x45xf64>) -> tensor<24x34x59xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [1:10, 0:20, 0:45] : (tensor<10x20x45xf64>) -> tensor<9x20x45xf64>
// CHECK-NEXT:    %1 = stablehlo.concatenate %arg2, %0, dim = 0 : (tensor<1x20x45xf64>, tensor<9x20x45xf64>) -> tensor<10x20x45xf64>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %arg0, %1, %c, %c, %c : (tensor<24x34x59xf64>, tensor<10x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
// CHECK-NEXT:    return %2 : tensor<24x34x59xf64>
// CHECK-NEXT:  }