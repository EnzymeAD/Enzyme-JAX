// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

module {
  func.func @fuse(%arg0: tensor<24x34x59xf64>, %arg1: tensor<1x20x45xf64>, %arg2: tensor<1x20x45xf64>) -> tensor<24x34x59xf64> {  
    %c_200 = stablehlo.constant dense<10> : tensor<i64>
    %c_201 = stablehlo.constant dense<11> : tensor<i64>
    %c_202 = stablehlo.constant dense<12> : tensor<i64>
    %c_214 = stablehlo.constant dense<7> : tensor<i64>

    %422 = stablehlo.dynamic_update_slice %arg0, %arg1, %c_200, %c_214, %c_214 : (tensor<24x34x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
    %446 = stablehlo.dynamic_update_slice %422, %arg2, %c_201, %c_214, %c_214 : (tensor<24x34x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
    %447 = stablehlo.dynamic_update_slice %446, %arg2, %c_202, %c_214, %c_214 : (tensor<24x34x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
    func.return %447 : tensor<24x34x59xf64>
  }
}

module {
  func.func @fuse(%arg0: tensor<24x34x59xf64>, %arg1: tensor<1x20x45xf64>, %arg2: tensor<1x20x45xf64>) -> tensor<24x34x59xf64> {  
    %c_200 = stablehlo.constant dense<12> : tensor<i64>
    %c_201 = stablehlo.constant dense<11> : tensor<i64>
    %c_214 = stablehlo.constant dense<7> : tensor<i64>

    %422 = stablehlo.dynamic_update_slice %arg0, %arg1, %c_200, %c_214, %c_214 : (tensor<24x34x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
    %446 = stablehlo.dynamic_update_slice %422, %arg2, %c_201, %c_214, %c_214 : (tensor<24x34x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
    func.return %446 : tensor<24x34x59xf64>
  }
}

module {
  func.func @fuse(%arg0: tensor<24x34x59xf64>, %arg1: tensor<1x20x45xf64>, %arg2: tensor<2x20x45xf64>) -> tensor<24x34x59xf64> {  
    %c_200 = stablehlo.constant dense<13> : tensor<i64>
    %c_201 = stablehlo.constant dense<11> : tensor<i64>
    %c_214 = stablehlo.constant dense<7> : tensor<i64>

    %422 = stablehlo.dynamic_update_slice %arg0, %arg1, %c_200, %c_214, %c_214 : (tensor<24x34x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
    %446 = stablehlo.dynamic_update_slice %422, %arg2, %c_201, %c_214, %c_214 : (tensor<24x34x59xf64>, tensor<2x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x34x59xf64>
    func.return %446 : tensor<24x34x59xf64>
  }
}

// CHECK: func.func @chlo_prop_mix() -> tensor<2xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<[false, true]> : tensor<2xi1>
// CHECK-NEXT:   return %c : tensor<2xi1>
// CHECK-NEXT: }

// CHECK: func.func @chlo_prop_false_splat() -> tensor<2xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<false> : tensor<2xi1>
// CHECK-NEXT:   return %c : tensor<2xi1>
// CHECK-NEXT: }

// CHECK: func.func @chlo_prop_true_splat() -> tensor<2xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:   return %c : tensor<2xi1>
// CHECK-NEXT: }
