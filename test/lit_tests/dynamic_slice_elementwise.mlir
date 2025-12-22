// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=dynamic_slice_elementwise<1>},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

func.func @main1(%arg0: tensor<16xf64>, %arg1: tensor<i32>) -> (tensor<5xf32>) {
  %0 = stablehlo.convert %arg0 : (tensor<16xf64>) -> tensor<16xf32>
  %1 = stablehlo.dynamic_slice %0, %arg1, sizes = [5] : (tensor<16xf32>, tensor<i32>) -> tensor<5xf32>
  return %1 : tensor<5xf32>
}

// CHECK: func.func @main1(%arg0: tensor<16xf64>, %arg1: tensor<i32>) -> tensor<5xf32> {
// CHECK-NEXT:   %0 = stablehlo.dynamic_slice %arg0, %arg1, sizes = [5] : (tensor<16xf64>, tensor<i32>) -> tensor<5xf64>
// CHECK-NEXT:   %1 = stablehlo.convert %0 : (tensor<5xf64>) -> tensor<5xf32>
// CHECK-NEXT:   return %1 : tensor<5xf32>
// CHECK-NEXT: }


func.func @main2(%arg0: tensor<16xf64>, %arg1: tensor<i32>, %arg2: tensor<16xf64>) -> (tensor<5xf64>) {
  %0 = stablehlo.add %arg0, %arg2 : tensor<16xf64>
  %1 = stablehlo.dynamic_slice %0, %arg1, sizes = [5] : (tensor<16xf64>, tensor<i32>) -> tensor<5xf64>
  return %1 : tensor<5xf64>
}

// CHECK: func.func @main2(%arg0: tensor<16xf64>, %arg1: tensor<i32>, %arg2: tensor<16xf64>) -> tensor<5xf64> {
// CHECK-NEXT:   %0 = stablehlo.dynamic_slice %arg0, %arg1, sizes = [5] : (tensor<16xf64>, tensor<i32>) -> tensor<5xf64>
// CHECK-NEXT:   %1 = stablehlo.dynamic_slice %arg2, %arg1, sizes = [5] : (tensor<16xf64>, tensor<i32>) -> tensor<5xf64>
// CHECK-NEXT:   %2 = stablehlo.add %0, %1 : tensor<5xf64>
// CHECK-NEXT:   return %2 : tensor<5xf64>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<16x32xf64>, %arg1: tensor<i32>, %arg2: tensor<16x32xf64>) -> (tensor<5x18xf64>, tensor<10x20xf64>) {
  %c5 = stablehlo.constant dense<5> : tensor<i32>
  %c7 = stablehlo.constant dense<7> : tensor<i32>
  %0 = stablehlo.add %arg0, %arg2 : tensor<16x32xf64>
  %1 = stablehlo.dynamic_slice %0, %arg1, %c5, sizes = [5, 18] : (tensor<16x32xf64>, tensor<i32>, tensor<i32>) -> tensor<5x18xf64>
  %2 = stablehlo.dynamic_slice %0, %arg1, %c7, sizes = [10, 20] : (tensor<16x32xf64>, tensor<i32>, tensor<i32>) -> tensor<10x20xf64>
  return %1, %2 : tensor<5x18xf64>, tensor<10x20xf64>
}

// CHECK: func.func @main3(%arg0: tensor<16x32xf64>, %arg1: tensor<i32>, %arg2: tensor<16x32xf64>) -> (tensor<5x18xf64>, tensor<10x20xf64>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<5> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, sizes = [10, 22] : (tensor<16x32xf64>, tensor<i32>, tensor<i32>) -> tensor<10x22xf64>
// CHECK-NEXT:   %1 = stablehlo.dynamic_slice %arg2, %arg1, %c, sizes = [10, 22] : (tensor<16x32xf64>, tensor<i32>, tensor<i32>) -> tensor<10x22xf64>
// CHECK-NEXT:   %2 = stablehlo.add %0, %1 : tensor<10x22xf64>
// CHECK-NEXT:   %3 = stablehlo.slice %2 [0:10, 2:22] : (tensor<10x22xf64>) -> tensor<10x20xf64>
// CHECK-NEXT:   %4 = stablehlo.slice %2 [0:5, 0:18] : (tensor<10x22xf64>) -> tensor<5x18xf64>
// CHECK-NEXT:   return %4, %3 : tensor<5x18xf64>, tensor<10x20xf64>
// CHECK-NEXT: }

func.func @main4_fail(%arg0: tensor<16x32xf64>, %arg1: tensor<i32>, %arg2: tensor<16x32xf64>) -> (tensor<5x18xf64>, tensor<10x20xf64>) {
  %c5 = stablehlo.constant dense<5> : tensor<i32>
  %c7 = stablehlo.constant dense<7> : tensor<i32>
  %0 = stablehlo.add %arg0, %arg2 : tensor<16x32xf64>
  // CHECK: stablehlo.dynamic_slice %0, %arg1, %c, sizes = [5, 18]
  %1 = stablehlo.dynamic_slice %0, %arg1, %c5, sizes = [5, 18] : (tensor<16x32xf64>, tensor<i32>, tensor<i32>) -> tensor<5x18xf64>
  // CHECK: stablehlo.dynamic_slice %0, %c_0, %arg1, sizes = [10, 20]
  %2 = stablehlo.dynamic_slice %0, %c7, %arg1, sizes = [10, 20] : (tensor<16x32xf64>, tensor<i32>, tensor<i32>) -> tensor<10x20xf64>
  return %1, %2 : tensor<5x18xf64>, tensor<10x20xf64>
}

