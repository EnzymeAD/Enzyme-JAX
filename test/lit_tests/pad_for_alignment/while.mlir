// RUN: enzymexlamlir-opt --pad-for-alignment --canonicalize --allow-unregistered-dialect %s | FileCheck %s

func.func @test_while(%init_i: tensor<i64>, %max_i: tensor<i64>, %init_sum: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0:2 = stablehlo.while(%i = %init_i, %sum = %init_sum) : tensor<i64>, tensor<4x760x1533xf32>
  cond {
    %cmp = stablehlo.compare LT, %i, %max_i : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %cst = stablehlo.constant dense<1> : tensor<i64>
    %next_i = stablehlo.add %i, %cst : tensor<i64>
    %next_sum = stablehlo.add %sum, %sum : tensor<4x760x1533xf32>
    stablehlo.return %next_i, %next_sum : tensor<i64>, tensor<4x760x1533xf32>
  }
  return %0#1 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_while(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.pad %arg2, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:   %1:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<i64>, tensor<4x768x1536xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %3 = stablehlo.compare LT, %iterArg, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %3 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %3 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:     %4 = stablehlo.add %iterArg_0, %iterArg_0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     stablehlo.return %3, %4 : tensor<i64>, tensor<4x768x1536xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %2 = stablehlo.slice %1#1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:   return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_while_no_pad(%init_i: tensor<i64>, %max_i: tensor<i64>, %init_sum: tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32> {
  %0:2 = stablehlo.while(%i = %init_i, %sum = %init_sum) : tensor<i64>, tensor<4x768x1536xf32>
  cond {
    %cmp = stablehlo.compare LT, %i, %max_i : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %cst = stablehlo.constant dense<1> : tensor<i64>
    %next_i = stablehlo.add %i, %cst : tensor<i64>
    %next_sum = stablehlo.add %sum, %sum : tensor<4x768x1536xf32>
    stablehlo.return %next_i, %next_sum : tensor<i64>, tensor<4x768x1536xf32>
  }
  return %0#1 : tensor<4x768x1536xf32>
}

// CHECK: func.func @test_while_no_pad(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %arg2) : tensor<i64>, tensor<4x768x1536xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %1 = stablehlo.compare LT, %iterArg, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:       %2 = stablehlo.add %iterArg_0, %iterArg_0 : tensor<4x768x1536xf32>
// CHECK-NEXT:       stablehlo.return %1, %2 : tensor<i64>, tensor<4x768x1536xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1 : tensor<4x768x1536xf32>
// CHECK-NEXT: }

func.func @test_while_mixed(%init_i: tensor<i64>, %max_i: tensor<i64>, %x: tensor<4x760x1533xf32>, %y: tensor<4x768x1536xf32>) -> (tensor<4x760x1533xf32>, tensor<4x768x1536xf32>) {
  %0:3 = stablehlo.while(%i = %init_i, %xit = %x, %yit = %y) : tensor<i64>, tensor<4x760x1533xf32>, tensor<4x768x1536xf32>
  cond {
    %cmp = stablehlo.compare LT, %i, %max_i : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %cst = stablehlo.constant dense<1> : tensor<i64>
    %next_i = stablehlo.add %i, %cst : tensor<i64>
    %next_x = stablehlo.add %xit, %xit : tensor<4x760x1533xf32>
    %next_y = stablehlo.add %yit, %yit : tensor<4x768x1536xf32>
    stablehlo.return %next_i, %next_x, %next_y : tensor<i64>, tensor<4x760x1533xf32>, tensor<4x768x1536xf32>
  }
  return %0#1, %0#2 : tensor<4x760x1533xf32>, tensor<4x768x1536xf32>
}

// CHECK: func.func @test_while_mixed(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<4x760x1533xf32>, %arg3: tensor<4x768x1536xf32>) -> (tensor<4x760x1533xf32>, tensor<4x768x1536xf32>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg2, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1:3 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0, %iterArg_1 = %arg3) : tensor<i64>, tensor<4x768x1536xf32>, tensor<4x768x1536xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %3 = stablehlo.compare LT, %iterArg, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %3 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %3 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:       %4 = stablehlo.add %iterArg_0, %iterArg_0 : tensor<4x768x1536xf32>
// CHECK-NEXT:       %5 = stablehlo.add %iterArg_1, %iterArg_1 : tensor<4x768x1536xf32>
// CHECK-NEXT:       stablehlo.return %3, %4, %5 : tensor<i64>, tensor<4x768x1536xf32>, tensor<4x768x1536xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = stablehlo.slice %1#1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2, %1#2 : tensor<4x760x1533xf32>, tensor<4x768x1536xf32>
// CHECK-NEXT: }
