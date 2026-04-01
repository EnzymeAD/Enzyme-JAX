// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<2> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<4> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_6 = %cst_5) : tensor<i64>, tensor<10x5xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_4, %iterArg : tensor<i64>
      %3 = stablehlo.multiply %c_1, %2 : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_0 : tensor<i32>
      %6 = stablehlo.dynamic_slice %arg0, %c, %5, %c, sizes = [10, 1, 5] : (tensor<10x8x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<10x1x5xf32>
      %7 = stablehlo.reshape %6 : (tensor<10x1x5xf32>) -> tensor<10x5xf32>
      %8 = stablehlo.add %iterArg_6, %7 : tensor<10x5xf32>
      stablehlo.return %2, %8 : tensor<i64>, tensor<10x5xf32>
    }
    %1 = stablehlo.multiply %0#1, %cst : tensor<10x5xf32>
    return %1 : tensor<10x5xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xf32>) -> tensor<10x4x5xf32>
// CHECK-NEXT:    %1 = stablehlo.reduce(%0 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<10x4x5xf32>, tensor<f32>) -> tensor<10x5xf32>
// CHECK-NEXT:    %2 = stablehlo.add %1, %cst_0 : tensor<10x5xf32>
// CHECK-NEXT:    %3 = stablehlo.multiply %2, %cst : tensor<10x5xf32>
// CHECK-NEXT:    return %3 : tensor<10x5xf32>
// CHECK-NEXT:  }

module {
  func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<2> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<4> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_6 = %cst_5) : tensor<i64>, tensor<10x5xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_4, %iterArg : tensor<i64>
      %3 = stablehlo.multiply %c_1, %2 : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_0 : tensor<i32>
      %6 = stablehlo.dynamic_slice %arg0, %c, %5, %c, sizes = [10, 1, 5] : (tensor<10x8x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<10x1x5xf32>
      %7 = stablehlo.reshape %6 : (tensor<10x1x5xf32>) -> tensor<10x5xf32>
      %8 = stablehlo.add %7, %iterArg_6 : tensor<10x5xf32>
      stablehlo.return %2, %8 : tensor<i64>, tensor<10x5xf32>
    }
    %1 = stablehlo.multiply %0#1, %cst : tensor<10x5xf32>
    return %1 : tensor<10x5xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xf32>) -> tensor<10x4x5xf32>
// CHECK-NEXT:    %1 = stablehlo.reduce(%0 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<10x4x5xf32>, tensor<f32>) -> tensor<10x5xf32>
// CHECK-NEXT:    %2 = stablehlo.add %1, %cst_0 : tensor<10x5xf32>
// CHECK-NEXT:    %3 = stablehlo.multiply %2, %cst : tensor<10x5xf32>
// CHECK-NEXT:    return %3 : tensor<10x5xf32>
// CHECK-NEXT:  }

module {
  func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<2> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<4> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_6 = %cst_5) : tensor<i64>, tensor<10x5xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_4, %iterArg : tensor<i64>
      %3 = stablehlo.multiply %c_1, %2 : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_0 : tensor<i32>
      %6 = stablehlo.dynamic_slice %arg0, %c, %5, %c, sizes = [10, 1, 5] : (tensor<10x8x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<10x1x5xf32>
      %7 = stablehlo.reshape %6 : (tensor<10x1x5xf32>) -> tensor<10x5xf32>
      %8 = stablehlo.subtract %iterArg_6, %7 : tensor<10x5xf32>
      stablehlo.return %2, %8 : tensor<i64>, tensor<10x5xf32>
    }
    %1 = stablehlo.multiply %0#1, %cst : tensor<10x5xf32>
    return %1 : tensor<10x5xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xf32>) -> tensor<10x4x5xf32>
// CHECK-NEXT:   %1 = stablehlo.negate %0 : tensor<10x4x5xf32>
// CHECK-NEXT:   %2 = stablehlo.reduce(%1 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<10x4x5xf32>, tensor<f32>) -> tensor<10x5xf32>
// CHECK-NEXT:   %3 = stablehlo.add %2, %cst_0 : tensor<10x5xf32>
// CHECK-NEXT:   %4 = stablehlo.multiply %3, %cst : tensor<10x5xf32>
// CHECK-NEXT:   return %4 : tensor<10x5xf32>
// CHECK-NEXT: }


module {
  func.func @main_fail(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<2> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<4> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
    // CHECK: stablehlo.while
    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_6 = %cst_5) : tensor<i64>, tensor<10x5xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_4, %iterArg : tensor<i64>
      %3 = stablehlo.multiply %c_1, %2 : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_0 : tensor<i32>
      %6 = stablehlo.dynamic_slice %arg0, %c, %5, %c, sizes = [10, 1, 5] : (tensor<10x8x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<10x1x5xf32>
      %7 = stablehlo.reshape %6 : (tensor<10x1x5xf32>) -> tensor<10x5xf32>
      %8 = stablehlo.subtract %7, %iterArg_6 : tensor<10x5xf32>
      stablehlo.return %2, %8 : tensor<i64>, tensor<10x5xf32>
    }
    %1 = stablehlo.multiply %0#1, %cst : tensor<10x5xf32>
    return %1 : tensor<10x5xf32>
  }
}

module {
  func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<2> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<4> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_6 = %cst_5) : tensor<i64>, tensor<10x5xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_4, %iterArg : tensor<i64>
      %3 = stablehlo.multiply %c_1, %2 : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_0 : tensor<i32>
      %6 = stablehlo.dynamic_slice %arg0, %c, %5, %c, sizes = [10, 1, 5] : (tensor<10x8x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<10x1x5xf32>
      %7 = stablehlo.reshape %6 : (tensor<10x1x5xf32>) -> tensor<10x5xf32>
      %8 = stablehlo.minimum %7, %iterArg_6 : tensor<10x5xf32>
      stablehlo.return %2, %8 : tensor<i64>, tensor<10x5xf32>
    }
    %1 = stablehlo.multiply %0#1, %cst : tensor<10x5xf32>
    return %1 : tensor<10x5xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xf32>) -> tensor<10x4x5xf32>
// CHECK-NEXT:   %1 = stablehlo.reduce(%0 init: %cst_1) applies stablehlo.minimum across dimensions = [1] : (tensor<10x4x5xf32>, tensor<f32>) -> tensor<10x5xf32>
// CHECK-NEXT:   %2 = stablehlo.minimum %1, %cst_0 : tensor<10x5xf32>
// CHECK-NEXT:   %3 = stablehlo.multiply %2, %cst : tensor<10x5xf32>
// CHECK-NEXT:   return %3 : tensor<10x5xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<2> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<4> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_6 = %cst_5) : tensor<i64>, tensor<10x5xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_4, %iterArg : tensor<i64>
      %3 = stablehlo.multiply %c_1, %2 : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_0 : tensor<i32>
      %6 = stablehlo.dynamic_slice %arg0, %c, %5, %c, sizes = [10, 1, 5] : (tensor<10x8x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<10x1x5xf32>
      %7 = stablehlo.reshape %6 : (tensor<10x1x5xf32>) -> tensor<10x5xf32>
      %8 = stablehlo.maximum %7, %iterArg_6 : tensor<10x5xf32>
      stablehlo.return %2, %8 : tensor<i64>, tensor<10x5xf32>
    }
    %1 = stablehlo.multiply %0#1, %cst : tensor<10x5xf32>
    return %1 : tensor<10x5xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xf32>) -> tensor<10x4x5xf32>
// CHECK-NEXT:   %1 = stablehlo.reduce(%0 init: %cst_1) applies stablehlo.maximum across dimensions = [1] : (tensor<10x4x5xf32>, tensor<f32>) -> tensor<10x5xf32>
// CHECK-NEXT:   %2 = stablehlo.maximum %1, %cst_0 : tensor<10x5xf32>
// CHECK-NEXT:   %3 = stablehlo.multiply %2, %cst : tensor<10x5xf32>
// CHECK-NEXT:   return %3 : tensor<10x5xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<2> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<4> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_6 = %cst_5) : tensor<i64>, tensor<10x5xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_4, %iterArg : tensor<i64>
      %3 = stablehlo.multiply %c_1, %2 : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_0 : tensor<i32>
      %6 = stablehlo.dynamic_slice %arg0, %c, %5, %c, sizes = [10, 1, 5] : (tensor<10x8x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<10x1x5xf32>
      %7 = stablehlo.reshape %6 : (tensor<10x1x5xf32>) -> tensor<10x5xf32>
      %8 = stablehlo.multiply %7, %iterArg_6 : tensor<10x5xf32>
      stablehlo.return %2, %8 : tensor<i64>, tensor<10x5xf32>
    }
    %1 = stablehlo.multiply %0#1, %cst : tensor<10x5xf32>
    return %1 : tensor<10x5xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %cst = stablehlo.constant dense<2.500000e-01> : tensor<10x5xf32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xf32>) -> tensor<10x4x5xf32>
// CHECK-NEXT:   %1 = stablehlo.reduce(%0 init: %cst_0) applies stablehlo.multiply across dimensions = [1] : (tensor<10x4x5xf32>, tensor<f32>) -> tensor<10x5xf32>
// CHECK-NEXT:   %2 = stablehlo.multiply %1, %cst : tensor<10x5xf32>
// CHECK-NEXT:   return %2 : tensor<10x5xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<10x5xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<2> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<4> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<10x5xf32>
    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_6 = %cst_5) : tensor<i64>, tensor<10x5xf32> attributes {enzyme.disable_mincut}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_4, %iterArg : tensor<i64>
      %3 = stablehlo.multiply %c_1, %2 : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_0 : tensor<i32>
      %6 = stablehlo.dynamic_slice %arg0, %c, %5, %c, sizes = [10, 1, 5] : (tensor<10x8x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<10x1x5xf32>
      %7 = stablehlo.reshape %6 : (tensor<10x1x5xf32>) -> tensor<10x5xf32>
      %8 = stablehlo.divide %iterArg_6, %7 : tensor<10x5xf32>
      stablehlo.return %2, %8 : tensor<i64>, tensor<10x5xf32>
    }
    %1 = stablehlo.multiply %0#1, %cst : tensor<10x5xf32>
    return %1 : tensor<10x5xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10x8x5xf32> {enzymexla.memory_effects = []}) -> tensor<10x5xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %cst = stablehlo.constant dense<2.500000e-01> : tensor<10x5xf32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<10x4x5xf32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xf32>) -> tensor<10x4x5xf32>
// CHECK-NEXT:   %1 = stablehlo.divide %cst_1, %0 : tensor<10x4x5xf32>
// CHECK-NEXT:   %2 = stablehlo.reduce(%1 init: %cst_0) applies stablehlo.multiply across dimensions = [1] : (tensor<10x4x5xf32>, tensor<f32>) -> tensor<10x5xf32>
// CHECK-NEXT:   %3 = stablehlo.multiply %2, %cst : tensor<10x5xf32>
// CHECK-NEXT:   return %3 : tensor<10x5xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<10x8x5xi32> {enzymexla.memory_effects = []}) -> tensor<10x5xi32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<2> : tensor<10x5xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<4> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xi32>) -> tensor<10x4x5xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 0, 3] : (tensor<10x4x5xi32>) -> tensor<4x10x1x5xi32>
    %2:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %c) : tensor<i64>, tensor<10x5xi32> attributes {enzyme.disable_mincut}
    cond {
      %3 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %4 = stablehlo.dynamic_slice %1, %iterArg, %c_0, %c_0, %c_0, sizes = [1, 10, 1, 5] : (tensor<4x10x1x5xi32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x10x1x5xi32>
      %5 = stablehlo.reshape %4 : (tensor<1x10x1x5xi32>) -> tensor<10x5xi32>
      %6 = stablehlo.and %iterArg_3, %5 : tensor<10x5xi32>
      stablehlo.return %3, %6 : tensor<i64>, tensor<10x5xi32>
    }
    return %2#1 : tensor<10x5xi32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10x8x5xi32> {enzymexla.memory_effects = []}) -> tensor<10x5xi32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %c = stablehlo.constant dense<2> : tensor<10x5xi32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<-1> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xi32>) -> tensor<10x4x5xi32>
// CHECK-NEXT:   %1 = stablehlo.reduce(%0 init: %c_0) applies stablehlo.and across dimensions = [1] : (tensor<10x4x5xi32>, tensor<i32>) -> tensor<10x5xi32>
// CHECK-NEXT:   %2 = stablehlo.and %1, %c : tensor<10x5xi32>
// CHECK-NEXT:   return %2 : tensor<10x5xi32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<10x8x5xi32> {enzymexla.memory_effects = []}) -> tensor<10x5xi32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<2> : tensor<10x5xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<4> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xi32>) -> tensor<10x4x5xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 0, 3] : (tensor<10x4x5xi32>) -> tensor<4x10x1x5xi32>
    %2:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %c) : tensor<i64>, tensor<10x5xi32> attributes {enzyme.disable_mincut}
    cond {
      %3 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %4 = stablehlo.dynamic_slice %1, %iterArg, %c_0, %c_0, %c_0, sizes = [1, 10, 1, 5] : (tensor<4x10x1x5xi32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x10x1x5xi32>
      %5 = stablehlo.reshape %4 : (tensor<1x10x1x5xi32>) -> tensor<10x5xi32>
      %6 = stablehlo.or %iterArg_3, %5 : tensor<10x5xi32>
      stablehlo.return %3, %6 : tensor<i64>, tensor<10x5xi32>
    }
    return %2#1 : tensor<10x5xi32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10x8x5xi32> {enzymexla.memory_effects = []}) -> tensor<10x5xi32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %c = stablehlo.constant dense<2> : tensor<10x5xi32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xi32>) -> tensor<10x4x5xi32>
// CHECK-NEXT:   %1 = stablehlo.reduce(%0 init: %c_0) applies stablehlo.or across dimensions = [1] : (tensor<10x4x5xi32>, tensor<i32>) -> tensor<10x5xi32>
// CHECK-NEXT:   %2 = stablehlo.or %1, %c : tensor<10x5xi32>
// CHECK-NEXT:   return %2 : tensor<10x5xi32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<10x8x5xi32> {enzymexla.memory_effects = []}) -> tensor<10x5xi32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<4> : tensor<i64>
    %c_0 = stablehlo.constant dense<2> : tensor<10x5xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xi32>) -> tensor<10x4x5xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 0, 3] : (tensor<10x4x5xi32>) -> tensor<4x10x1x5xi32>
    %2:2 = stablehlo.while(%iterArg = %c_1, %iterArg_3 = %c_0) : tensor<i64>, tensor<10x5xi32> attributes {enzyme.disable_mincut}
    cond {
      %3 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %4 = stablehlo.dynamic_slice %1, %iterArg, %c_1, %c_1, %c_1, sizes = [1, 10, 1, 5] : (tensor<4x10x1x5xi32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x10x1x5xi32>
      %5 = stablehlo.reshape %4 : (tensor<1x10x1x5xi32>) -> tensor<10x5xi32>
      %6 = stablehlo.xor %iterArg_3, %5 : tensor<10x5xi32>
      stablehlo.return %3, %6 : tensor<i64>, tensor<10x5xi32>
    }
    return %2#1 : tensor<10x5xi32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10x8x5xi32> {enzymexla.memory_effects = []}) -> tensor<10x5xi32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:   %c = stablehlo.constant dense<2> : tensor<10x5xi32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:10, 1:8:2, 0:5] : (tensor<10x8x5xi32>) -> tensor<10x4x5xi32>
// CHECK-NEXT:   %1 = stablehlo.reduce(%0 init: %c_0) applies stablehlo.xor across dimensions = [1] : (tensor<10x4x5xi32>, tensor<i32>) -> tensor<10x5xi32>
// CHECK-NEXT:   %2 = stablehlo.xor %1, %c : tensor<10x5xi32>
// CHECK-NEXT:   return %2 : tensor<10x5xi32>
// CHECK-NEXT: }
