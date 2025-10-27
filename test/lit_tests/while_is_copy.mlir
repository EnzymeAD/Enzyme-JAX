// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_dynamic_slice(1);reshape_licm(1);transpose_dynamic_slice;transpose_licm(1)" --transform-interpreter --enzyme-hlo-remove-transform --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<10> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %arg0) : tensor<i64>, tensor<10xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %3 = stablehlo.subtract %2, %c : tensor<i32>
      %4 = stablehlo.dynamic_slice %arg1, %3, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
      %5 = stablehlo.remainder %iterArg, %c_1 : tensor<i64>
      %6 = stablehlo.add %5, %c_2 : tensor<i64>
      %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
      %8 = stablehlo.subtract %7, %c : tensor<i32>
      %9 = stablehlo.dynamic_update_slice %iterArg_3, %4, %8 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
      stablehlo.return %1, %9 : tensor<i64>, tensor<10xf32>
    }
    return %0#1 : tensor<10xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-NEXT:     return %arg1 : tensor<10xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<10xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<9> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %arg0) : tensor<i64>, tensor<10xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %3 = stablehlo.subtract %2, %c : tensor<i32>
      %4 = stablehlo.dynamic_slice %arg1, %3, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
      %5 = stablehlo.remainder %iterArg, %c_1 : tensor<i64>
      %6 = stablehlo.add %5, %c_2 : tensor<i64>
      %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
      %8 = stablehlo.subtract %7, %c : tensor<i32>
      %9 = stablehlo.dynamic_update_slice %iterArg_3, %4, %8 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
      stablehlo.return %1, %9 : tensor<i64>, tensor<10xf32>
    }
    return %0#1 : tensor<10xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-NEXT:      %0 = stablehlo.slice %arg1 [0:9] : (tensor<10xf32>) -> tensor<9xf32>
// CHECK-NEXT:      %1 = stablehlo.slice %arg0 [9:10] : (tensor<10xf32>) -> tensor<1xf32>
// CHECK-NEXT:      %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<9xf32>, tensor<1xf32>) -> tensor<10xf32>
// CHECK-NEXT:      return %2 : tensor<10xf32>
// CHECK-NEXT:    }

module {
  func.func @main(%arg0: tensor<3x4x5xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<4xf32> {tf.aliasing_output = 1 : i32}, %arg2: tensor<3x4x5xf32>, %arg3: tensor<4xf32>) -> (tensor<3x4x5xf32>, tensor<4xf32>) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<4> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0:3 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg0, %iterArg_5 = %arg1) : tensor<i64>, tensor<3x4x5xf32>, tensor<4xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %3 = stablehlo.subtract %2, %c_0 : tensor<i32>
      %4 = stablehlo.dynamic_slice %arg3, %3, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
      %5 = stablehlo.remainder %iterArg, %c_2 : tensor<i64>
      %6 = stablehlo.add %5, %c_3 : tensor<i64>
      %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
      %8 = stablehlo.subtract %7, %c_0 : tensor<i32>
      %9 = stablehlo.dynamic_update_slice %iterArg_5, %4, %8 : (tensor<4xf32>, tensor<1xf32>, tensor<i32>) -> tensor<4xf32>
      %10 = stablehlo.dynamic_slice %arg2, %c, %3, %c, sizes = [3, 1, 5] : (tensor<3x4x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x1x5xf32>
      %11 = stablehlo.dynamic_update_slice %iterArg_4, %10, %c, %3, %c : (tensor<3x4x5xf32>, tensor<3x1x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3x4x5xf32>
      stablehlo.return %1, %11, %9 : tensor<i64>, tensor<3x4x5xf32>, tensor<4xf32>
    }
    return %0#1, %0#2 : tensor<3x4x5xf32>, tensor<4xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3x4x5xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<4xf32> {tf.aliasing_output = 1 : i32}, %arg2: tensor<3x4x5xf32>, %arg3: tensor<4xf32>) -> (tensor<3x4x5xf32>, tensor<4xf32>) {
// CHECK-NEXT:     return %arg2, %arg3 : tensor<3x4x5xf32>, tensor<4xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<11x9x7xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<9x6x4xf32>) -> tensor<11x9x7xf32> {
    %c = stablehlo.constant dense<2> : tensor<i32>
    %c_0 = stablehlo.constant dense<3> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %c_5 = stablehlo.constant dense<2> : tensor<i64>
    %c_6 = stablehlo.constant dense<5> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_3, %iterArg_7 = %arg0) : tensor<i64>, tensor<11x9x7xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_5, %iterArg : tensor<i64>
      %2 = stablehlo.add %iterArg, %c_4 : tensor<i64>
      %3 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_2 : tensor<i32>
      %5 = stablehlo.dynamic_slice %arg1, %c_0, %4, %c_1, sizes = [6, 1, 4] : (tensor<9x6x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x1x4xf32>
      %6 = stablehlo.dynamic_update_slice %iterArg_7, %5, %c_2, %4, %c : (tensor<11x9x7xf32>, tensor<6x1x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<11x9x7xf32>
      stablehlo.return %2, %6 : tensor<i64>, tensor<11x9x7xf32>
    }
    return %0#1 : tensor<11x9x7xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<11x9x7xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<9x6x4xf32>) -> tensor<11x9x7xf32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %0 = stablehlo.slice %arg1 [3:9, 1:6, 0:4] : (tensor<9x6x4xf32>) -> tensor<6x5x4xf32>
// CHECK-NEXT:     %1 = stablehlo.dynamic_update_slice %arg0, %0, %c_0, %c_0, %c : (tensor<11x9x7xf32>, tensor<6x5x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<11x9x7xf32>
// CHECK-NEXT:     return %1 : tensor<11x9x7xf32>
// CHECK-NEXT: }


module {
  func.func @main(%arg0: tensor<3x5x10xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x5x10xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<5> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x5x10xf32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<4x3xf32>) -> tensor<5x4x3xf32>
    %1 = stablehlo.dot_general %arg0, %0, batching_dims = [1] x [0], contracting_dims = [0] x [2], precision = [DEFAULT, DEFAULT] : (tensor<3x5x10xf32>, tensor<5x4x3xf32>) -> tensor<5x10x4xf32>
    %2:2 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %cst) : tensor<i64>, tensor<4x5x10xf32>
    cond {
      %3 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
      %5 = stablehlo.subtract %4, %c_1 : tensor<i32>
      %6 = stablehlo.dynamic_slice %1, %iterArg, %c_2, %c_2, sizes = [1, 10, 4] : (tensor<5x10x4xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x10x4xf32>
      %7 = stablehlo.transpose %6, dims = [2, 0, 1] : (tensor<1x10x4xf32>) -> tensor<4x1x10xf32>
      %8 = stablehlo.dynamic_update_slice %iterArg_4, %7, %c, %5, %c : (tensor<4x5x10xf32>, tensor<4x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x5x10xf32>
      stablehlo.return %3, %8 : tensor<i64>, tensor<4x5x10xf32>
    }
    return %2#1 : tensor<4x5x10xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3x5x10xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x5x10xf32> {
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<4x3xf32>) -> tensor<5x4x3xf32>
// CHECK-NEXT:     %1 = stablehlo.dot_general %arg0, %0, batching_dims = [1] x [0], contracting_dims = [0] x [2], precision = [DEFAULT, DEFAULT] : (tensor<3x5x10xf32>, tensor<5x4x3xf32>) -> tensor<5x10x4xf32>
// CHECK-NEXT:     %2 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<5x10x4xf32>) -> tensor<4x5x10xf32>
// CHECK-NEXT:     return %2 : tensor<4x5x10xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<10xf64>) -> tensor<10xf64> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<10> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
    %0 = stablehlo.reshape %arg0 : (tensor<10xf64>) -> tensor<10x1xf64>
    %1:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %cst) : tensor<i64>, tensor<10xf64>
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c : tensor<i32>
      %5 = stablehlo.dynamic_slice %0, %iterArg, %c_0, sizes = [1, 1] : (tensor<10x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
      %6 = stablehlo.reshape %5 : (tensor<1x1xf64>) -> tensor<1xf64>
      %7 = stablehlo.dynamic_update_slice %iterArg_3, %6, %4 : (tensor<10xf64>, tensor<1xf64>, tensor<i32>) -> tensor<10xf64>
      stablehlo.return %2, %7 : tensor<i64>, tensor<10xf64>
    }
    return %1#1 : tensor<10xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<10xf64>) -> tensor<10xf64> {
// CHECK-NEXT:   return %arg0 : tensor<10xf64>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<5x4x3xf32>) -> tensor<4x5x3xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x5x3xf32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<4> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2, 0, 3] : (tensor<5x4x3xf32>) -> tensor<4x1x5x3xf32>
    %1:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %cst) : tensor<i64>, tensor<4x5x3xf32>
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_3, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_0 : tensor<i32>
      %5 = stablehlo.dynamic_slice %0, %iterArg, %c_1, %c_1, %c_1, sizes = [1, 1, 5, 3] : (tensor<4x1x5x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x5x3xf32>
      %6 = stablehlo.reshape %5 : (tensor<1x1x5x3xf32>) -> tensor<1x5x3xf32>
      %7 = stablehlo.dynamic_update_slice %iterArg_4, %6, %4, %c, %c : (tensor<4x5x3xf32>, tensor<1x5x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x5x3xf32>
      stablehlo.return %2, %7 : tensor<i64>, tensor<4x5x3xf32>
    }
    return %1#1 : tensor<4x5x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x4x3xf32>) -> tensor<4x5x3xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<5x4x3xf32>) -> tensor<4x5x3xf32>
// CHECK-NEXT:   return %0 : tensor<4x5x3xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<5x4x3xf32>, %arg1: tensor<3x1x4x1x5xf32>) -> tensor<5x4x3xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %c_3 = stablehlo.constant dense<4> : tensor<i64>
    %0 = stablehlo.transpose %arg1, dims = [4, 1, 2, 3, 0] : (tensor<3x1x4x1x5xf32>) -> tensor<5x1x4x1x3xf32>
    %1:2 = stablehlo.while(%iterArg = %c_1, %iterArg_3 = %arg0) : tensor<i64>, tensor<5x4x3xf32>
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_0 : tensor<i32>
      %5 = stablehlo.dynamic_slice %0, %c, %c, %4, %c, %c, sizes = [5, 1, 1, 1, 3] : (tensor<5x1x4x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x1x1x3xf32>
      %6 = stablehlo.reshape %5 : (tensor<5x1x1x1x3xf32>) -> tensor<5x1x3xf32>
      %7 = stablehlo.dynamic_update_slice %iterArg_3, %6, %c, %4, %c : (tensor<5x4x3xf32>, tensor<5x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x4x3xf32>
      stablehlo.return %2, %7 : tensor<i64>, tensor<5x4x3xf32>
    }
    return %1#1 : tensor<5x4x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<5x4x3xf32>, %arg1: tensor<3x1x4x1x5xf32>) -> tensor<5x4x3xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg1, dims = [4, 1, 2, 3, 0] : (tensor<3x1x4x1x5xf32>) -> tensor<5x1x4x1x3xf32>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<5x1x4x1x3xf32>) -> tensor<5x4x3xf32>
// CHECK-NEXT:   return %1 : tensor<5x4x3xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<1x3x4x1x5xf32>, %arg1: tensor<5x4x3xf32>) -> tensor<1x3x4x1x5xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %c_3 = stablehlo.constant dense<3> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg0) : tensor<i64>, tensor<1x3x4x1x5xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %3 = stablehlo.subtract %2, %c_0 : tensor<i32>
      %4 = stablehlo.dynamic_slice %arg1, %c, %3, %c, sizes = [5, 1, 3] : (tensor<5x4x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x3xf32>
      %5 = stablehlo.reshape %4 : (tensor<5x1x3xf32>) -> tensor<5x1x1x3x1xf32>
      %6 = stablehlo.transpose %5, dims = [4, 3, 2, 1, 0] : (tensor<5x1x1x3x1xf32>) -> tensor<1x3x1x1x5xf32>
      %7 = stablehlo.dynamic_update_slice %iterArg_4, %6, %c, %c, %3, %c, %c : (tensor<1x3x4x1x5xf32>, tensor<1x3x1x1x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x3x4x1x5xf32>
      stablehlo.return %1, %7 : tensor<i64>, tensor<1x3x4x1x5xf32>
    }
    return %0#1 : tensor<1x3x4x1x5xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<1x3x4x1x5xf32>, %arg1: tensor<5x4x3xf32>) -> tensor<1x3x4x1x5xf32> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg1 [0:5, 0:3, 0:3] : (tensor<5x4x3xf32>) -> tensor<5x3x3xf32>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<5x3x3xf32>) -> tensor<5x3x1x3x1xf32>
// CHECK-NEXT:   %2 = stablehlo.transpose %1, dims = [4, 3, 2, 1, 0] : (tensor<5x3x1x3x1xf32>) -> tensor<1x3x1x3x5xf32>
// CHECK-NEXT:   %3 = stablehlo.reshape %2 : (tensor<1x3x1x3x5xf32>) -> tensor<1x3x3x1x5xf32>
// CHECK-NEXT:   %4 = stablehlo.slice %arg0 [0:1, 0:3, 3:4, 0:1, 0:5] : (tensor<1x3x4x1x5xf32>) -> tensor<1x3x1x1x5xf32>
// CHECK-NEXT:   %5 = stablehlo.concatenate %3, %4, dim = 2 : (tensor<1x3x3x1x5xf32>, tensor<1x3x1x1x5xf32>) -> tensor<1x3x4x1x5xf32>
// CHECK-NEXT:   return %5 : tensor<1x3x4x1x5xf32>
// CHECK-NEXT: }
