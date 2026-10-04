// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reduce_const_prop" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A loop bound the raiser makes of the largest per-lane trip count, once the
// lanes' counts are all the same constant.
func.func @splat_max() -> tensor<i32> {
  %c = stablehlo.constant dense<1> : tensor<24xi32>
  %init = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.reduce(%c init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<24xi32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK:  func.func @splat_max() -> tensor<i32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    return %c : tensor<i32>
// CHECK-NEXT:  }

func.func @sum_rows() -> tensor<2xi64> {
  %c = stablehlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
  %init = stablehlo.constant dense<10> : tensor<i64>
  %0 = stablehlo.reduce(%c init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x3xi64>, tensor<i64>) -> tensor<2xi64>
  return %0 : tensor<2xi64>
}

// CHECK:  func.func @sum_rows() -> tensor<2xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[16, 25]> : tensor<2xi64>
// CHECK-NEXT:    return %c : tensor<2xi64>
// CHECK-NEXT:  }

func.func @max_cols() -> tensor<3xf32> {
  %c = stablehlo.constant dense<[[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]> : tensor<2x3xf32>
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = stablehlo.reduce(%c init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK:  func.func @max_cols() -> tensor<3xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf32>
// CHECK-NEXT:    return %cst : tensor<3xf32>
// CHECK-NEXT:  }

// A body that is not one of the folded ones stays.
func.func @other_body() -> tensor<i32> {
  %c = stablehlo.constant dense<1> : tensor<4xi32>
  %init = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.reduce(%c init: %init) across dimensions = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
   reducer(%a: tensor<i32>, %b: tensor<i32>) {
    %s = stablehlo.subtract %a, %b : tensor<i32>
    stablehlo.return %s : tensor<i32>
  }
  return %0 : tensor<i32>
}

// CHECK:  func.func @other_body() -> tensor<i32> {
// CHECK:    stablehlo.reduce

// A splat input folds in closed form, whatever its size: a sum is the value
// times the number of elements reduced (2621441 is 0x4A200004), a product its
// power.
func.func @splat_sum() -> tensor<4xf32> {
  %c = stablehlo.constant dense<2.500000e+00> : tensor<4x1048576xf32>
  %init = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%c init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x1048576xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK:  func.func @splat_sum() -> tensor<4xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0x4A200004> : tensor<4xf32>
// CHECK-NEXT:    return %cst : tensor<4xf32>
// CHECK-NEXT:  }

func.func @splat_product() -> tensor<i32> {
  %c = stablehlo.constant dense<3> : tensor<1048576x2xi32>
  %init = stablehlo.constant dense<1> : tensor<i32>
  %0 = stablehlo.reduce(%c init: %init) applies stablehlo.multiply across dimensions = [0, 1] : (tensor<1048576x2xi32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// 3^2097152 mod 2^32
// CHECK:  func.func @splat_product() -> tensor<i32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<511705089> : tensor<i32>
// CHECK-NEXT:    return %c : tensor<i32>
// CHECK-NEXT:  }

func.func @splat_sum_i8() -> tensor<i8> {
  %c = stablehlo.constant dense<3> : tensor<100xi8>
  %init = stablehlo.constant dense<7> : tensor<i8>
  %0 = stablehlo.reduce(%c init: %init) applies stablehlo.add across dimensions = [0] : (tensor<100xi8>, tensor<i8>) -> tensor<i8>
  return %0 : tensor<i8>
}

// 7 + 300 wraps to 51
// CHECK:  func.func @splat_sum_i8() -> tensor<i8> {
// CHECK-NEXT:    %c = stablehlo.constant dense<51> : tensor<i8>
// CHECK-NEXT:    return %c : tensor<i8>
// CHECK-NEXT:  }

// The bound the raiser makes of a masked loop: the largest of the lanes' trip
// counts, each a function of the lane (an iota) and constants, is the count
// at an end of the lanes.
func.func @iota_bound() -> tensor<i32> {
  %c1 = stablehlo.constant dense<1> : tensor<1024xi32>
  %c0 = stablehlo.constant dense<0> : tensor<1024xi32>
  %c16 = stablehlo.constant dense<16> : tensor<1024xi32>
  %c3000 = stablehlo.constant dense<3000> : tensor<1024xi32>
  %init = stablehlo.constant dense<0> : tensor<i32>
  %lane = stablehlo.iota dim = 0 : tensor<1024xi32>
  // trip = max(0, (3000 - lane + 15) / 16)
  %a = stablehlo.subtract %c3000, %lane : tensor<1024xi32>
  %b = stablehlo.add %a, %c16 : tensor<1024xi32>
  %c = stablehlo.subtract %b, %c1 : tensor<1024xi32>
  %d = stablehlo.divide %c, %c16 : tensor<1024xi32>
  %e = stablehlo.maximum %d, %c0 : tensor<1024xi32>
  %0 = stablehlo.reduce(%e init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<1024xi32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK:  func.func @iota_bound() -> tensor<i32> {
// CHECK-NEXT:  %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:  %c_0 = stablehlo.constant dense<16> : tensor<i32>
// CHECK-NEXT:  %c_1 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:  %c_2 = stablehlo.constant dense<3000> : tensor<i32>
// CHECK-NEXT:  %0 = stablehlo.subtract %c_2, %c_1 : tensor<i32>
// CHECK-NEXT:  %1 = stablehlo.add %0, %c_0 : tensor<i32>
// CHECK-NEXT:  %2 = stablehlo.subtract %1, %c : tensor<i32>
// CHECK-NEXT:  %3 = stablehlo.divide %2, %c_0 : tensor<i32>
// CHECK-NEXT:  %4 = stablehlo.maximum %3, %c_1 : tensor<i32>
// CHECK-NEXT:  %5 = stablehlo.maximum %c_1, %4 : tensor<i32>
// CHECK-NEXT:  return %5 : tensor<i32>
// CHECK-NEXT:  }

// The lanes' trips in the sum-factorized kernels, (max(c, x) + k - x + k') / k:
// (max(c, x) + k) - x is max(c - x, 0) + k, largest at the smallest x. The
// trip at the first lane, 2.
func.func @lane_trip() -> tensor<i32> {
  %c40 = stablehlo.constant dense<40> : tensor<38088xi32>
  %c41 = stablehlo.constant dense<41> : tensor<38088xi32>
  %c43 = stablehlo.constant dense<43> : tensor<38088xi32>
  %init = stablehlo.constant dense<0> : tensor<i32>
  %lane = stablehlo.iota dim = 0 : tensor<38088xi32>
  %a = stablehlo.add %lane, %c41 : tensor<38088xi32>
  %b = stablehlo.maximum %c40, %a : tensor<38088xi32>
  %c = stablehlo.add %b, %c41 : tensor<38088xi32>
  %d = stablehlo.subtract %c, %a : tensor<38088xi32>
  %e = stablehlo.add %d, %c43 : tensor<38088xi32>
  %f = stablehlo.divide %e, %c41 : tensor<38088xi32>
  %0 = stablehlo.reduce(%f init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<38088xi32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK:  func.func @lane_trip() -> tensor<i32> {
// CHECK-NEXT:  %c = stablehlo.constant dense<43> : tensor<i32>
// CHECK-NEXT:  %c_0 = stablehlo.constant dense<40> : tensor<i32>
// CHECK-NEXT:  %c_1 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:  %c_2 = stablehlo.constant dense<41> : tensor<i32>
// CHECK-NEXT:  %0 = stablehlo.add %c_1, %c_2 : tensor<i32>
// CHECK-NEXT:  %1 = stablehlo.subtract %c_0, %0 : tensor<i32>
// CHECK-NEXT:  %2 = stablehlo.maximum %1, %c_1 : tensor<i32>
// CHECK-NEXT:  %3 = stablehlo.add %2, %c_2 : tensor<i32>
// CHECK-NEXT:  %4 = stablehlo.add %3, %c : tensor<i32>
// CHECK-NEXT:  %5 = stablehlo.divide %4, %c_2 : tensor<i32>
// CHECK-NEXT:  %6 = stablehlo.maximum %c_1, %5 : tensor<i32>
// CHECK-NEXT:  return %6 : tensor<i32>
// CHECK-NEXT:  }

// The lanes broadcast along another dimension, the reduce over the lanes
// only: a value of the other dimension's shape.
func.func @lane_rows(%n: tensor<i64>) -> tensor<4xi64> {
  %init = stablehlo.constant dense<0> : tensor<i64>
  %lane = stablehlo.iota dim = 0 : tensor<1000xi64>
  %b = stablehlo.broadcast_in_dim %lane, dims = [1] : (tensor<1000xi64>) -> tensor<4x1000xi64>
  %nb = stablehlo.broadcast_in_dim %n, dims = [] : (tensor<i64>) -> tensor<4x1000xi64>
  %m = stablehlo.subtract %nb, %b : tensor<4x1000xi64>
  %0 = stablehlo.reduce(%m init: %init) applies stablehlo.minimum across dimensions = [1] : (tensor<4x1000xi64>, tensor<i64>) -> tensor<4xi64>
  return %0 : tensor<4xi64>
}

// CHECK:  func.func @lane_rows(%arg0: tensor<i64>) -> tensor<4xi64> {
// CHECK-NEXT:  %c = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:  %c_0 = stablehlo.constant dense<999> : tensor<i64>
// CHECK-NEXT:  %0 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:  %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:  %2 = stablehlo.subtract %1, %0 : tensor<4xi64>
// CHECK-NEXT:  %3 = stablehlo.minimum %c, %2 : tensor<4xi64>
// CHECK-NEXT:  return %3 : tensor<4xi64>
// CHECK-NEXT:  }

// The smallest of a maximum is not the maximum of the smallest: the reduce
// stays.
func.func @min_of_max() -> tensor<i64> {
  %c9 = stablehlo.constant dense<9> : tensor<1000xi64>
  %init = stablehlo.constant dense<0> : tensor<i64>
  %lane = stablehlo.iota dim = 0 : tensor<1000xi64>
  %d = stablehlo.subtract %c9, %lane : tensor<1000xi64>
  %m = stablehlo.maximum %lane, %d : tensor<1000xi64>
  %0 = stablehlo.reduce(%m init: %init) applies stablehlo.minimum across dimensions = [0] : (tensor<1000xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}

// CHECK:  func.func @min_of_max
// CHECK:    stablehlo.reduce

// An input the evaluator does not know stays.
func.func @iota_bound_unknown(%x: tensor<1024xi32>) -> tensor<i32> {
  %init = stablehlo.constant dense<0> : tensor<i32>
  %lane = stablehlo.iota dim = 0 : tensor<1024xi32>
  %a = stablehlo.add %x, %lane : tensor<1024xi32>
  %0 = stablehlo.reduce(%a init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<1024xi32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK:  func.func @iota_bound_unknown
// CHECK:    stablehlo.reduce
