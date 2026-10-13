// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// A raised kernel's iteration variables start from the layout ops on its
// arguments (`bitcast_convert(reshape(%arg))`); the ones it never updates
// hoist out of the loop, so the untouched arguments come back as themselves.
func.func @main(%arg0: tensor<96xi8>, %arg1: tensor<96xi8>, %arg2: tensor<96xi8>, %arg3: tensor<i32>) -> (tensor<96xi8>, tensor<96xi8>, tensor<96xi8>) {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<1> : tensor<i64>
  %0 = stablehlo.reshape %arg0 : (tensor<96xi8>) -> tensor<12x8xi8>
  %1 = stablehlo.bitcast_convert %0 : (tensor<12x8xi8>) -> tensor<12xf64>
  %2 = stablehlo.reshape %arg1 : (tensor<96xi8>) -> tensor<12x8xi8>
  %3 = stablehlo.bitcast_convert %2 : (tensor<12x8xi8>) -> tensor<12xf64>
  %4 = stablehlo.reshape %arg2 : (tensor<96xi8>) -> tensor<12x8xi8>
  %5 = stablehlo.bitcast_convert %4 : (tensor<12x8xi8>) -> tensor<12xf64>
  %6 = stablehlo.convert %arg3 : (tensor<i32>) -> tensor<i64>
  %7:4 = stablehlo.while(%iterArg = %c, %iterArg_1 = %1, %iterArg_2 = %3, %iterArg_3 = %5) : tensor<i64>, tensor<12xf64>, tensor<12xf64>, tensor<12xf64>
  cond {
    %14 = stablehlo.compare LT, %iterArg, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %14 : tensor<i1>
  } do {
    %14 = stablehlo.dynamic_slice %iterArg_1, %iterArg, sizes = [1] : (tensor<12xf64>, tensor<i64>) -> tensor<1xf64>
    %15 = stablehlo.dynamic_slice %iterArg_2, %iterArg, sizes = [1] : (tensor<12xf64>, tensor<i64>) -> tensor<1xf64>
    %16 = stablehlo.subtract %14, %15 : tensor<1xf64>
    %17 = stablehlo.dynamic_update_slice %iterArg_3, %16, %iterArg : (tensor<12xf64>, tensor<1xf64>, tensor<i64>) -> tensor<12xf64>
    %18 = stablehlo.add %iterArg, %c_0 : tensor<i64>
    stablehlo.return %18, %iterArg_1, %iterArg_2, %17 : tensor<i64>, tensor<12xf64>, tensor<12xf64>, tensor<12xf64>
  }
  %8 = stablehlo.bitcast_convert %7#1 : (tensor<12xf64>) -> tensor<12x8xi8>
  %9 = stablehlo.reshape %8 : (tensor<12x8xi8>) -> tensor<96xi8>
  %10 = stablehlo.bitcast_convert %7#2 : (tensor<12xf64>) -> tensor<12x8xi8>
  %11 = stablehlo.reshape %10 : (tensor<12x8xi8>) -> tensor<96xi8>
  %12 = stablehlo.bitcast_convert %7#3 : (tensor<12xf64>) -> tensor<12x8xi8>
  %13 = stablehlo.reshape %12 : (tensor<12x8xi8>) -> tensor<96xi8>
  return %9, %11, %13 : tensor<96xi8>, tensor<96xi8>, tensor<96xi8>
}

// CHECK:    func.func @main(%arg0: tensor<96xi8>, %arg1: tensor<96xi8>, %arg2: tensor<96xi8>, %arg3: tensor<i32>) -> (tensor<96xi8>, tensor<96xi8>, tensor<96xi8>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<96xi8>) -> tensor<12x8xi8>
// CHECK-NEXT:    %1 = stablehlo.bitcast_convert %0 : (tensor<12x8xi8>) -> tensor<12xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %arg1 : (tensor<96xi8>) -> tensor<12x8xi8>
// CHECK-NEXT:    %3 = stablehlo.bitcast_convert %2 : (tensor<12x8xi8>) -> tensor<12xf64>
// CHECK-NEXT:    %4 = stablehlo.reshape %arg2 : (tensor<96xi8>) -> tensor<12x8xi8>
// CHECK-NEXT:    %5 = stablehlo.bitcast_convert %4 : (tensor<12x8xi8>) -> tensor<12xf64>
// CHECK-NEXT:    %6 = stablehlo.convert %arg3 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:    %7:2 = stablehlo.while(%iterArg = %c, %iterArg_1 = %5) : tensor<i64>, tensor<12xf64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %10 = stablehlo.compare LT, %iterArg, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %10 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %10 = stablehlo.dynamic_slice %1, %iterArg, sizes = [1] : (tensor<12xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:      %11 = stablehlo.dynamic_slice %3, %iterArg, sizes = [1] : (tensor<12xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:      %12 = stablehlo.subtract %10, %11 : tensor<1xf64>
// CHECK-NEXT:      %13 = stablehlo.dynamic_update_slice %iterArg_1, %12, %iterArg : (tensor<12xf64>, tensor<1xf64>, tensor<i64>) -> tensor<12xf64>
// CHECK-NEXT:      %14 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %14, %13 : tensor<i64>, tensor<12xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %8 = stablehlo.bitcast_convert %7#1 : (tensor<12xf64>) -> tensor<12x8xi8>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<12x8xi8>) -> tensor<96xi8>
// CHECK-NEXT:    return %arg0, %arg1, %9 : tensor<96xi8>, tensor<96xi8>, tensor<96xi8>
// CHECK-NEXT:  }
