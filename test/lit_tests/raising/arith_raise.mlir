// RUN: enzymexlamlir-opt --libdevice-funcs-raise %s | FileCheck %s
// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s --check-prefix=TRUNC

module {
  // CHECK: @func_fcmp
  // CHECK-NEXT: arith.cmpf ogt, %arg0, %arg1 : f64
  func.func @func_fcmp(%arg0: f64, %arg1: f64) -> i1 {
      %res = llvm.fcmp "ogt" %arg0, %arg1 : f64
      func.return %res : i1
  }
  // CHECK: @func_icmp
  // CHECK-NEXT: arith.cmpi ugt, %arg0, %arg1 : i64
  func.func @func_icmp(%arg0: i64, %arg1: i64) -> i1 {
      %res = llvm.icmp "ugt" %arg0, %arg1 : i64
      func.return %res : i1
  }
  // CHECK: @func_flt
  // CHECK-NEXT: arith.constant 1.000000e+00 : f32
  func.func @func_flt() -> f32 {
      %res = llvm.mlir.constant(1.0 : f32) : f32
      func.return %res : f32
  }
  // CHECK: @func_int
  // CHECK-NEXT: arith.constant 1 : i32
  func.func @func_int() -> i32 {
      %res = llvm.mlir.constant(1 : i32) : i32
      func.return %res : i32
  }
  // CHECK-NOT: arith.cmpi
  func.func @ptr_icmp(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i1 {
      %res = llvm.icmp "eq" %arg0, %arg1 : !llvm.ptr
      func.return %res : i1
  }

  // trunc(x) rounds towards zero: select(x >= 0, floor(x), ceil(x)).
  // TRUNC-LABEL:   func.func @trunc_f32(
  // TRUNC-SAME:                     %[[VAL_0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
  // TRUNC:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  // TRUNC:           %[[VAL_2:.*]] = stablehlo.compare  GE, %[[VAL_0]], %[[VAL_1]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // TRUNC:           %[[VAL_3:.*]] = stablehlo.floor %[[VAL_0]] : tensor<4xf32>
  // TRUNC:           %[[VAL_4:.*]] = stablehlo.ceil %[[VAL_0]] : tensor<4xf32>
  // TRUNC:           %[[VAL_5:.*]] = stablehlo.select %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : tensor<4xi1>, tensor<4xf32>
  // TRUNC:           return %[[VAL_5]] : tensor<4xf32>
  // TRUNC:         }
  func.func @trunc_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "math.trunc"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // TRUNC-LABEL:   func.func @trunc_f64(
  // TRUNC-SAME:                     %[[VAL_0:.*]]: tensor<4xf64>) -> tensor<4xf64> {
  // TRUNC:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
  // TRUNC:           %[[VAL_2:.*]] = stablehlo.compare  GE, %[[VAL_0]], %[[VAL_1]] : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
  // TRUNC:           %[[VAL_3:.*]] = stablehlo.floor %[[VAL_0]] : tensor<4xf64>
  // TRUNC:           %[[VAL_4:.*]] = stablehlo.ceil %[[VAL_0]] : tensor<4xf64>
  // TRUNC:           %[[VAL_5:.*]] = stablehlo.select %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : tensor<4xi1>, tensor<4xf64>
  // TRUNC:           return %[[VAL_5]] : tensor<4xf64>
  // TRUNC:         }
  func.func @trunc_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = "math.trunc"(%arg0) : (tensor<4xf64>) -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}

