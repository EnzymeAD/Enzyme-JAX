// RUN: enzymexlamlir-opt --libdevice-funcs-raise %s | FileCheck %s

module {
  // CHECK: @func_fcmp
  // CHECK-NEXT: arith.cmpf ogt, %arg0, %arg1 {fastmathFlags = #llvm.fastmath<none>} : f64
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
}

