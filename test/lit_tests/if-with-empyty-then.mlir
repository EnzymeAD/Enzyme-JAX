// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(func.func(canonicalize-loops))" --split-input-file | FileCheck %s

// CHECK-LABEL: func @if_with_empty_then
// CHECK-SAME: %[[ARG0:.+]]: i1
func.func @if_with_empty_then(%arg0: i1) {
    // CHECK: %[[TRUE:.+]] = arith.constant true
    // CHECK-NEXT: %[[COND:.+]] = arith.xori %[[ARG0]], %[[TRUE]] : i1
    // CHECK: scf.if %[[COND]]
    // CHECK-NOT: else
    scf.if %arg0 {
    } else {
        func.call @some_effect() : () -> ()
    }
    return
}

func.func private @some_effect() 

// -----

// CHECK-LABEL: func @if_with_non_empty_then
// CHECK-SAME: %[[ARG0:.+]]: i1
func.func @if_with_non_empty_then(%arg0: i1) {
    // CHECK: scf.if %[[ARG0]] {
    // CHECK-NEXT:   func.call @some_effect
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   func.call @some_effect
    // CHECK-NEXT: }
    scf.if %arg0 {
        func.call @some_effect() : () -> ()
    } else {
        func.call @some_effect() : () -> ()
    }
    return
}

func.func private @some_effect() 

// -----

// CHECK-LABEL: func.func @unary_if(
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: memref<34x99x194xf64, 1>, %[[ARG2:.*]]: i1, %[[ARG3:.*]]: f64, %[[ARG4:.*]]: f64)
func.func @unary_if(%arg0: i64, %arg1: memref<34x99x194xf64, 1>, %arg2: i1, %arg3: f64, %arg4: f64) -> i1 {
  %false = arith.constant false
  %0 = "enzymexla.memref2pointer"(%arg1) : (memref<34x99x194xf64, 1>) -> !llvm.ptr<1>
  // CHECK-NOT: scf.if
  %1 = scf.if %arg2 -> (i1) {
    %2 = llvm.load %0 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
    %3 = arith.subf %arg3, %2 {fastmathFlags = #llvm.fastmath<none>} : f64
    %4 = arith.divf %3, %arg4 {fastmathFlags = #llvm.fastmath<none>} : f64
    %5 = arith.cmpf une, %4, %arg4 : f64
    scf.yield %5 : i1
  } else {
    scf.yield %false : i1
  }
  // CHECK: %[[AND:.+]] = arith.andi %[[ARG2]], %{{.+}} : i1
  // CHECK-NEXT: return %[[AND]] : i1
  return %1 : i1
} 
