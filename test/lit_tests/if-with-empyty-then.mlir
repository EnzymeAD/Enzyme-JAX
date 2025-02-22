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
