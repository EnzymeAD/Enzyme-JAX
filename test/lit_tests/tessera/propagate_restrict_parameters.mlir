// RUN: enzymexlamlir-opt %s -tessera-propagate-properties | FileCheck %s

// A `restrict` (noalias) pointer parameter is a handle whose object no other
// pointer reaches during the call: a fact established in the function holds
// until a call reaches it. Without restrict, another parameter may point to
// the same object, so any call that may write memory ends the fact.

module {
  tessera.define private @lib.solve(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  llvm.func @fill_spd(!llvm.ptr {tessera.establishes = ["SPD"], tessera.noescape})
  llvm.func @touch(!llvm.ptr)

  // CHECK-LABEL: llvm.func @restrict_parameter
  // CHECK: tessera.call @lib.solve({{.*}}) <arg_attrs = [{tessera.property = ["SPD"]}, {}]>
  llvm.func @restrict_parameter(%g: !llvm.ptr {llvm.noalias}, %y: !llvm.ptr, %b: !llvm.ptr) -> i32 {
    llvm.call @fill_spd(%g) : (!llvm.ptr) -> ()
    llvm.call @touch(%y) : (!llvm.ptr) -> ()
    %r = tessera.call @lib.solve(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // CHECK-LABEL: llvm.func @plain_parameter
  // CHECK: tessera.call @lib.solve
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @plain_parameter(%g: !llvm.ptr, %y: !llvm.ptr, %b: !llvm.ptr) -> i32 {
    llvm.call @fill_spd(%g) : (!llvm.ptr) -> ()
    llvm.call @touch(%y) : (!llvm.ptr) -> ()
    %r = tessera.call @lib.solve(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // Restrict, but given to a call that may change it.
  // CHECK-LABEL: llvm.func @restrict_then_changed
  // CHECK: tessera.call @lib.solve
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @restrict_then_changed(%g: !llvm.ptr {llvm.noalias}, %b: !llvm.ptr) -> i32 {
    llvm.call @fill_spd(%g) : (!llvm.ptr) -> ()
    llvm.call @touch(%g) : (!llvm.ptr) -> ()
    %r = tessera.call @lib.solve(%g, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }
}
