// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-llvm-to-cf)" | FileCheck %s

// CHECK-LABEL: func @test_br
func.func @test_br() {
  // CHECK: cf.br ^bb1
  llvm.br ^bb1
^bb1:
  return
}

// CHECK-LABEL: func @test_cond_br
func.func @test_cond_br(%cond: i1) {
  // CHECK: cf.cond_br %arg0, ^bb1, ^bb2
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  return
^bb2:
  llvm.unreachable
}

llvm.func internal unnamed_addr fastcc @throw_boundserror_2676() attributes {dso_local, no_inline, sym_visibility = "private"} {
  llvm.unreachable
}
// CHECK-LABEL: func @test_switch
func.func @test_switch(%val: i32) {
  // CHECK: cf.switch %arg0 : i32, [
  // CHECK-NEXT: default: ^bb3,
  // CHECK-NEXT: 0: ^bb1,
  // CHECK-NEXT: 1: ^bb2
  llvm.switch %val : i32, ^bb3 [
    0: ^bb1,
    1: ^bb2
  ]
^bb1:
  return
^bb2:
  llvm.unreachable
^bb3:
  llvm.call fastcc @throw_boundserror_2676() : () -> ()
  llvm.unreachable
}
