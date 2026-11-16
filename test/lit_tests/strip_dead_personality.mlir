// RUN: enzymexlamlir-opt %s --strip-dead-personality | FileCheck %s

// A personality is consulted only through landing pads.

llvm.func @__gxx_personality_v0(...) -> i32
llvm.func @maythrow()

llvm.func @no_landing_pad() attributes {personality = @__gxx_personality_v0} {
  llvm.call @maythrow() : () -> ()
  llvm.return
}

llvm.func @catches() attributes {personality = @__gxx_personality_v0} {
  llvm.invoke @maythrow() to ^ok unwind ^lp : () -> ()
^ok:
  llvm.return
^lp:
  %0 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %0 : !llvm.struct<(ptr, i32)>
}

// CHECK:    llvm.func @__gxx_personality_v0(...) -> i32
// CHECK-NEXT:  llvm.func @maythrow()
// CHECK-NEXT:  llvm.func @no_landing_pad() {
// CHECK-NEXT:    llvm.call @maythrow() : () -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func @catches() attributes {personality = @__gxx_personality_v0} {
// CHECK-NEXT:    llvm.invoke @maythrow() to ^bb1 unwind ^bb2 : () -> ()
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  ^bb2:  // pred: ^bb0
// CHECK-NEXT:    %0 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
// CHECK-NEXT:    llvm.resume %0 : !llvm.struct<(ptr, i32)>
// CHECK-NEXT:  }
