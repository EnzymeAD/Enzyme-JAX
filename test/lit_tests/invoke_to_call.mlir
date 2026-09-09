// RUN: enzymexlamlir-opt %s --invoke-to-call | FileCheck %s

// @leaf calls nothing and @through_leaf only calls it, so neither can unwind
// and their invokes become calls; the landing pad @only_leaf no longer reaches
// goes with it. @may_throw is a declaration and @rethrows resumes, so their
// invokes stay.

llvm.func @__gxx_personality_v0(...) -> i32
llvm.func @may_throw()

llvm.func @leaf(%arg0: !llvm.ptr) -> i32 {
  %0 = llvm.load %arg0 : !llvm.ptr -> i32
  llvm.return %0 : i32
}

llvm.func @through_leaf(%arg0: !llvm.ptr) -> i32 {
  %0 = llvm.call @leaf(%arg0) : (!llvm.ptr) -> i32
  llvm.return %0 : i32
}

llvm.func @rethrows() attributes {personality = @__gxx_personality_v0} {
  llvm.invoke @may_throw() to ^bb1 unwind ^bb2 : () -> ()
^bb1:
  llvm.return
^bb2:
  %0 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %0 : !llvm.struct<(ptr, i32)>
}

llvm.func @caller(%arg0: !llvm.ptr) -> i32 attributes {personality = @__gxx_personality_v0} {
  %0 = llvm.invoke @through_leaf(%arg0) to ^bb1 unwind ^bb3 : (!llvm.ptr) -> i32
^bb1:
  llvm.invoke @rethrows() to ^bb2 unwind ^bb3 : () -> ()
^bb2:
  llvm.return %0 : i32
^bb3:
  %1 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %1 : !llvm.struct<(ptr, i32)>
}

llvm.func @only_leaf(%arg0: !llvm.ptr) -> i32 attributes {personality = @__gxx_personality_v0} {
  %0 = llvm.invoke @leaf(%arg0) to ^bb1 unwind ^bb2 : (!llvm.ptr) -> i32
^bb1:
  llvm.return %0 : i32
^bb2:
  %1 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %1 : !llvm.struct<(ptr, i32)>
}

// CHECK:    llvm.func @__gxx_personality_v0(...) -> i32
// CHECK-NEXT:  llvm.func @may_throw()
// CHECK-NEXT:  llvm.func @leaf(%arg0: !llvm.ptr) -> i32 attributes {no_unwind} {
// CHECK-NEXT:    %0 = llvm.load %arg0 : !llvm.ptr -> i32
// CHECK-NEXT:    llvm.return %0 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func @through_leaf(%arg0: !llvm.ptr) -> i32 attributes {no_unwind} {
// CHECK-NEXT:    %0 = llvm.call @leaf(%arg0) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.return %0 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func @rethrows() attributes {personality = @__gxx_personality_v0} {
// CHECK-NEXT:    llvm.invoke @may_throw() to ^bb1 unwind ^bb2 : () -> ()
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  ^bb2:  // pred: ^bb0
// CHECK-NEXT:    %0 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
// CHECK-NEXT:    llvm.resume %0 : !llvm.struct<(ptr, i32)>
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func @caller(%arg0: !llvm.ptr) -> i32 attributes {personality = @__gxx_personality_v0} {
// CHECK-NEXT:    %0 = llvm.call @through_leaf(%arg0) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    llvm.invoke @rethrows() to ^bb2 unwind ^bb3 : () -> ()
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    llvm.return %0 : i32
// CHECK-NEXT:  ^bb3:  // pred: ^bb1
// CHECK-NEXT:    %1 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
// CHECK-NEXT:    llvm.resume %1 : !llvm.struct<(ptr, i32)>
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func @only_leaf(%arg0: !llvm.ptr) -> i32 attributes {personality = @__gxx_personality_v0} {
// CHECK-NEXT:    %0 = llvm.call @leaf(%arg0) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    llvm.return %0 : i32
// CHECK-NEXT:  }
