// RUN: enzymexlamlir-opt %s --strip-dead-personality --inline --discard-unreferenced-linkonce --polygeist-mem2reg --canonicalize | FileCheck %s

// A method that loads its whole capture from `this` and takes it apart with
// extractvalue, called by the function that stored the fields one by one:
// inlined and its dead definition discarded, mem2reg forwards every field.

llvm.func @__gxx_personality_v0(...) -> i32
llvm.func @use(i32, !llvm.ptr)

llvm.func linkonce_odr @run(%this: !llvm.ptr) attributes {personality = @__gxx_personality_v0} {
  %view = "enzymexla.pointer2memref"(%this) : (!llvm.ptr) -> memref<?x!llvm.struct<(ptr, i32)>>
  %functor = affine.load %view[0] : memref<?x!llvm.struct<(ptr, i32)>>
  %p = llvm.extractvalue %functor[0] : !llvm.struct<(ptr, i32)>
  %n = llvm.extractvalue %functor[1] : !llvm.struct<(ptr, i32)>
  llvm.call @use(%n, %p) : (i32, !llvm.ptr) -> ()
  llvm.return
}

llvm.func @caller(%buf: !llvm.ptr, %n: i32) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %functor = llvm.alloca %c1 x !llvm.struct<(ptr, i32)> : (i32) -> !llvm.ptr
  %ptrs = "enzymexla.pointer2memref"(%functor) : (!llvm.ptr) -> memref<?x!llvm.ptr>
  affine.store %buf, %ptrs[0] : memref<?x!llvm.ptr>
  %ints = "enzymexla.pointer2memref"(%functor) : (!llvm.ptr) -> memref<?xi32>
  affine.store %n, %ints[2] : memref<?xi32>
  llvm.call @run(%functor) : (!llvm.ptr) -> ()
  llvm.return
}

// CHECK:    llvm.func @__gxx_personality_v0(...) -> i32
// CHECK-NEXT:  llvm.func @use(i32, !llvm.ptr)
// CHECK-NEXT:  llvm.func @caller(%arg0: !llvm.ptr, %arg1: i32) {
// CHECK-NEXT:    %0 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i32)> : (i32) -> !llvm.ptr
// CHECK-NEXT:    %2 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
// CHECK-NEXT:    affine.store %arg0, %2[0] : memref<?x!llvm.ptr>
// CHECK-NEXT:    %3 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:    affine.store %arg1, %3[2] : memref<?xi32>
// CHECK-NEXT:    llvm.call @use(%arg1, %arg0) : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
