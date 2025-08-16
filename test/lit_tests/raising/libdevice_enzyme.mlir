// RUN: enzymexlamlir-opt --libdevice-funcs-raise %s | FileCheck %s

llvm.mlir.global external @enzyme_dup(0 : i32) {addr_space = 1 : i32, alignment = 4 : i64, dso_local } : i32
llvm.mlir.global external @enzyme_const(0 : i32) {addr_space = 1 : i32, alignment = 4 : i64, dso_local } : i32

llvm.func internal @store(%x: !llvm.ptr, %out: !llvm.ptr) {
  %val = llvm.load %x : !llvm.ptr -> f32
  llvm.store %val, %out : f32, !llvm.ptr
  llvm.return
}

llvm.func external @_Z17__enzyme_autodiffIJiPfS0_iS0_S0_EEiPvDpT_(%arg0: !llvm.ptr, %arg1: i32, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i32, %arg5: !llvm.ptr, %arg6: !llvm.ptr)

// CHECK: llvm.func @marked_dup
llvm.func @marked_dup(%x: !llvm.ptr, %dx: !llvm.ptr, %out: !llvm.ptr, %dout: !llvm.ptr) {
  %edup = llvm.mlir.addressof @enzyme_dup : !llvm.ptr<1>
  %cast = llvm.addrspacecast %edup : !llvm.ptr<1> to !llvm.ptr
  %dupv = llvm.load %cast : !llvm.ptr -> i32
  %f = llvm.mlir.addressof @store : !llvm.ptr
  // CHECK: enzyme.autodiff @store(%arg0, %arg1, %arg2, %arg3) {activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity = []} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.call @_Z17__enzyme_autodiffIJiPfS0_iS0_S0_EEiPvDpT_(%f, %dupv, %x, %dx, %dupv, %out, %dout) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

llvm.func external @__enzyme_autodiff1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr)

// CHECK: llvm.func @infer_dup
llvm.func @infer_dup(%x: !llvm.ptr, %dx: !llvm.ptr, %out: !llvm.ptr, %dout: !llvm.ptr) {
  %f = llvm.mlir.addressof @store : !llvm.ptr
  // CHECK: enzyme.autodiff @store(%arg0, %arg1, %arg2, %arg3) {activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity = []} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.call @__enzyme_autodiff1(%f, %x, %dx, %out, %dout) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

llvm.func external @__enzyme_autodiff2(%arg0: !llvm.ptr, %arg1: i32, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr)

// CHECK: llvm.func @marked_const
llvm.func @marked_const(%x: !llvm.ptr, %out: !llvm.ptr, %dout: !llvm.ptr) {
  %ecst = llvm.mlir.addressof @enzyme_const : !llvm.ptr<1>
  %cast = llvm.addrspacecast %ecst : !llvm.ptr<1> to !llvm.ptr
  %cstv = llvm.load %cast : !llvm.ptr -> i32
  %f = llvm.mlir.addressof @store : !llvm.ptr
  // CHECK: enzyme.autodiff @store(%arg0, %arg1, %arg2) {activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>], ret_activity = []} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.call @__enzyme_autodiff2(%f, %cstv, %x, %out, %dout) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}
