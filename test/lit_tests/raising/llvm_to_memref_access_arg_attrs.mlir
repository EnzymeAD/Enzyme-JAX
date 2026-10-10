// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-memref-access)" | FileCheck %s

module {
  llvm.func @cpp_kernel(
      %out: !llvm.ptr {enzymexla.memref_type = memref<8xf32>},
      %in: !llvm.ptr {enzymexla.memref_type = memref<8xf32>, llvm.readonly}) {
    %zero = llvm.mlir.constant(0 : i64) : i64
    %in_ptr = llvm.getelementptr inbounds %in[%zero]
      : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %value = llvm.load %in_ptr : !llvm.ptr -> f32
    %out_ptr = llvm.getelementptr inbounds %out[%zero]
      : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %value, %out_ptr : f32, !llvm.ptr
    llvm.return
  }
}

// CHECK-LABEL: func.func @cpp_kernel(
// CHECK-SAME: %[[OUT:[^:]+]]: memref<8xf32>
// CHECK-SAME: %[[IN:[^:]+]]: memref<8xf32> {llvm.readonly}
// CHECK: %[[OUT_PTR:.*]] = "enzymexla.memref2pointer"(%[[OUT]])
// CHECK: %[[IN_PTR:.*]] = "enzymexla.memref2pointer"(%[[IN]])
// CHECK: llvm.getelementptr {{.*}} %[[IN_PTR]]
// CHECK: llvm.getelementptr {{.*}} %[[OUT_PTR]]
