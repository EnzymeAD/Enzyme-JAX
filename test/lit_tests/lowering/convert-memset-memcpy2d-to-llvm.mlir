// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=cuda})" | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=cpu})" | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=xla-gpu})" | FileCheck %s --check-prefix=XLA

module {
  func.func @test_memset(%ptr: memref<?xi8, 1>, %val: i32, %count: index) {
    enzymexla.memset %ptr, %val, %count : memref<?xi8, 1>, i32
    return
  }

  func.func @test_memcpy2d(%dst: memref<?xi8>, %dpitch: index, %src: memref<?xi8, 1>, %spitch: index, %width: index, %height: index) {
    enzymexla.memcpy2d %dst, %dpitch, %src, %spitch, %width, %height : memref<?xi8>, memref<?xi8, 1>
    return
  }
}

// CUDA-LABEL: llvm.func @test_memset
// CUDA:         llvm.call @cudaMemset({{.*}}, %arg1, %arg2) : (!llvm.ptr, i32, i64) -> i32

// CUDA-LABEL: llvm.func @test_memcpy2d
// CUDA:         [[KIND:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CUDA:         llvm.call @cudaMemcpy2D({{.*}}, %arg1, {{.*}}, %arg3, %arg4, %arg5, [[KIND]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i32) -> i32

// CPU-LABEL: llvm.func @test_memset(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i64) {
// CPU:         [[VAL:%.+]] = llvm.trunc %arg1 : i32 to i8
// CPU:         [[PTR:%.+]] = llvm.addrspacecast %arg0 : !llvm.ptr<1> to !llvm.ptr
// CPU:         "llvm.intr.memset"([[PTR]], [[VAL]], %arg2) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CPU:         llvm.return

// CPU-LABEL: llvm.func @test_memcpy2d(%arg0: !llvm.ptr, %arg1: i64, %arg2: !llvm.ptr<1>, %arg3: i64, %arg4: i64, %arg5: i64) {
// CPU:         [[C1:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CPU:         [[C0:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CPU:         llvm.br ^[[BB1:.+]]([[C0]] : i64)
// CPU:       ^[[BB1]]([[IV:%.+]]: i64):
// CPU:         [[CMP:%.+]] = llvm.icmp "slt" [[IV]], %arg5 : i64
// CPU:         llvm.cond_br [[CMP]], ^[[BB2:.+]], ^[[BB4:.+]]
// CPU:       ^[[BB2]]:
// CPU:         [[SRC_CAST:%.+]] = llvm.addrspacecast %arg2 : !llvm.ptr<1> to !llvm.ptr
// CPU:         [[DST_OFF:%.+]] = llvm.mul [[IV]], %arg1 : i64
// CPU:         [[SRC_OFF:%.+]] = llvm.mul [[IV]], %arg3 : i64
// CPU:         [[DST_GEP:%.+]] = llvm.getelementptr %arg0[[[DST_OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CPU:         [[SRC_GEP:%.+]] = llvm.getelementptr [[SRC_CAST]][[[SRC_OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CPU:         "llvm.intr.memcpy"([[DST_GEP]], [[SRC_GEP]], %arg4) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CPU:         llvm.br ^[[BB3:.+]]
// CPU:       ^[[BB3]]:
// CPU:         [[NEXT_IV:%.+]] = llvm.add [[IV]], [[C1]] : i64
// CPU:         llvm.br ^[[BB1]]([[NEXT_IV]] : i64)
// CPU:       ^[[BB4]]:
// CPU:         llvm.return

// XLA-LABEL: llvm.func @test_memset(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i64) {
// XLA:         [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// XLA:         [[XDATA:%.+]] = llvm.mlir.addressof @__reactant_xla_data : !llvm.ptr
// XLA:         [[VAL_I8:%.+]] = llvm.trunc %arg1 : i32 to i8
// XLA:         [[HOST_PTR:%.+]] = llvm.call @malloc(%arg2) : (i64) -> !llvm.ptr
// XLA:         "llvm.intr.memset"([[HOST_PTR]], [[VAL_I8]], %arg2) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// XLA:         [[DST_PTR:%.+]] = llvm.addrspacecast %arg0 : !llvm.ptr<1> to !llvm.ptr
// XLA:         llvm.call @reactantXLAMemcpy([[XDATA]], [[DST_PTR]], [[HOST_PTR]], %arg2, [[C1]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> ()
// XLA:         llvm.call @free([[HOST_PTR]]) : (!llvm.ptr) -> ()
// XLA:         llvm.return

// XLA-LABEL: llvm.func @test_memcpy2d(%arg0: !llvm.ptr, %arg1: i64, %arg2: !llvm.ptr<1>, %arg3: i64, %arg4: i64, %arg5: i64) {
// XLA:         [[C1:%.+]] = llvm.mlir.constant(1 : i64) : i64
// XLA:         [[KIND:%.+]] = llvm.mlir.constant(2 : i32) : i32
// XLA:         [[C0:%.+]] = llvm.mlir.constant(0 : i64) : i64
// XLA:         [[XDATA:%.+]] = llvm.mlir.addressof @__reactant_xla_data : !llvm.ptr
// XLA:         llvm.br ^[[BB1:.+]]([[C0]] : i64)
// XLA:       ^[[BB1]]([[IV:%.+]]: i64):
// XLA:         [[CMP:%.+]] = llvm.icmp "slt" [[IV]], %arg5 : i64
// XLA:         llvm.cond_br [[CMP]], ^[[BB2:.+]], ^[[BB4:.+]]
// XLA:       ^bb2:
// XLA:         [[SRC_CAST:%.+]] = llvm.addrspacecast %arg2 : !llvm.ptr<1> to !llvm.ptr
// XLA:         [[DST_OFF:%.+]] = llvm.mul [[IV]], %arg1 : i64
// XLA:         [[SRC_OFF:%.+]] = llvm.mul [[IV]], %arg3 : i64
// XLA:         [[DST_GEP:%.+]] = llvm.getelementptr %arg0[[[DST_OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// XLA:         [[SRC_GEP:%.+]] = llvm.getelementptr [[SRC_CAST]][[[SRC_OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// XLA:         llvm.call @reactantXLAMemcpy([[XDATA]], [[DST_GEP]], [[SRC_GEP]], %arg4, [[KIND]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> ()
// XLA:         llvm.br ^[[BB3:.+]]
// XLA:       ^[[BB3]]:
// XLA:         [[NEXT_IV:%.+]] = llvm.add [[IV]], [[C1]] : i64
// XLA:         llvm.br ^[[BB1]]([[NEXT_IV]] : i64)
// XLA:       ^[[BB4]]:
// XLA:         llvm.return
