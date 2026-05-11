// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition)" | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>>, llvm.target_triple = "x86_64-unknown-linux-gnu"} {

  llvm.func local_unnamed_addr @main() -> (i32) {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c512 = llvm.mlir.constant(512 : i64) : i64
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c10 = llvm.mlir.constant(10 : i64) : i64

    %ptr = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %dst_array = llvm.alloca %c1 x !llvm.array<128 x i8> : (i32) -> !llvm.ptr
    %src_array = llvm.alloca %c1 x !llvm.array<128 x i8> : (i32) -> !llvm.ptr

    // Test cudaMalloc
    %0 = llvm.call @cudaMalloc(%ptr, %c512) : (!llvm.ptr, i64) -> i32

    %dev_ptr = llvm.load %ptr : !llvm.ptr -> !llvm.ptr

    // Test cudaMemset
    %1 = llvm.call @cudaMemset(%dev_ptr, %c0, %c512) : (!llvm.ptr, i32, i64) -> i32

    // Test cudaMemcpy2D
    %2 = llvm.call @cudaMemcpy2D(%dst_array, %c10, %dev_ptr, %c10, %c10, %c10, %c2) : (!llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i32) -> i32

    llvm.return %c0 : i32
  }

  llvm.func local_unnamed_addr @cudaMalloc(!llvm.ptr, i64) -> i32
  llvm.func local_unnamed_addr @cudaMemset(!llvm.ptr, i32, i64) -> i32
  llvm.func local_unnamed_addr @cudaMemcpy2D(!llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i32) -> i32
}

// CHECK:      llvm.func local_unnamed_addr @main()
// CHECK:        [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:        [[C512_VAL:%.+]] = llvm.mlir.constant(512 : i64) : i64
// CHECK:        [[C0_VAL:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:        [[C10_VAL:%.+]] = llvm.mlir.constant(10 : i64) : i64

// CHECK:        [[PTR:%.+]] = llvm.alloca [[C1]] x !llvm.ptr
// CHECK:        [[DST_ARRAY:%.+]] = llvm.alloca [[C1]] x !llvm.array<128 x i8>
// CHECK:        [[SRC_ARRAY:%.+]] = llvm.alloca [[C1]] x !llvm.array<128 x i8>

// CHECK:        [[C512:%.+]] = arith.index_cast [[C512_VAL]] : i64 to index
// CHECK:        [[MEMREF:%.+]] = gpu.alloc  ([[C512]]) : memref<?xi8, 1>
// CHECK:        [[PTR_VAL:%.+]] = "enzymexla.memref2pointer"([[MEMREF]]) : (memref<?xi8, 1>) -> !llvm.ptr

// CHECK:        llvm.store [[PTR_VAL]], [[PTR]] : !llvm.ptr, !llvm.ptr
// CHECK:        [[DEV_PTR:%.+]] = llvm.load [[PTR]]
// CHECK:        [[MEMREF_SET:%.+]] = "enzymexla.pointer2memref"([[DEV_PTR]])
// CHECK:        [[C512_2:%.+]] = arith.index_cast [[C512_VAL]] : i64 to index
// CHECK:        enzymexla.memset [[MEMREF_SET]], [[C0_VAL]], [[C512_2]] : memref<?xi8, 1>, i32

// CHECK-DAG:        [[DST_MEMREF:%.+]] = "enzymexla.pointer2memref"([[DST_ARRAY]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK-DAG:        [[SRC_MEMREF:%.+]] = "enzymexla.pointer2memref"([[DEV_PTR]]) : (!llvm.ptr) -> memref<?xi8, 1>
// CHECK-DAG:        [[DPITCH:%.+]] = arith.index_cast [[C10_VAL]] : i64 to index
// CHECK-DAG:        [[SPITCH:%.+]] = arith.index_cast [[C10_VAL]] : i64 to index
// CHECK-DAG:        [[WIDTH:%.+]] = arith.index_cast [[C10_VAL]] : i64 to index
// CHECK-DAG:        [[HEIGHT:%.+]] = arith.index_cast [[C10_VAL]] : i64 to index

// CHECK:        enzymexla.memcpy2d [[DST_MEMREF]], [[DPITCH]], [[SRC_MEMREF]], [[SPITCH]], [[WIDTH]], [[HEIGHT]] : memref<?xi8>, memref<?xi8, 1>
