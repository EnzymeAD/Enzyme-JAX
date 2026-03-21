// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// Tests that ParallelizeBlockOps correctly handles LLVM::AllocaOp:
//   BarrierOp inserted before C' section
//   LLVM::AllocaOp in "after parallel" cloned to innerBlock start

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func local_unnamed_addr @test_parallelize_block_ops_llvm_alloca(%arg0: !llvm.ptr {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {dso_local, passthrough = ["mustprogress"]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64, 1>
    %1 = "enzymexla.gpu_wrapper"(%c4, %c1, %c1, %c8, %c1, %c1) ({
      scf.parallel (%arg1) = (%c0) to (%c4) step (%c1) {
        // A: read ops before inner parallel
        %2 = memref.load %0[%arg1] : memref<?xf64, 1>
        scf.parallel (%arg2) = (%c0) to (%c8) step (%c1) {
          // B: inner parallel body
          %5 = memref.load %0[%arg2] : memref<?xf64, 1>
          %6 = arith.addf %5, %5 : f64
          memref.store %6, %0[%arg2] : memref<?xf64, 1>
          scf.reduce
        }
        // C: LLVM::AllocaOp + write after inner parallel
        %3 = llvm.alloca %c1_i64 x !llvm.struct<"struct.Result", (f64)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
        %4 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<?xf64>
        memref.store %2, %4[%c0] : memref<?xf64>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    llvm.return %c0_i32 : i32
  }
}

// CHECK:  llvm.func local_unnamed_addr @test_parallelize_block_ops_llvm_alloca(%arg0: !llvm.ptr {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {dso_local, passthrough = ["mustprogress"]} {
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %c8 = arith.constant 8 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64, 1>
// CHECK-NEXT:    %1 = "enzymexla.gpu_error"() ({
// CHECK-NEXT:      scf.if %true {
// CHECK-NEXT:        gpu.launch blocks(%arg1, %arg2, %arg3) in (%arg7 = %c4, %arg8 = %c1, %arg9 = %c1) threads(%arg4, %arg5, %arg6) in (%arg10 = %c8, %arg11 = %c1, %arg12 = %c1) {
// CHECK-NEXT:          %c1_i64_0 = arith.constant 1 : i64
// CHECK-NEXT:          %c0_1 = arith.constant 0 : index
// CHECK-NEXT:          %block_id_x = gpu.block_id  x
// CHECK-NEXT:          %thread_id_x = gpu.thread_id  x
// CHECK-NEXT:          %2 = llvm.alloca %c1_i64_0 x !llvm.struct<"struct.Result", (f64)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:          %3 = memref.load %0[%block_id_x] : memref<?xf64, 1>
// CHECK-NEXT:          %4 = memref.load %0[%thread_id_x] : memref<?xf64, 1>
// CHECK-NEXT:          %5 = arith.addf %4, %4 : f64
// CHECK-NEXT:          memref.store %5, %0[%thread_id_x] : memref<?xf64, 1>
// CHECK-NEXT:          gpu.barrier
// CHECK-NEXT:          %6 = arith.cmpi eq, %thread_id_x, %c0_1 : index
// CHECK-NEXT:          scf.if %6 {
// CHECK-NEXT:            %7 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:            memref.store %3, %7[%c0_1] : memref<?xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:          gpu.terminator
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : () -> index
// CHECK-NEXT:    llvm.return %c0_i32 : i32
// CHECK-NEXT:  }
