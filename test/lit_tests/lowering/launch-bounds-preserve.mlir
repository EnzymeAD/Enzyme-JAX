// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu2{backend=cuda emitGPUKernelLaunchBounds=true})" | FileCheck %s

// __launch_bounds__(maxThreads, minBlocks) arrives from the device IR as
// nvvm passthrough pairs on the error op; both must land on the kernel, and
// the preserved maxntid beats the architecture-maximum fallback when the
// block size is not known.

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, gpu.container_module, llvm.target_triple = "x86_64-unknown-linux-gnu"} {

  func.func @main(%n: index, %out: memref<100xf64, 1>) -> i32 {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = "enzymexla.gpu_error"() ({
      gpu.launch_func @main_kernel::@main_kernel blocks in (%n, %c1, %c1) threads in (%n, %c1, %c1) args(%out : memref<100xf64, 1>)
      "enzymexla.polygeist_yield"() : () -> ()
    }) {passthrough = [["target-cpu", "sm_120"], ["nvvm.maxntid", "128"], ["nvvm.minctasm", "4"]]} : () -> index
    return %c0_i32 : i32
  }

  // With statically known threads the exact bound wins over the declared one,
  // and without a matching recorded original shape the minctasm promise does
  // not transfer.
  func.func @second(%n: index, %out: memref<100xf64, 1>) -> i32 {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %1 = "enzymexla.gpu_error"() ({
      gpu.launch_func @second_kernel::@second_kernel blocks in (%n, %c1, %c1) threads in (%c64, %c1, %c1) args(%out : memref<100xf64, 1>)
      "enzymexla.polygeist_yield"() : () -> ()
    }) {passthrough = [["target-cpu", "sm_120"], ["nvvm.maxntid", "128"], ["nvvm.minctasm", "4"]]} : () -> index
    return %c0_i32 : i32
  }

  // Constant dims that reproduce the recorded original shape keep minctasm.
  func.func @third(%n: index, %out: memref<100xf64, 1>) -> i32 {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %2 = "enzymexla.gpu_error"() ({
      gpu.launch_func @third_kernel::@third_kernel blocks in (%n, %c1, %c1) threads in (%c64, %c1, %c1) args(%out : memref<100xf64, 1>)
      "enzymexla.polygeist_yield"() : () -> ()
    }) {passthrough = [["target-cpu", "sm_120"], ["nvvm.maxntid", "128"], ["nvvm.minctasm", "4"]], reactant.launch_block_size = 64 : i64} : () -> index
    return %c0_i32 : i32
  }

  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<100xf64, 1>) kernel {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.0 : f64
      memref.store %cst, %arg0[%c0] : memref<100xf64, 1>
      gpu.return
    }
  }
  gpu.module @second_kernel {
    gpu.func @second_kernel(%arg0: memref<100xf64, 1>) kernel {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.0 : f64
      memref.store %cst, %arg0[%c0] : memref<100xf64, 1>
      gpu.return
    }
  }
  gpu.module @third_kernel {
    gpu.func @third_kernel(%arg0: memref<100xf64, 1>) kernel {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 3.0 : f64
      memref.store %cst, %arg0[%c0] : memref<100xf64, 1>
      gpu.return
    }
  }
}

// CHECK: gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_120"
// CHECK: gpu.func @main_kernel
// CHECK-SAME: nvvm.maxntid = array<i32: 128, 1, 1>
// CHECK-SAME: nvvm.minctasm = 4 : i32
// CHECK: gpu.func @second_kernel
// CHECK-SAME: nvvm.maxntid = array<i32: 64, 1, 1>
// CHECK-NOT: nvvm.minctasm
// CHECK: gpu.func @third_kernel
// CHECK-SAME: nvvm.maxntid = array<i32: 64, 1, 1>
// CHECK-SAME: nvvm.minctasm = 4 : i32
