// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// Test that convert-parallel-to-gpu1 correctly handles non-zero lower bounds
// in scf.parallel inside gpu_wrapper. Previously, createSplitOp hardcoded
// lower bounds to 0 and used upper bounds directly as grid/block dims instead
// of trip counts (ub - lb)

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @stencil_kernel(%A: memref<?xf32, 1>, %B: memref<?xf32, 1>) {
    %c1 = arith.constant 1 : index
    %c1023 = arith.constant 1023 : index
    %wrapper_result = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      scf.parallel (%i) = (%c1) to (%c1023) step (%c1) {
        %val = memref.load %A[%i] : memref<?xf32, 1>
        memref.store %val, %B[%i] : memref<?xf32, 1>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// The scf.parallel has lb=1, ub=1023, so trip count = 1022.
// Grid/block dims should use 1022 (not 1023), and the IV should be offset
// by the lower bound: iv = new_iv + 1.
// Before fix: dims used ub=1023 (arr[0..1022] instead of arr[1..1022])
// After fix:  dims use trip_count=1022, IV = new_iv + 1

// CHECK-LABEL: func.func @stencil_kernel
// CHECK:       gpu.launch
// Trip count should be 1022 = ub(1023) - lb(1), not the raw upper bound 1023
// CHECK:         %[[LB:.*]] = arith.constant 1 : index
// CHECK:         %[[TRIP:.*]] = arith.constant 1022 : index
// IV must be offset by lower bound: offset_iv = idx + lb
// CHECK:         %[[LINEAR:.*]] = arith.addi
// CHECK:         %[[OFFSET_IV:.*]] = arith.addi %[[LINEAR]], %[[LB]] : index
// Guard uses trip count (1022), not upper bound (1023)
// CHECK:         arith.cmpi ult, %[[LINEAR]], %[[TRIP]] : index
// Memory access uses the offset IV, not the raw index
// CHECK:         memref.load %{{.*}}[%[[OFFSET_IV]]] : memref<?xf32, 1>
