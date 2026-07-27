// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// Test that convert-parallel-to-gpu1 correctly handles non-zero lower bounds
// in scf.parallel inside gpu_wrapper. Previously, createSplitOp hardcoded
// lower bounds to 0 and used upper bounds directly as grid/block dims instead
// of trip counts (ub - lb)

module {
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
