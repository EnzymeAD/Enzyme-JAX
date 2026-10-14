// RUN: enzymexlamlir-opt %s --polygeist-mem2reg --split-input-file | FileCheck %s

module {
  // CHECK-LABEL: func.func @gemm_tt
  // CHECK-SAME: (%[[ARG0:.*]]: f64, %[[ARG1:.*]]: memref<1000x1000xf64>, %[[ARG2:.*]]: memref<1000x1000xf64>, %[[ARG3:.*]]: memref<1000x1000xf64>)
  // CHECK-NOT: memref.alloca
  // CHECK: affine.for %[[I:.*]] = 0 to 1000 {
  // CHECK-NEXT: affine.for %[[J:.*]] = 0 to 1000 {
  // CHECK-NEXT: %[[CST:.*]] = arith.constant 0.000000e+00 : f64
  // CHECK-NEXT: %[[RES:.*]] = affine.for %[[K:.*]] = 0 to 1000 iter_args(%[[ACC:.*]] = %[[CST]]) -> (f64) {
  // CHECK-NEXT: %[[V1:.*]] = affine.load %[[ARG1]][%[[K]], %[[I]]] : memref<1000x1000xf64>
  // CHECK-NEXT: %[[V2:.*]] = affine.load %[[ARG2]][%[[J]], %[[K]]] : memref<1000x1000xf64>
  // CHECK-NEXT: %[[V3:.*]] = arith.mulf %[[V1]], %[[V2]] : f64
  // CHECK-NEXT: %[[V4:.*]] = arith.mulf %[[ARG0]], %[[V3]] : f64
  // CHECK-NEXT: %[[V5:.*]] = arith.addf %[[ACC]], %[[V4]] : f64
  // CHECK-NEXT: affine.yield %[[V5]] : f64
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.store %[[RES]], %[[ARG3]][%[[I]], %[[J]]] : memref<1000x1000xf64>
  func.func @gemm_tt(%arg0: f64, %arg1: memref<1000x1000xf64>, %arg2: memref<1000x1000xf64>, %arg3: memref<1000x1000xf64>) {
    affine.for %arg4 = 0 to 1000 {
      affine.for %arg5 = 0 to 1000 {
        %alloca = memref.alloca() : memref<f64>
        %cst = arith.constant 0.000000e+00 : f64
        affine.store %cst, %alloca[] : memref<f64>
        affine.for %arg6 = 0 to 1000 {
          %1 = affine.load %arg1[%arg6, %arg4] : memref<1000x1000xf64>
          %2 = affine.load %arg2[%arg5, %arg6] : memref<1000x1000xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = arith.mulf %arg0, %3 : f64
          %5 = affine.load %alloca[] : memref<f64>
          %6 = arith.addf %5, %4 : f64
          affine.store %6, %alloca[] : memref<f64>
        }
        %0 = affine.load %alloca[] : memref<f64>
        affine.store %0, %arg3[%arg4, %arg5] : memref<1000x1000xf64>
      }
    }
    return
  }
}

// -----

module {
  // CHECK-LABEL: func.func @scf_for_reduction
  // CHECK-SAME: (%[[ARG0:.*]]: memref<100xf32>, %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index) -> f32
  // CHECK-NOT: memref.alloca
  // CHECK-NEXT: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT: %[[RES:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ACC:.*]] = %[[CST]]) -> (f32) {
  // CHECK-NEXT: %[[V:.*]] = memref.load %[[ARG0]][%[[IV]]] : memref<100xf32>
  // CHECK-NEXT: %[[NEXT:.*]] = arith.addf %[[ACC]], %[[V]] : f32
  // CHECK-NEXT: scf.yield %[[NEXT]] : f32
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[RES]] : f32
  func.func @scf_for_reduction(%arg0: memref<100xf32>, %lb: index, %ub: index, %step: index) -> f32 {
    %alloca = memref.alloca() : memref<f32>
    %c0 = arith.constant 0.0 : f32
    memref.store %c0, %alloca[] : memref<f32>
    scf.for %iv = %lb to %ub step %step {
      %v = memref.load %arg0[%iv] : memref<100xf32>
      %cur = memref.load %alloca[] : memref<f32>
      %next = arith.addf %cur, %v : f32
      memref.store %next, %alloca[] : memref<f32>
    }
    %res = memref.load %alloca[] : memref<f32>
    return %res : f32
  }
}

// -----

module {
  // CHECK-LABEL: func.func @existing_iter_args
  // CHECK-SAME: (%[[ARG0:.*]]: memref<10xf64>) -> (f64, f64)
  // CHECK-NOT: memref.alloca
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0.000000e+00 : f64
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1.000000e+00 : f64
  // CHECK-NEXT: %[[RES:.*]]:2 = affine.for %[[I:.*]] = 0 to 10 iter_args(%[[P:.*]] = %[[C1]], %[[S:.*]] = %[[C0]]) -> (f64, f64) {
  // CHECK-NEXT: %[[X:.*]] = affine.load %[[ARG0]][%[[I]]] : memref<10xf64>
  // CHECK-NEXT: %[[SNEXT:.*]] = arith.addf %[[S]], %[[X]] : f64
  // CHECK-NEXT: %[[PNEXT:.*]] = arith.mulf %[[P]], %[[X]] : f64
  // CHECK-NEXT: affine.yield %[[PNEXT]], %[[SNEXT]] : f64, f64
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[RES]]#0, %[[RES]]#1 : f64, f64
  func.func @existing_iter_args(%arg0: memref<10xf64>) -> (f64, f64) {
    %alloca = memref.alloca() : memref<f64>
    %c0 = arith.constant 0.0 : f64
    %c1 = arith.constant 1.0 : f64
    affine.store %c0, %alloca[] : memref<f64>
    %prod = affine.for %i = 0 to 10 iter_args(%p = %c1) -> (f64) {
      %x = affine.load %arg0[%i] : memref<10xf64>
      %s = affine.load %alloca[] : memref<f64>
      %s_next = arith.addf %s, %x : f64
      affine.store %s_next, %alloca[] : memref<f64>
      %p_next = arith.mulf %p, %x : f64
      affine.yield %p_next : f64
    }
    %sum = affine.load %alloca[] : memref<f64>
    return %prod, %sum : f64, f64
  }
}

// -----

module {
  // CHECK-LABEL: func.func @conditional_store_in_for
  // CHECK-SAME: (%[[ARG0:.*]]: memref<10xf32>, %[[THRESH:.*]]: f32) -> f32
  // CHECK-NOT: memref.alloca
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT: %[[RES:.*]] = affine.for %[[I:.*]] = 0 to 10 iter_args(%[[ACC:.*]] = %[[C0]]) -> (f32) {
  // CHECK-NEXT: %[[X:.*]] = affine.load %[[ARG0]][%[[I]]] : memref<10xf32>
  // CHECK-NEXT: %[[COND:.*]] = arith.cmpf ogt, %[[X]], %[[THRESH]] : f32
  // CHECK-NEXT: %[[IFRES:.*]] = scf.if %[[COND]] -> (f32) {
  // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[ACC]], %[[X]] : f32
  // CHECK-NEXT: scf.yield %[[ADD]] : f32
  // CHECK-NEXT: } else {
  // CHECK-NEXT: scf.yield %[[ACC]] : f32
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.yield %[[IFRES]] : f32
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[RES]] : f32
  func.func @conditional_store_in_for(%arg0: memref<10xf32>, %thresh: f32) -> f32 {
    %alloca = memref.alloca() : memref<f32>
    %c0 = arith.constant 0.0 : f32
    memref.store %c0, %alloca[] : memref<f32>
    affine.for %i = 0 to 10 {
      %x = affine.load %arg0[%i] : memref<10xf32>
      %cond = arith.cmpf ogt, %x, %thresh : f32
      scf.if %cond {
        %cur = memref.load %alloca[] : memref<f32>
        %next = arith.addf %cur, %x : f32
        memref.store %next, %alloca[] : memref<f32>
      }
    }
    %res = memref.load %alloca[] : memref<f32>
    return %res : f32
  }
}

// -----

module {
  // CHECK-LABEL: func.func @nested_for_carried_no_post_load
  // CHECK-SAME: (%[[ARG0:.*]]: memref<10x10xf64>, %[[OUT:.*]]: memref<10x10xf64>)
  // CHECK-NOT: memref.alloca
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0.000000e+00 : f64
  // CHECK-NEXT: %[[OUTER:.*]] = affine.for %[[I:.*]] = 0 to 10 iter_args(%[[ACC0:.*]] = %[[C0]]) -> (f64) {
  // CHECK-NEXT: %[[INNER:.*]] = affine.for %[[J:.*]] = 0 to 10 iter_args(%[[ACC1:.*]] = %[[ACC0]]) -> (f64) {
  // CHECK-NEXT: %[[X:.*]] = affine.load %[[ARG0]][%[[I]], %[[J]]] : memref<10x10xf64>
  // CHECK-NEXT: %[[NEXT:.*]] = arith.addf %[[ACC1]], %[[X]] : f64
  // CHECK-NEXT: affine.store %[[NEXT]], %[[OUT]][%[[I]], %[[J]]] : memref<10x10xf64>
  // CHECK-NEXT: affine.yield %[[NEXT]] : f64
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.yield %[[INNER]] : f64
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  func.func @nested_for_carried_no_post_load(%arg0: memref<10x10xf64>, %out: memref<10x10xf64>) {
    %alloca = memref.alloca() : memref<f64>
    %c0 = arith.constant 0.0 : f64
    affine.store %c0, %alloca[] : memref<f64>
    affine.for %i = 0 to 10 {
      affine.for %j = 0 to 10 {
        %x = affine.load %arg0[%i, %j] : memref<10x10xf64>
        %cur = affine.load %alloca[] : memref<f64>
        %next = arith.addf %cur, %x : f64
        affine.store %next, %alloca[] : memref<f64>
        affine.store %next, %out[%i, %j] : memref<10x10xf64>
      }
    }
    return
  }
}

// -----

module {
  func.func private @clobber(memref<f64>)
  // CHECK-LABEL: func.func @clobbered_in_for
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<f64>
  // CHECK: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: %{{.*}} = affine.load %[[ALLOCA]][] : memref<f64>
  func.func @clobbered_in_for(%arg0: memref<10xf64>) -> f64 {
    %alloca = memref.alloca() : memref<f64>
    %c0 = arith.constant 0.0 : f64
    affine.store %c0, %alloca[] : memref<f64>
    affine.for %i = 0 to 10 {
      %cur = affine.load %alloca[] : memref<f64>
      %x = affine.load %arg0[%i] : memref<10xf64>
      %next = arith.addf %cur, %x : f64
      affine.store %next, %alloca[] : memref<f64>
      func.call @clobber(%alloca) : (memref<f64>) -> ()
    }
    %res = affine.load %alloca[] : memref<f64>
    return %res : f64
  }
}

// -----

module {
  // CHECK-LABEL: func.func @uninitialized_before_for
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<f64>
  // CHECK: affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT: %{{.*}} = affine.load %[[ALLOCA]][] : memref<f64>
  func.func @uninitialized_before_for(%arg0: memref<10xf64>) -> f64 {
    %alloca = memref.alloca() : memref<f64>
    affine.for %i = 0 to 10 {
      %cur = affine.load %alloca[] : memref<f64>
      %x = affine.load %arg0[%i] : memref<10xf64>
      %next = arith.addf %cur, %x : f64
      affine.store %next, %alloca[] : memref<f64>
    }
    %res = affine.load %alloca[] : memref<f64>
    return %res : f64
  }
}
