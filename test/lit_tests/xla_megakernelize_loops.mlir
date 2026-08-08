// RUN: enzymexlamlir-opt %s --xla-megakernelize --symbol-dce | FileCheck %s

module {
  llvm.func @lift_dynamic_loop(%lb: i32, %ub: i32, %step: i32,
                              %arg0: !llvm.ptr {llvm.noalias},
                              %arg1: !llvm.ptr {llvm.noalias}) {
    scf.for unsigned %iv = %lb to %ub step %step : i32 {
      %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
      %1 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @loop_body (%0, %1) :
          (memref<?xf32>, memref<?xf32>) -> ()
    }
    llvm.return
  }

  func.func private @loop_body(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<?xf32>
    return %arg0, %0 : tensor<?xf32>, tensor<?xf32>
  }

  // The device buffer contains n. Reconstruct the canonicalized upper-bound
  // expression inside StableHLO rather than copying its host result.
  llvm.func @lift_mirrored_expression(
      %n: i32, %device_n: !llvm.ptr {llvm.noalias},
      %arg0: !llvm.ptr {llvm.noalias},
      %arg1: !llvm.ptr {llvm.noalias}) {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %mirrored_n = enzymexla.device_mirror %n, %device_n :
        (i32, !llvm.ptr) -> i32
    %half = arith.divsi %mirrored_n, %c2_i32 : i32
    %upper = arith.addi %half, %c1_i32 : i32
    scf.for %iv = %c1_i32 to %upper step %c1_i32 : i32 {
      %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
      %1 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @mirrored_loop_body (%0, %1) :
          (memref<?xf32>, memref<?xf32>) -> ()
    }
    llvm.return
  }

  func.func private @mirrored_loop_body(%arg0: tensor<?xf32>,
                                         %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<?xf32>
    return %arg0, %0 : tensor<?xf32>, tensor<?xf32>
  }

  // A one-element memref is also a valid scalar mirror. Its allocation is
  // owned by the caller and remains live across the replacement wrapper call.
  llvm.func @lift_memref_mirror(%ub: i32,
                                %arg0: !llvm.ptr {llvm.noalias},
                                %arg1: !llvm.ptr {llvm.noalias}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %device_ub = gpu.alloc () : memref<1xi32, 1>
    %mirrored_ub = enzymexla.device_mirror %ub, %device_ub :
        (i32, memref<1xi32, 1>) -> i32
    scf.for %iv = %c0_i32 to %mirrored_ub step %c1_i32 : i32 {
      %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
      %1 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @memref_mirrored_loop_body (%0, %1) :
          (memref<?xf32>, memref<?xf32>) -> ()
    }
    gpu.dealloc %device_ub : memref<1xi32, 1>
    llvm.return
  }

  func.func private @memref_mirrored_loop_body(%arg0: tensor<?xf32>,
                                                %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    return %arg0, %arg1 : tensor<?xf32>, tensor<?xf32>
  }

  // With a single tensor buffer the loop itself is safe to lift, but the
  // unannotated pointer relationship remains MayAlias. Keep scalar staging.
  llvm.func @mirror_may_alias_falls_back(%ub: i32,
                                         %device_ub: !llvm.ptr,
                                         %buffer: !llvm.ptr) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %mirrored_ub = enzymexla.device_mirror %ub, %device_ub :
        (i32, !llvm.ptr) -> i32
    scf.for %iv = %c0_i32 to %mirrored_ub step %c1_i32 : i32 {
      %0 = "enzymexla.pointer2memref"(%buffer) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @one_buffer_loop_body (%0) :
          (memref<?xf32>) -> ()
    }
    llvm.return
  }

  func.func private @one_buffer_loop_body(%arg0: tensor<?xf32>)
      -> tensor<?xf32> {
    return %arg0 : tensor<?xf32>
  }

  llvm.func @lift_static_loop(%arg0: !llvm.ptr {llvm.noalias},
                              %arg1: !llvm.ptr {llvm.noalias}) {
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c2_i32 = arith.constant 2 : i32
    scf.for %iv = %c1_i32 to %c5_i32 step %c2_i32 : i32 {
      %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
      %1 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @static_loop_body (%0, %1) :
          (memref<?xf32>, memref<?xf32>) -> ()
    }
    llvm.return
  }

  func.func private @static_loop_body(%arg0: tensor<?xf32>,
                                      %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<?xf32>
    return %0, %arg1 : tensor<?xf32>, tensor<?xf32>
  }

  // Without a NoAlias proof, keeping each aliased-or-not input as a separate
  // StableHLO loop state would be unsound.
  llvm.func @do_not_lift_may_alias(%lb: i32, %ub: i32, %step: i32,
                                   %arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    scf.for %iv = %lb to %ub step %step : i32 {
      %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
      %1 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @may_alias_loop_body (%0, %1) :
          (memref<?xf32>, memref<?xf32>) -> ()
    }
    llvm.return
  }

  func.func private @may_alias_loop_body(%arg0: tensor<?xf32>,
                                         %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    return %arg0, %arg1 : tensor<?xf32>, tensor<?xf32>
  }

}

// CHECK-LABEL: llvm.func @lift_dynamic_loop(
// CHECK-SAME:    %[[HOST_LB:.*]]: i32, %[[HOST_UB:.*]]: i32, %[[HOST_STEP:.*]]: i32,
// CHECK-SAME:    %[[HOST_BUFFER0:.*]]: !llvm.ptr {{.*}}, %[[HOST_BUFFER1:.*]]: !llvm.ptr
// CHECK-NOT:     scf.for
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[LB_HOST:.*]] = memref.alloca() : memref<1xi32>
// CHECK:         memref.store %[[HOST_LB]], %[[LB_HOST]][%{{.*}}] : memref<1xi32>
// CHECK:         %[[LB_DEVICE:.*]] = gpu.alloc {{.*}}: memref<1xi32, 1>
// CHECK:         enzymexla.memcpy %[[LB_DEVICE]], %[[LB_HOST]], %[[C4]] : memref<1xi32, 1>, memref<1xi32>
// CHECK:         %[[UB_HOST:.*]] = memref.alloca() : memref<1xi32>
// CHECK:         memref.store %[[HOST_UB]], %[[UB_HOST]][%{{.*}}] : memref<1xi32>
// CHECK:         %[[UB_DEVICE:.*]] = gpu.alloc {{.*}}: memref<1xi32, 1>
// CHECK:         enzymexla.memcpy %[[UB_DEVICE]], %[[UB_HOST]], %[[C4]] : memref<1xi32, 1>, memref<1xi32>
// CHECK:         %[[STEP_HOST:.*]] = memref.alloca() : memref<1xi32>
// CHECK:         memref.store %[[HOST_STEP]], %[[STEP_HOST]][%{{.*}}] : memref<1xi32>
// CHECK:         %[[STEP_DEVICE:.*]] = gpu.alloc {{.*}}: memref<1xi32, 1>
// CHECK:         enzymexla.memcpy %[[STEP_DEVICE]], %[[STEP_HOST]], %[[C4]] : memref<1xi32, 1>, memref<1xi32>
// CHECK:         %[[ARG0_MEMREF:.*]] = "enzymexla.pointer2memref"(%[[HOST_BUFFER0]])
// CHECK:         %[[ARG1_MEMREF:.*]] = "enzymexla.pointer2memref"(%[[HOST_BUFFER1]])
// CHECK:         enzymexla.xla_wrapper @[[SCF_KERNEL:rxla[$]megakernel_[0-9]+]]
// CHECK-SAME:      (%[[LB_DEVICE]], %[[UB_DEVICE]], %[[STEP_DEVICE]], %[[ARG0_MEMREF]], %[[ARG1_MEMREF]])
// CHECK:         gpu.dealloc %[[LB_DEVICE]] : memref<1xi32, 1>
// CHECK:         gpu.dealloc %[[UB_DEVICE]] : memref<1xi32, 1>
// CHECK:         gpu.dealloc %[[STEP_DEVICE]] : memref<1xi32, 1>
// CHECK:         llvm.return

// CHECK-NOT: func.func private @loop_body

// CHECK-LABEL: llvm.func @lift_mirrored_expression(
// CHECK-SAME:    %[[HOST_N:[a-zA-Z0-9_]+]]: i32, %[[DEVICE_N:[a-zA-Z0-9_]+]]: !llvm.ptr
// CHECK-NOT:     gpu.alloc
// CHECK-NOT:     enzymexla.memcpy
// CHECK:         %[[N_MEMREF:.*]] = "enzymexla.pointer2memref"(%[[DEVICE_N]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK:         enzymexla.xla_wrapper @[[MIRROR_KERNEL:rxla[$]megakernel_[0-9]+]]
// CHECK-SAME:      (%[[N_MEMREF]], {{.*}}, {{.*}})
// CHECK-NOT:     gpu.dealloc
// CHECK:         llvm.return

// CHECK-LABEL: llvm.func @lift_memref_mirror(
// CHECK:         %[[MEMREF_MIRROR:.*]] = gpu.alloc () : memref<1xi32, 1>
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     enzymexla.memcpy
// CHECK:         enzymexla.xla_wrapper @[[MEMREF_MIRROR_KERNEL:rxla[$]megakernel_[0-9]+]]
// CHECK-SAME:      (%[[MEMREF_MIRROR]], {{.*}}, {{.*}})
// CHECK:         gpu.dealloc %[[MEMREF_MIRROR]] : memref<1xi32, 1>
// CHECK:         llvm.return

// CHECK-LABEL: llvm.func @mirror_may_alias_falls_back(
// CHECK:         %[[FALLBACK_DEVICE:.*]] = gpu.alloc {{.*}}: memref<1xi32, 1>
// CHECK:         enzymexla.memcpy %[[FALLBACK_DEVICE]],
// CHECK:         enzymexla.xla_wrapper @[[FALLBACK_KERNEL:rxla[$]megakernel_[0-9]+]]
// CHECK-SAME:      (%[[FALLBACK_DEVICE]], {{.*}})
// CHECK:         gpu.dealloc %[[FALLBACK_DEVICE]] : memref<1xi32, 1>
// CHECK:         llvm.return

// CHECK-LABEL: llvm.func @lift_static_loop(
// CHECK-NOT:     scf.for
// CHECK-NOT:     gpu.alloc
// CHECK:         %[[STATIC_ARG0:.*]] = "enzymexla.pointer2memref"
// CHECK:         %[[STATIC_ARG1:.*]] = "enzymexla.pointer2memref"
// CHECK:         enzymexla.xla_wrapper @[[STATIC_KERNEL:rxla[$]megakernel_[0-9]+]]
// CHECK-SAME:      (%[[STATIC_ARG0]], %[[STATIC_ARG1]])
// CHECK-NOT:     gpu.dealloc
// CHECK:         llvm.return

// CHECK-NOT: func.func private @static_loop_body

// CHECK-LABEL: llvm.func @do_not_lift_may_alias
// CHECK:         scf.for
// CHECK:           enzymexla.xla_wrapper @may_alias_loop_body
// CHECK:         llvm.return

// CHECK:       func.func private @[[STATIC_KERNEL]](
// CHECK-SAME:      %[[STATIC_BUFFER0:.*]]: tensor<?xf32>, %[[STATIC_BUFFER1:.*]]: tensor<?xf32>
// CHECK:         stablehlo.constant dense<1> : tensor<i32>
// CHECK:         stablehlo.constant dense<5> : tensor<i32>
// CHECK:         stablehlo.constant dense<2> : tensor<i32>
// CHECK:         %[[STATIC_RESULTS:.*]]:5 = stablehlo.while
// CHECK:         stablehlo.subtract
// CHECK:         return %[[STATIC_RESULTS]]#3, %[[STATIC_RESULTS]]#4

// CHECK:       func.func private @[[MIRROR_KERNEL]](
// CHECK-SAME:      %[[DEVICE_N_BUFFER:.*]]: tensor<?xi32>,
// CHECK-DAG:     %[[DEVICE_N_VALUE:.*]] = stablehlo.reshape %[[DEVICE_N_BUFFER]] : (tensor<?xi32>) -> tensor<i32>
// CHECK-DAG:     %[[ONE:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-DAG:     %[[TWO:.*]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK:         %[[HALF:.*]] = stablehlo.divide %[[DEVICE_N_VALUE]], %[[TWO]] : tensor<i32>
// CHECK:         %[[UPPER:.*]] = stablehlo.add %[[HALF]], %[[ONE]] : tensor<i32>
// CHECK:         %[[MIRROR_RESULTS:.*]]:5 = stablehlo.while
// CHECK:         return %[[DEVICE_N_BUFFER]], %[[MIRROR_RESULTS]]#3, %[[MIRROR_RESULTS]]#4

// CHECK:       func.func private @[[SCF_KERNEL]](
// CHECK-SAME:      %[[LB_BUFFER:.*]]: tensor<?xi32>, %[[UB_BUFFER:.*]]: tensor<?xi32>, %[[STEP_BUFFER:.*]]: tensor<?xi32>
// CHECK-SAME:      %[[BUFFER0:.*]]: tensor<?xf32>, %[[BUFFER1:.*]]: tensor<?xf32>
// CHECK:         %[[LB:.*]] = stablehlo.reshape %[[LB_BUFFER]] : (tensor<?xi32>) -> tensor<i32>
// CHECK:         %[[UB:.*]] = stablehlo.reshape %[[UB_BUFFER]] : (tensor<?xi32>) -> tensor<i32>
// CHECK:         %[[STEP:.*]] = stablehlo.reshape %[[STEP_BUFFER]] : (tensor<?xi32>) -> tensor<i32>
// CHECK:         %[[RESULTS:.*]]:5 = stablehlo.while
// CHECK-SAME:      %[[ITER:.*]] = %[[LB]], %[[LIMIT:.*]] = %[[UB]], %[[STRIDE:.*]] = %[[STEP]]
// CHECK:         cond {
// CHECK:           %[[CONTINUE:.*]] = stablehlo.compare LT, %[[ITER]], %[[LIMIT]], UNSIGNED
// CHECK:           stablehlo.return %[[CONTINUE]]
// CHECK:         } do {
// CHECK:           %[[NEXT:.*]] = stablehlo.add %[[ITER]], %[[STRIDE]]
// CHECK:           %[[UPDATED:.*]] = stablehlo.add
// CHECK:           stablehlo.return %[[NEXT]], %[[LIMIT]], %[[STRIDE]], {{.*}}, %[[UPDATED]]
// CHECK:         }
// CHECK:         return %[[LB_BUFFER]], %[[UB_BUFFER]], %[[STEP_BUFFER]],
// CHECK-SAME:           %[[RESULTS]]#3, %[[RESULTS]]#4
