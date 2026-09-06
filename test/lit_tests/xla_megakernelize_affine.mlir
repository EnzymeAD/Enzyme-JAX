// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s

// Multi-result affine bounds take the maximum lower bound and minimum upper
// bound. Preserve signed index comparisons and a non-unit step.
module {
  llvm.func @affine_bounds(%lb_arg: i64, %ub_arg: i64,
                          %a: !llvm.ptr {llvm.noalias},
                          %b: !llvm.ptr {llvm.noalias}) {
    %lb = arith.index_cast %lb_arg : i64 to index
    %ub = arith.index_cast %ub_arg : i64 to index
    affine.for %iv = max affine_map<(d0) -> (d0, -3)>(%lb)
        to min affine_map<(d0) -> (d0 + 1, 17)>(%ub) step 2 {
      %0 = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
      %1 = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @body (%0, %1) : (memref<?xf32>, memref<?xf32>) -> ()
    }
    llvm.return
  }
  func.func private @body(%a: tensor<?xf32>, %b: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.add %a, %b : tensor<?xf32>
    return %a, %0 : tensor<?xf32>, tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func @affine_bounds(
// CHECK-NOT: affine.for
// CHECK: %[[LOWER:.*]] = arith.maxsi
// CHECK: %[[UPPER:.*]] = arith.minsi
// CHECK: %[[LOWER_I64:.*]] = arith.index_cast %[[LOWER]] : index to i64
// CHECK: memref.store %[[LOWER_I64]],
// CHECK: %[[LOWER_DEVICE:.*]] = gpu.alloc {{.*}}: memref<i64, 1>
// CHECK: enzymexla.memcpy %[[LOWER_DEVICE]],
// CHECK: %[[UPPER_I64:.*]] = arith.index_cast %[[UPPER]] : index to i64
// CHECK: memref.store %[[UPPER_I64]],
// CHECK: %[[UPPER_DEVICE:.*]] = gpu.alloc {{.*}}: memref<i64, 1>
// CHECK: enzymexla.memcpy %[[UPPER_DEVICE]],
// CHECK: enzymexla.xla_wrapper @rxla$megakernel_0 (%[[LOWER_DEVICE]], %[[UPPER_DEVICE]],
// CHECK: gpu.dealloc %[[LOWER_DEVICE]]
// CHECK: gpu.dealloc %[[UPPER_DEVICE]]
// CHECK: llvm.return
// CHECK: func.func private @rxla$megakernel_0(
// CHECK-SAME: %[[LB:.*]]: tensor<i64>, %[[UB:.*]]: tensor<i64>,
// CHECK: %[[STEP:.*]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK: stablehlo.while
// CHECK-SAME: %[[IV:.*]] = %[[LB]], %[[LIMIT:.*]] = %[[UB]], %[[STRIDE:.*]] = %[[STEP]],
// CHECK: stablehlo.compare LT, %[[IV]], %[[LIMIT]], SIGNED
// CHECK: stablehlo.add %[[IV]], %[[STRIDE]] : tensor<i64>
// CHECK: stablehlo.add {{.*}} : tensor<?xf32>

// -----

module {
  llvm.func @affine_static(%a: !llvm.ptr) {
    affine.for %iv = -3 to 3 step 2 {
      %0 = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @body (%0) : (memref<?xf32>) -> ()
    }
    llvm.return
  }
  func.func private @body(%a: tensor<?xf32>) -> tensor<?xf32> {
    %0 = stablehlo.add %a, %a : tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func @affine_static(
// CHECK-NOT: affine.for
// CHECK-NOT: gpu.alloc
// CHECK: enzymexla.xla_wrapper @rxla$megakernel_0
// CHECK-NOT: gpu.dealloc
// CHECK: llvm.return
// CHECK: func.func private @rxla$megakernel_0(
// CHECK: stablehlo.constant dense<-3> : tensor<i64>
// CHECK: stablehlo.constant dense<3> : tensor<i64>
// CHECK: stablehlo.constant dense<2> : tensor<i64>
// CHECK: stablehlo.while
// CHECK: stablehlo.compare LT, {{.*}}, SIGNED

// -----

// Keep IV-dependent buffers, host side effects, host loop-carried values, and
// potentially aliased wrapper arguments on the host.
module {
  llvm.func @affine_iv_used(%ub_arg: i64, %a: !llvm.ptr) {
    %ub = arith.index_cast %ub_arg : i64 to index
    affine.for %iv = 0 to %ub {
      %offset = arith.index_cast %iv : index to i64
      %ptr = llvm.getelementptr %a[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %0 = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @body (%0) : (memref<?xf32>) -> ()
    }
    llvm.return
  }
  llvm.func @affine_host_effect(%ub_arg: i64, %a: !llvm.ptr, %counter: !llvm.ptr) {
    %ub = arith.index_cast %ub_arg : i64 to index
    affine.for %iv = 0 to %ub {
      llvm.store %ub_arg, %counter : i64, !llvm.ptr
      %0 = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @body (%0) : (memref<?xf32>) -> ()
    }
    llvm.return
  }
  llvm.func @affine_iter_arg(%ub_arg: i64, %a: !llvm.ptr) -> i32 {
    %ub = arith.index_cast %ub_arg : i64 to index
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %count = affine.for %iv = 0 to %ub iter_args(%n = %c0) -> i32 {
      %0 = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @body (%0) : (memref<?xf32>) -> ()
      %next = arith.addi %n, %c1 : i32
      affine.yield %next : i32
    }
    llvm.return %count : i32
  }
  llvm.func @affine_may_alias(%ub_arg: i64, %a: !llvm.ptr, %b: !llvm.ptr) {
    %ub = arith.index_cast %ub_arg : i64 to index
    affine.for %iv = 0 to %ub {
      %0 = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
      %1 = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @pair_body (%0, %1) : (memref<?xf32>, memref<?xf32>) -> ()
    }
    llvm.return
  }
  func.func private @body(%a: tensor<?xf32>) -> tensor<?xf32> {
    %0 = stablehlo.add %a, %a : tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  func.func private @pair_body(%a: tensor<?xf32>, %b: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.add %a, %b : tensor<?xf32>
    return %a, %0 : tensor<?xf32>, tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func @affine_iv_used(
// CHECK: affine.for
// CHECK: enzymexla.xla_wrapper @body
// CHECK-LABEL: llvm.func @affine_host_effect(
// CHECK: affine.for
// CHECK: llvm.store
// CHECK: enzymexla.xla_wrapper @body
// CHECK-LABEL: llvm.func @affine_iter_arg(
// CHECK: affine.for {{.*}} iter_args
// CHECK: enzymexla.xla_wrapper @body
// CHECK: affine.yield
// CHECK-LABEL: llvm.func @affine_may_alias(
// CHECK: affine.for
// CHECK: enzymexla.xla_wrapper @pair_body
// CHECK-NOT: stablehlo.while
