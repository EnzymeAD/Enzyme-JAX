// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s --implicit-check-not=gpu.alloc --implicit-check-not=gpu.dealloc

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
}

// -----

module {
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
}

// -----

// An existing wrapper already requires independent input buffers.
// Do not require a second alias proof to lift its loop.
module {
  llvm.func @lift_without_noalias(%lb: i32, %ub: i32, %step: i32,
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
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<?xf32>
    return %arg0, %0 : tensor<?xf32>, tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func @lift_dynamic_loop(
// CHECK-SAME:    %[[HOST_LB:.*]]: i32, %[[HOST_UB:.*]]: i32, %[[HOST_STEP:.*]]: i32,
// CHECK-SAME:    %[[HOST_BUFFER0:.*]]: !llvm.ptr {{.*}}, %[[HOST_BUFFER1:.*]]: !llvm.ptr
// CHECK-NOT:     scf.for
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[LB_HOST:.*]] = memref.alloca() : memref<i32>
// CHECK:         memref.store %[[HOST_LB]], %[[LB_HOST]][] : memref<i32>
// CHECK:         %[[LB_DEVICE:.*]] = enzymexla.get_global_temp @[[LB_TEMP:rxla[$]megakernel_[0-9]+_bound_0]] : memref<i32, 1>
// CHECK:         enzymexla.memcpy %[[LB_DEVICE]], %[[LB_HOST]], %[[C4]] : memref<i32, 1>, memref<i32>
// CHECK:         %[[UB_HOST:.*]] = memref.alloca() : memref<i32>
// CHECK:         memref.store %[[HOST_UB]], %[[UB_HOST]][] : memref<i32>
// CHECK:         %[[UB_DEVICE:.*]] = enzymexla.get_global_temp @[[UB_TEMP:rxla[$]megakernel_[0-9]+_bound_1]] : memref<i32, 1>
// CHECK:         enzymexla.memcpy %[[UB_DEVICE]], %[[UB_HOST]], %[[C4]] : memref<i32, 1>, memref<i32>
// CHECK:         %[[STEP_HOST:.*]] = memref.alloca() : memref<i32>
// CHECK:         memref.store %[[HOST_STEP]], %[[STEP_HOST]][] : memref<i32>
// CHECK:         %[[STEP_DEVICE:.*]] = enzymexla.get_global_temp @[[STEP_TEMP:rxla[$]megakernel_[0-9]+_bound_2]] : memref<i32, 1>
// CHECK:         enzymexla.memcpy %[[STEP_DEVICE]], %[[STEP_HOST]], %[[C4]] : memref<i32, 1>, memref<i32>
// CHECK:         %[[ARG0_MEMREF:.*]] = "enzymexla.pointer2memref"(%[[HOST_BUFFER0]])
// CHECK:         %[[ARG1_MEMREF:.*]] = "enzymexla.pointer2memref"(%[[HOST_BUFFER1]])
// CHECK:         enzymexla.xla_wrapper @[[SCF_KERNEL:rxla[$]megakernel_[0-9]+]]
// CHECK-SAME:      (%[[LB_DEVICE]], %[[UB_DEVICE]], %[[STEP_DEVICE]], %[[ARG0_MEMREF]], %[[ARG1_MEMREF]])
// CHECK:         llvm.return

// CHECK-NOT: func.func private @loop_body

// CHECK:       func.func private @[[SCF_KERNEL]](
// CHECK-SAME:      %[[LB:.*]]: tensor<i32>, %[[UB:.*]]: tensor<i32>, %[[STEP:.*]]: tensor<i32>
// CHECK-SAME:      %[[BUFFER0:.*]]: tensor<?xf32>, %[[BUFFER1:.*]]: tensor<?xf32>
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
// CHECK:         return %[[RESULTS]]#0, %[[RESULTS]]#1, %[[RESULTS]]#2,
// CHECK-SAME:           %[[RESULTS]]#3, %[[RESULTS]]#4
// CHECK: enzymexla.temp_alloc "private" @[[LB_TEMP]] : memref<i32, 1>
// CHECK: enzymexla.temp_alloc "private" @[[UB_TEMP]] : memref<i32, 1>
// CHECK: enzymexla.temp_alloc "private" @[[STEP_TEMP]] : memref<i32, 1>

// CHECK-LABEL: llvm.func @lift_static_loop(
// CHECK-NOT:     scf.for
// CHECK-NOT:     enzymexla.get_global_temp
// CHECK:         %[[STATIC_ARG0:.*]] = "enzymexla.pointer2memref"
// CHECK:         %[[STATIC_ARG1:.*]] = "enzymexla.pointer2memref"
// CHECK:         enzymexla.xla_wrapper @[[STATIC_KERNEL:rxla[$]megakernel_[0-9]+]]
// CHECK-SAME:      (%[[STATIC_ARG0]], %[[STATIC_ARG1]])
// CHECK-NOT:     enzymexla.get_global_temp
// CHECK:         llvm.return

// CHECK-NOT: func.func private @static_loop_body

// CHECK:       func.func private @[[STATIC_KERNEL]](
// CHECK-SAME:      %[[STATIC_BUFFER0:.*]]: tensor<?xf32>, %[[STATIC_BUFFER1:.*]]: tensor<?xf32>
// CHECK:         stablehlo.constant dense<1> : tensor<i32>
// CHECK:         stablehlo.constant dense<5> : tensor<i32>
// CHECK:         stablehlo.constant dense<2> : tensor<i32>
// CHECK:         %[[STATIC_RESULTS:.*]]:5 = stablehlo.while
// CHECK:         stablehlo.subtract
// CHECK:         return %[[STATIC_RESULTS]]#3, %[[STATIC_RESULTS]]#4

// CHECK-NOT: enzymexla.temp_alloc

// CHECK-LABEL: llvm.func @lift_without_noalias
// CHECK-NOT: scf.for
// CHECK: enzymexla.get_global_temp
// CHECK: enzymexla.xla_wrapper @[[UNANNOTATED_KERNEL:rxla[$]megakernel_[0-9]+]]
// CHECK: llvm.return
// CHECK: func.func private @[[UNANNOTATED_KERNEL]](
// CHECK: stablehlo.while
// CHECK: stablehlo.subtract

// -----

// Two host functions can use the same raised body. Each loop must have
// a separate bound allocation so one call cannot change another bound.
module {
  llvm.func @first_call_site(%limit: i32, %a: !llvm.ptr) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %iv = %zero to %limit step %one : i32 {
      %buffer = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @shared_body(%buffer) : (memref<?xf32>) -> ()
    }
    llvm.return
  }
  llvm.func @second_call_site(%limit: i32, %a: !llvm.ptr) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %iv = %zero to %limit step %one : i32 {
      %buffer = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @shared_body(%buffer) : (memref<?xf32>) -> ()
    }
    llvm.return
  }
  func.func private @shared_body(%a: tensor<?xf32>) -> tensor<?xf32> {
    %updated = stablehlo.add %a, %a : tensor<?xf32>
    return %updated : tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func @first_call_site(
// CHECK: %[[FIRST:.*]] = enzymexla.get_global_temp @[[$FIRST_TEMP:.*]] : memref<i32, 1>
// CHECK: enzymexla.memcpy %[[FIRST]],
// CHECK: enzymexla.xla_wrapper @{{.*}} (%[[FIRST]],
// CHECK-LABEL: llvm.func @second_call_site(
// CHECK: %[[SECOND:.*]] = enzymexla.get_global_temp @[[SECOND_TEMP:.*]] : memref<i32, 1>
// CHECK: enzymexla.memcpy %[[SECOND]],
// CHECK: enzymexla.xla_wrapper @{{.*}} (%[[SECOND]],
// CHECK-DAG: enzymexla.temp_alloc "private" @[[$FIRST_TEMP]] : memref<i32, 1>
// CHECK-DAG: enzymexla.temp_alloc "private" @[[SECOND_TEMP]] : memref<i32, 1>

// -----

// A new bound argument would change the position of argument attributes.
// Keep the loop unchanged until these attributes can be moved safely.
module {
  llvm.func @keep_argument_attributes(%limit: i32, %a: !llvm.ptr) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %iv = %zero to %limit step %one : i32 {
      %buffer = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @annotated_body(%buffer) : (memref<?xf32>) -> ()
    }
    llvm.return
  }
  func.func private @annotated_body(%a: tensor<?xf32> {test.keep})
      -> tensor<?xf32> {
    %updated = stablehlo.add %a, %a : tensor<?xf32>
    return %updated : tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func @keep_argument_attributes(
// CHECK: scf.for
// CHECK: enzymexla.xla_wrapper @annotated_body
// CHECK: func.func private @annotated_body(%{{.*}}: tensor<?xf32> {test.keep})
// CHECK-NOT: enzymexla.temp_alloc
