// RUN: enzymexlamlir-opt %s --xla-megakernelize --symbol-dce | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(xla-megakernelize,symbol-dce,convert-polygeist-to-llvm{backend=xla-gpu})" | FileCheck %s --check-prefix=LOWER

// Minimized from Enzyme-GPU-Tests/LBM/lbm.o.bmk.mlir. Keep the host control
// flow and wrapper ordering representative of LBM while making the raised
// StableHLO bodies small enough for a focused regression test.
module {
  llvm.func local_unnamed_addr @_Z26CUDA_LBM_kernel_loop_inneriPKiPfS1_(
      %arg0: i32 {llvm.noundef},
      %arg1: !llvm.ptr {llvm.noalias, llvm.noundef},
      %arg2: !llvm.ptr {llvm.noalias, llvm.noundef},
      %arg3: !llvm.ptr {llvm.noalias, llvm.noundef})
      attributes {dso_local, no_signed_zeros_fp_math = true, no_unwind,
                  passthrough = ["mustprogress", ["min-legal-vector-width", "0"],
                                 ["no-trapping-math", "true"]],
                  "uniform-work-group-size"} {
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %mirrored = enzymexla.device_mirror %arg0, %arg1 :
        (i32, !llvm.ptr) -> i32
    %0 = arith.addi %mirrored, %c1_i32 : i32
    %1 = arith.cmpi uge, %0, %c3_i32 : i32
    scf.if %1 {
      %2 = arith.divsi %mirrored, %c2_i32 : i32
      %3 = arith.maxui %2, %c1_i32 : i32
      %4 = arith.maxsi %3, %c1_i32 : i32
      %5 = arith.addi %4, %c1_i32 : i32
      scf.for %arg4 = %c1_i32 to %5 step %c1_i32 : i32 {
        %6 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xf32>
        %7 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?xf32>
        enzymexla.xla_wrapper @rxla$raised_0 (%6, %7) :
            (memref<?xf32>, memref<?xf32>) -> ()
        %8 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?xf32>
        %9 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xf32>
        enzymexla.xla_wrapper @rxla$raised_1 (%8, %9) :
            (memref<?xf32>, memref<?xf32>) -> ()
      }
    }
    llvm.return
  }

  func.func private @rxla$raised_0(%arg0: tensor<?xf32>,
                                   %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<?xf32>
    return %arg0, %0 : tensor<?xf32>, tensor<?xf32>
  }

  func.func private @rxla$raised_1(%arg0: tensor<?xf32>,
                                   %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<?xf32>
    return %arg0, %0 : tensor<?xf32>, tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func local_unnamed_addr @_Z26CUDA_LBM_kernel_loop_inneriPKiPfS1_(
// CHECK:         scf.if
// CHECK-NOT:       scf.for
// CHECK-NOT:       memref.alloca
// CHECK-NOT:       gpu.alloc
// CHECK-NOT:       enzymexla.memcpy
// CHECK:           %[[DEVICE_BOUND:.*]] = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xi32>
// CHECK:           enzymexla.xla_wrapper @[[LBM_KERNEL:rxla[$]megakernel_[0-9]+]]
// CHECK-SAME:        (%[[DEVICE_BOUND]], {{.*}}, {{.*}})
// CHECK-NOT:       enzymexla.xla_wrapper
// CHECK-NOT:       gpu.dealloc
// CHECK:         }
// CHECK:         llvm.return

// CHECK-NOT: func.func private @rxla$raised_0
// CHECK-NOT: func.func private @rxla$raised_1

// CHECK: func.func private @[[LBM_KERNEL]](
// CHECK-SAME:  %[[BOUND_BUFFER:.*]]: tensor<?xi32>, %{{.*}}: tensor<?xf32>, %{{.*}}: tensor<?xf32>
// CHECK-DAG:   %[[BOUND:.*]] = stablehlo.reshape %[[BOUND_BUFFER]] : (tensor<?xi32>) -> tensor<i32>
// CHECK-DAG:   stablehlo.constant dense<1> : tensor<i32>
// CHECK-DAG:   stablehlo.constant dense<2> : tensor<i32>
// CHECK:       stablehlo.divide
// CHECK:       stablehlo.maximum
// CHECK:       stablehlo.maximum
// CHECK:       stablehlo.while
// CHECK:       cond {
// CHECK:         stablehlo.compare LT, {{.*}}, SIGNED
// CHECK:       } do {
// CHECK:         stablehlo.add {{.*}} : tensor<i32>
// CHECK:         stablehlo.add {{.*}} : tensor<?xf32>
// CHECK:         stablehlo.multiply {{.*}} : tensor<?xf32>
// CHECK:         stablehlo.return
// CHECK:       }

// LOWER-LABEL: llvm.func local_unnamed_addr @_Z26CUDA_LBM_kernel_loop_inneriPKiPfS1_(
// LOWER-NOT:     llvm.call @reactantXLAMalloc
// LOWER-NOT:     llvm.call @reactantXLAMemcpy
// LOWER:         llvm.call @reactantXLAExec
// LOWER-NOT:     llvm.call @reactantXLAFree
