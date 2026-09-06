// RUN: enzymexlamlir-opt %s --xla-megakernelize --symbol-dce | FileCheck %s
// RUN: enzymexlamlir-opt %s --lower-affine --xla-megakernelize --symbol-dce | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(xla-megakernelize,symbol-dce,convert-polygeist-to-llvm{backend=xla-gpu})" | FileCheck %s --check-prefix=LOWER

// Minimized from Enzyme-GPU-Tests/LBM/lbm.o.bmk.mlir. Keep the host control
// flow and wrapper ordering representative of LBM while making the raised
// StableHLO bodies small enough for a focused regression test.
module {
  llvm.func local_unnamed_addr @_Z26CUDA_LBM_kernel_loop_inneriPfS_(
      %arg0: i32 {llvm.noundef},
      %arg1: !llvm.ptr {llvm.noalias, llvm.noundef},
      %arg2: !llvm.ptr {llvm.noalias, llvm.noundef})
      attributes {dso_local, no_signed_zeros_fp_math = true, no_unwind,
                  passthrough = ["mustprogress", ["min-legal-vector-width", "0"],
                                 ["no-trapping-math", "true"]],
                  "uniform-work-group-size"} {
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.divsi %arg0, %c2_i32 : i32
    %1 = arith.maxui %0, %c1_i32 : i32
    %2 = arith.maxsi %1, %c1_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.addi %arg0, %c1_i32 : i32
    %5 = arith.cmpi uge, %4, %c3_i32 : i32
    scf.if %5 {
      affine.for %arg3 = 0 to %3 {
        %6 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
        %7 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xf32>
        enzymexla.xla_wrapper @rxla$raised_0 (%6, %7) :
            (memref<?xf32>, memref<?xf32>) -> ()
        %8 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xf32>
        %9 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
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

// CHECK-LABEL: llvm.func local_unnamed_addr @_Z26CUDA_LBM_kernel_loop_inneriPfS_(
// CHECK:         %[[C8:.*]] = arith.constant 8 : index
// CHECK:         scf.if
// CHECK-NOT:       affine.for
// CHECK-NOT:       scf.for
// CHECK:           %[[HOST_BOUND:.*]] = memref.alloca() : memref<i64>
// CHECK:           memref.store %{{.*}}, %[[HOST_BOUND]][] : memref<i64>
// CHECK:           %[[DEVICE_BOUND:.*]] = gpu.alloc {{.*}}: memref<i64, 1>
// CHECK:           enzymexla.memcpy %[[DEVICE_BOUND]], %[[HOST_BOUND]], %[[C8]] : memref<i64, 1>, memref<i64>
// CHECK:           enzymexla.xla_wrapper @[[LBM_KERNEL:rxla[$]megakernel_[0-9]+]]
// CHECK-NOT:       enzymexla.xla_wrapper
// CHECK:           gpu.dealloc %[[DEVICE_BOUND]] : memref<i64, 1>
// CHECK-NOT:       gpu.alloc
// CHECK:         }
// CHECK:         llvm.return

// CHECK-NOT: func.func private @rxla$raised_0
// CHECK-NOT: func.func private @rxla$raised_1

// CHECK: func.func private @[[LBM_KERNEL]](
// CHECK-SAME:  tensor<i64>, tensor<?xf32>, tensor<?xf32>
// CHECK:       stablehlo.constant dense<0> : tensor<i64>
// CHECK:       stablehlo.constant dense<1> : tensor<i64>
// CHECK:       stablehlo.while
// CHECK:       cond {
// CHECK:         stablehlo.compare LT, {{.*}}, SIGNED
// CHECK:       } do {
// CHECK:         stablehlo.add {{.*}} : tensor<i64>
// CHECK:         stablehlo.add {{.*}} : tensor<?xf32>
// CHECK:         stablehlo.multiply {{.*}} : tensor<?xf32>
// CHECK:         stablehlo.return
// CHECK:       }

// LOWER-LABEL: llvm.func local_unnamed_addr @_Z26CUDA_LBM_kernel_loop_inneriPfS_(
// LOWER:         %[[DEVICE_BOUND:.*]] = llvm.call @reactantXLAMalloc
// LOWER:         llvm.call @reactantXLAMemcpy({{.*}}, %[[DEVICE_BOUND]],
// LOWER:         llvm.call @reactantXLAExec
// LOWER:         llvm.call @reactantXLAFree({{.*}}, %[[DEVICE_BOUND]])
