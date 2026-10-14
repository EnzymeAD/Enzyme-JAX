// RUN: enzymexlamlir-opt %s --xla-megakernelize --symbol-dce | FileCheck %s --match-full-lines --implicit-check-not=gpu.alloc --implicit-check-not=gpu.dealloc
// RUN: enzymexlamlir-opt %s --lower-affine --xla-megakernelize --symbol-dce | FileCheck %s --match-full-lines --implicit-check-not=gpu.alloc --implicit-check-not=gpu.dealloc
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(xla-megakernelize,symbol-dce,convert-polygeist-to-llvm{backend=xla-gpu})" | FileCheck %s --match-full-lines --check-prefix=LOWER

// This case keeps the host loop and wrapper order from LBM.
// Use small StableHLO bodies to test the loop and allocation lifetime.
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

// CHECK-LABEL: module {
// CHECK-NEXT:   llvm.func local_unnamed_addr @_Z26CUDA_LBM_kernel_loop_inneriPfS_(%[[VAL_0:.*]]: i32 {llvm.noundef}, %[[VAL_1:.*]]: !llvm.ptr {llvm.noalias, llvm.noundef}, %[[VAL_2:.*]]: !llvm.ptr {llvm.noalias, llvm.noundef}) attributes {"uniform-work-group-size", dso_local, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"]]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 8 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 2 : i32
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 3 : i32
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 1 : i32
// CHECK-NEXT:     %[[ALLOCA_0:.*]] = memref.alloca() : memref<i64>
// CHECK-NEXT:     %[[DIVSI_0:.*]] = arith.divsi %[[VAL_0]], %[[CONSTANT_1]] : i32
// CHECK-NEXT:     %[[MAXUI_0:.*]] = arith.maxui %[[DIVSI_0]], %[[CONSTANT_3]] : i32
// CHECK-NEXT:     %[[MAXSI_0:.*]] = arith.maxsi %[[MAXUI_0]], %[[CONSTANT_3]] : i32
// CHECK-NEXT:     %[[INDEX_CAST_0:.*]] = arith.index_cast %[[MAXSI_0]] : i32 to index
// CHECK-NEXT:     %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_3]] : i32
// CHECK-NEXT:     %[[CMPI_0:.*]] = arith.cmpi uge, %[[ADDI_0]], %[[CONSTANT_2]] : i32
// CHECK-NEXT:     scf.if %[[CMPI_0]] {
// CHECK-NEXT:       %[[INDEX_CAST_1:.*]] = arith.index_cast %[[INDEX_CAST_0]] : index to i64
// CHECK-NEXT:       memref.store %[[INDEX_CAST_1]], %[[ALLOCA_0]][] : memref<i64>
// CHECK-NEXT:       %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @rxla$raised_0_bound_1 : memref<i64, 1>
// CHECK-NEXT:       enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_0]], %[[CONSTANT_0]] : memref<i64, 1>, memref<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = "enzymexla.pointer2memref"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:       %[[VAL_4:.*]] = "enzymexla.pointer2memref"(%[[VAL_2]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:       enzymexla.xla_wrapper @rxla$raised_0 (%[[GET_GLOBAL_TEMP_0]], %[[VAL_3]], %[[VAL_4]]) : (memref<i64, 1>, memref<?xf32>, memref<?xf32>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @rxla$raised_0(%[[VAL_5:.*]]: tensor<i64>, %[[VAL_6:.*]]: tensor<?xf32>, %[[VAL_7:.*]]: tensor<?xf32>) -> (tensor<i64>, tensor<?xf32>, tensor<?xf32>) {
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %[[WHILE_0:.*]]:3 = stablehlo.while(%[[VAL_8:.*]] = %[[CONSTANT_4]], %[[VAL_9:.*]] = %[[VAL_6]], %[[VAL_10:.*]] = %[[VAL_7]]) : tensor<i64>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_8]], %[[VAL_5]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_8]], %[[CONSTANT_5]] : tensor<i64>
// CHECK-NEXT:       %[[ADD_1:.*]] = stablehlo.add %[[VAL_9]], %[[VAL_10]] : tensor<?xf32>
// CHECK-NEXT:       %[[MULTIPLY_0:.*]] = stablehlo.multiply %[[ADD_1]], %[[VAL_9]] : tensor<?xf32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[MULTIPLY_0]], %[[ADD_1]] : tensor<i64>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_5]], %[[WHILE_0]]#1, %[[WHILE_0]]#2 : tensor<i64>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @rxla$raised_0_bound_1 : memref<i64, 1>
// CHECK-NEXT: }

// Allocate the bound at module initialization. Each host call loads the
// same handle and copies the new bound. Free it at module finalization.
// LOWER-LABEL: module {
// LOWER-NEXT:   llvm.func @reactantXLAFree(!llvm.ptr, !llvm.ptr)
// LOWER-NEXT:   llvm.func @reactantXLAMalloc(!llvm.ptr, i64, i64, !llvm.ptr) -> !llvm.ptr
// LOWER-NEXT:   llvm.func @reactantXLAExec(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, ...)
// LOWER-NEXT:   llvm.mlir.global internal constant @xlamod$75b6a4dbdbe2193a24055e8b95757147("func.func private @reactant_kernel(%arg0: tensor<i64>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> (tensor<i64>, tensor<?xf32>, tensor<?xf32>) {\0A  %c = stablehlo.constant dense<0> : tensor<i64>\0A  %c_0 = stablehlo.constant dense<1> : tensor<i64>\0A  %0:3 = stablehlo.while(%iterArg = %c, %iterArg_1 = %arg1, %iterArg_2 = %arg2) : tensor<i64>, tensor<?xf32>, tensor<?xf32>\0A  cond {\0A    %1 = stablehlo.compare LT, %iterArg, %arg0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>\0A    stablehlo.return %1 : tensor<i1>\0A  } do {\0A    %1 = stablehlo.add %iterArg, %c_0 : tensor<i64>\0A    %2 = stablehlo.add %iterArg_1, %iterArg_2 : tensor<?xf32>\0A    %3 = stablehlo.multiply %2, %iterArg_1 : tensor<?xf32>\0A    stablehlo.return %1, %3, %2 : tensor<i64>, tensor<?xf32>, tensor<?xf32>\0A  }\0A  return %arg0, %0#1, %0#2 : tensor<i64>, tensor<?xf32>, tensor<?xf32>\0A}\0A\00") {addr_space = 0 : i32}
// LOWER-NEXT:   llvm.mlir.global internal constant @xlabackend("xla-gpu\00") {addr_space = 0 : i32}
// LOWER-NEXT:   llvm.func @reactantXLADeInit(!llvm.ptr)
// LOWER-NEXT:   llvm.func @reactantXLAInit(!llvm.ptr, !llvm.ptr)
// LOWER-NEXT:   llvm.func @reactantXLAMemcpy(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32)
// LOWER-NEXT:   llvm.func local_unnamed_addr @_Z26CUDA_LBM_kernel_loop_inneriPfS_(%[[VAL_0:.*]]: i32 {llvm.noundef}, %[[VAL_1:.*]]: !llvm.ptr {llvm.noalias, llvm.noundef}, %[[VAL_2:.*]]: !llvm.ptr {llvm.noalias, llvm.noundef}) attributes {"uniform-work-group-size", dso_local, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"]]} {
// LOWER-NEXT:     %[[MLIR_0:.*]] = llvm.mlir.zero : !llvm.ptr
// LOWER-NEXT:     %[[MLIR_1:.*]] = llvm.mlir.constant(3 : i64) : i64
// LOWER-NEXT:     %[[MLIR_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// LOWER-NEXT:     %[[MLIR_3:.*]] = llvm.mlir.addressof @xlamod$75b6a4dbdbe2193a24055e8b95757147 : !llvm.ptr
// LOWER-NEXT:     %[[MLIR_4:.*]] = llvm.mlir.addressof @__reactant_xla_data : !llvm.ptr
// LOWER-NEXT:     %[[MLIR_5:.*]] = llvm.mlir.addressof @__reactant_temp_rxla$raised_0_bound_1 : !llvm.ptr
// LOWER-NEXT:     %[[MLIR_6:.*]] = llvm.mlir.constant(1 : i32) : i32
// LOWER-NEXT:     %[[MLIR_7:.*]] = llvm.mlir.constant(3 : i32) : i32
// LOWER-NEXT:     %[[MLIR_8:.*]] = llvm.mlir.constant(2 : i32) : i32
// LOWER-NEXT:     %[[MLIR_9:.*]] = llvm.mlir.constant(8 : i64) : i64
// LOWER-NEXT:     %[[MLIR_10:.*]] = llvm.mlir.constant(1 : i64) : i64
// LOWER-NEXT:     %[[ALLOCA_0:.*]] = llvm.alloca %[[MLIR_10]] x !llvm.array<3 x i64> : (i64) -> !llvm.ptr
// LOWER-NEXT:     %[[ALLOCA_1:.*]] = llvm.alloca %[[MLIR_10]] x i64 : (i64) -> !llvm.ptr
// LOWER-NEXT:     %[[SDIV_0:.*]] = llvm.sdiv %[[VAL_0]], %[[MLIR_8]] : i32
// LOWER-NEXT:     %[[INTR_0:.*]] = llvm.intr.umax(%[[SDIV_0]], %[[MLIR_6]]) : (i32, i32) -> i32
// LOWER-NEXT:     %[[INTR_1:.*]] = llvm.intr.smax(%[[INTR_0]], %[[MLIR_6]]) : (i32, i32) -> i32
// LOWER-NEXT:     %[[SEXT_0:.*]] = llvm.sext %[[INTR_1]] : i32 to i64
// LOWER-NEXT:     %[[ADD_0:.*]] = llvm.add %[[VAL_0]], %[[MLIR_6]] : i32
// LOWER-NEXT:     %[[ICMP_0:.*]] = llvm.icmp "uge" %[[ADD_0]], %[[MLIR_7]] : i32
// LOWER-NEXT:     llvm.cond_br %[[ICMP_0]], ^bb1, ^bb2
// LOWER-NEXT:   ^bb1:  // pred: ^bb0
// LOWER-NEXT:     %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[ALLOCA_1]][] : (!llvm.ptr) -> !llvm.ptr, i64
// LOWER-NEXT:     llvm.store %[[SEXT_0]], %[[GETELEMENTPTR_0]] : i64, !llvm.ptr
// LOWER-NEXT:     %[[LOAD_0:.*]] = llvm.load %[[MLIR_5]] : !llvm.ptr -> !llvm.ptr<1>
// LOWER-NEXT:     %[[ADDRSPACECAST_0:.*]] = llvm.addrspacecast %[[LOAD_0]] : !llvm.ptr<1> to !llvm.ptr
// LOWER-NEXT:     llvm.call @reactantXLAMemcpy(%[[MLIR_4]], %[[ADDRSPACECAST_0]], %[[ALLOCA_1]], %[[MLIR_9]], %[[MLIR_6]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> ()
// LOWER-NEXT:     %[[GETELEMENTPTR_1:.*]] = llvm.getelementptr %[[MLIR_3]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<852 x i8>
// LOWER-NEXT:     %[[GETELEMENTPTR_2:.*]] = llvm.getelementptr %[[ALLOCA_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
// LOWER-NEXT:     llvm.store %[[LOAD_0]], %[[GETELEMENTPTR_2]] : !llvm.ptr<1>, !llvm.ptr
// LOWER-NEXT:     %[[GETELEMENTPTR_3:.*]] = llvm.getelementptr %[[ALLOCA_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
// LOWER-NEXT:     llvm.store %[[VAL_1]], %[[GETELEMENTPTR_3]] : !llvm.ptr, !llvm.ptr
// LOWER-NEXT:     %[[GETELEMENTPTR_4:.*]] = llvm.getelementptr %[[ALLOCA_0]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
// LOWER-NEXT:     llvm.store %[[VAL_2]], %[[GETELEMENTPTR_4]] : !llvm.ptr, !llvm.ptr
// LOWER-NEXT:     llvm.call @reactantXLAExec(%[[MLIR_4]], %[[GETELEMENTPTR_1]], %[[MLIR_1]], %[[ALLOCA_0]], %[[MLIR_2]], %[[MLIR_0]]) vararg(!llvm.func<void (ptr, ptr, i64, ptr, i64, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr) -> ()
// LOWER-NEXT:     llvm.br ^bb2
// LOWER-NEXT:   ^bb2:  // 2 preds: ^bb0, ^bb1
// LOWER-NEXT:     llvm.return
// LOWER-NEXT:   }
// LOWER-NEXT:   llvm.mlir.global internal @__reactant_temp_rxla$raised_0_bound_1() {addr_space = 0 : i32} : !llvm.ptr<1> {
// LOWER-NEXT:     %[[MLIR_11:.*]] = llvm.mlir.zero : !llvm.ptr<1>
// LOWER-NEXT:     llvm.return %[[MLIR_11]] : !llvm.ptr<1>
// LOWER-NEXT:   }
// LOWER-NEXT:   llvm.func internal @__reactant_temps_init() {
// LOWER-NEXT:     %[[MLIR_12:.*]] = llvm.mlir.addressof @__reactant_temp_rxla$raised_0_bound_1 : !llvm.ptr
// LOWER-NEXT:     %[[MLIR_13:.*]] = llvm.mlir.addressof @__reactant_xla_data : !llvm.ptr
// LOWER-NEXT:     %[[MLIR_14:.*]] = llvm.mlir.constant(5 : i64) : i64
// LOWER-NEXT:     %[[MLIR_15:.*]] = llvm.mlir.constant(0 : i64) : i64
// LOWER-NEXT:     %[[MLIR_16:.*]] = llvm.mlir.constant(1 : i64) : i64
// LOWER-NEXT:     %[[ALLOCA_2:.*]] = llvm.alloca %[[MLIR_16]] x !llvm.array<0 x i64> : (i64) -> !llvm.ptr
// LOWER-NEXT:     %[[CALL_0:.*]] = llvm.call @reactantXLAMalloc(%[[MLIR_13]], %[[MLIR_14]], %[[MLIR_15]], %[[ALLOCA_2]]) : (!llvm.ptr, i64, i64, !llvm.ptr) -> !llvm.ptr
// LOWER-NEXT:     %[[ADDRSPACECAST_1:.*]] = llvm.addrspacecast %[[CALL_0]] : !llvm.ptr to !llvm.ptr<1>
// LOWER-NEXT:     llvm.store %[[ADDRSPACECAST_1]], %[[MLIR_12]] : !llvm.ptr<1>, !llvm.ptr
// LOWER-NEXT:     llvm.return
// LOWER-NEXT:   }
// LOWER-NEXT:   llvm.func internal @__reactant_temps_deinit() {
// LOWER-NEXT:     %[[MLIR_17:.*]] = llvm.mlir.zero : !llvm.ptr<1>
// LOWER-NEXT:     %[[MLIR_18:.*]] = llvm.mlir.addressof @__reactant_xla_data : !llvm.ptr
// LOWER-NEXT:     %[[MLIR_19:.*]] = llvm.mlir.addressof @__reactant_temp_rxla$raised_0_bound_1 : !llvm.ptr
// LOWER-NEXT:     %[[LOAD_1:.*]] = llvm.load %[[MLIR_19]] : !llvm.ptr -> !llvm.ptr<1>
// LOWER-NEXT:     %[[ADDRSPACECAST_2:.*]] = llvm.addrspacecast %[[LOAD_1]] : !llvm.ptr<1> to !llvm.ptr
// LOWER-NEXT:     llvm.call @reactantXLAFree(%[[MLIR_18]], %[[ADDRSPACECAST_2]]) : (!llvm.ptr, !llvm.ptr) -> ()
// LOWER-NEXT:     llvm.store %[[MLIR_17]], %[[MLIR_19]] : !llvm.ptr<1>, !llvm.ptr
// LOWER-NEXT:     llvm.return
// LOWER-NEXT:   }
// LOWER-NEXT:   llvm.mlir.global_ctors ctors = [@__reactant_temps_init], priorities = [65535 : i32], data = [#llvm.zero]
// LOWER-NEXT:   llvm.mlir.global_dtors dtors = [@__reactant_temps_deinit], priorities = [65535 : i32], data = [#llvm.zero]
// LOWER-NEXT:   llvm.func linkonce @__reactant_xla_init() {
// LOWER-NEXT:     %[[MLIR_20:.*]] = llvm.mlir.addressof @__reactant_xla_data : !llvm.ptr
// LOWER-NEXT:     %[[MLIR_21:.*]] = llvm.mlir.addressof @xlabackend : !llvm.ptr
// LOWER-NEXT:     %[[GETELEMENTPTR_5:.*]] = llvm.getelementptr %[[MLIR_21]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
// LOWER-NEXT:     llvm.call @reactantXLAInit(%[[MLIR_20]], %[[GETELEMENTPTR_5]]) : (!llvm.ptr, !llvm.ptr) -> ()
// LOWER-NEXT:     llvm.return
// LOWER-NEXT:   }
// LOWER-NEXT:   llvm.func linkonce @__reactant_xla_deinit() {
// LOWER-NEXT:     %[[MLIR_22:.*]] = llvm.mlir.addressof @__reactant_xla_data : !llvm.ptr
// LOWER-NEXT:     llvm.call @reactantXLADeInit(%[[MLIR_22]]) : (!llvm.ptr) -> ()
// LOWER-NEXT:     llvm.return
// LOWER-NEXT:   }
// LOWER-NEXT:   llvm.mlir.global_ctors ctors = [@__reactant_xla_init], priorities = [65534 : i32], data = [#llvm.zero]
// LOWER-NEXT:   llvm.mlir.global_dtors dtors = [@__reactant_xla_deinit], priorities = [65534 : i32], data = [#llvm.zero]
// LOWER-NEXT:   llvm.mlir.global linkonce @__reactant_xla_data() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.ptr
// LOWER-NEXT: }
