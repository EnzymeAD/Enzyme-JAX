// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s --match-full-lines --implicit-check-not=gpu.alloc --implicit-check-not=gpu.dealloc

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

// CHECK-LABEL: module {
// CHECK-NEXT:   llvm.func @affine_bounds(%[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: !llvm.ptr {llvm.noalias}, %[[VAL_3:.*]]: !llvm.ptr {llvm.noalias}) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 8 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 17 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant -3 : index
// CHECK-NEXT:     %[[ALLOCA_0:.*]] = memref.alloca() : memref<i64>
// CHECK-NEXT:     %[[ALLOCA_1:.*]] = memref.alloca() : memref<i64>
// CHECK-NEXT:     %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i64 to index
// CHECK-NEXT:     %[[INDEX_CAST_1:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK-NEXT:     %[[MAXSI_0:.*]] = arith.maxsi %[[INDEX_CAST_0]], %[[CONSTANT_3]] : index
// CHECK-NEXT:     %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_1]], %[[CONSTANT_2]] : index
// CHECK-NEXT:     %[[MINSI_0:.*]] = arith.minsi %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:     %[[INDEX_CAST_2:.*]] = arith.index_cast %[[MAXSI_0]] : index to i64
// CHECK-NEXT:     memref.store %[[INDEX_CAST_2]], %[[ALLOCA_1]][] : memref<i64>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @body_bound_0 : memref<i64, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_1]], %[[CONSTANT_0]] : memref<i64, 1>, memref<i64>
// CHECK-NEXT:     %[[INDEX_CAST_3:.*]] = arith.index_cast %[[MINSI_0]] : index to i64
// CHECK-NEXT:     memref.store %[[INDEX_CAST_3]], %[[ALLOCA_0]][] : memref<i64>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_1:.*]] = enzymexla.get_global_temp @body_bound_1 : memref<i64, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_1]], %[[ALLOCA_0]], %[[CONSTANT_0]] : memref<i64, 1>, memref<i64>
// CHECK-NEXT:     %[[VAL_4:.*]] = "enzymexla.pointer2memref"(%[[VAL_2]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     %[[VAL_5:.*]] = "enzymexla.pointer2memref"(%[[VAL_3]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     enzymexla.xla_wrapper @body (%[[GET_GLOBAL_TEMP_0]], %[[GET_GLOBAL_TEMP_1]], %[[VAL_4]], %[[VAL_5]]) : (memref<i64, 1>, memref<i64, 1>, memref<?xf32>, memref<?xf32>) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @body(%[[VAL_6:.*]]: tensor<i64>, %[[VAL_7:.*]]: tensor<i64>, %[[VAL_8:.*]]: tensor<?xf32>, %[[VAL_9:.*]]: tensor<?xf32>) -> (tensor<i64>, tensor<i64>, tensor<?xf32>, tensor<?xf32>) {
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %[[WHILE_0:.*]]:3 = stablehlo.while(%[[VAL_10:.*]] = %[[VAL_6]], %[[VAL_11:.*]] = %[[VAL_8]], %[[VAL_12:.*]] = %[[VAL_9]]) : tensor<i64>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_10]], %[[VAL_7]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_10]], %[[CONSTANT_4]] : tensor<i64>
// CHECK-NEXT:       %[[ADD_1:.*]] = stablehlo.add %[[VAL_11]], %[[VAL_12]] : tensor<?xf32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[VAL_11]], %[[ADD_1]] : tensor<i64>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_6]], %[[VAL_7]], %[[WHILE_0]]#1, %[[WHILE_0]]#2 : tensor<i64>, tensor<i64>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @body_bound_0 : memref<i64, 1>
// CHECK-NEXT:   enzymexla.temp_alloc "private" @body_bound_1 : memref<i64, 1>
// CHECK-NEXT: }

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

// CHECK-LABEL: module {
// CHECK-NEXT:   llvm.func @affine_static(%[[VAL_0:.*]]: !llvm.ptr) {
// CHECK-NEXT:     %[[VAL_1:.*]] = "enzymexla.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     enzymexla.xla_wrapper @body (%[[VAL_1]]) : (memref<?xf32>) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @body(%[[VAL_2:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = stablehlo.constant dense<-3> : tensor<i64>
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %[[WHILE_0:.*]]:2 = stablehlo.while(%[[VAL_3:.*]] = %[[CONSTANT_0]], %[[VAL_4:.*]] = %[[VAL_2]]) : tensor<i64>, tensor<?xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_3]], %[[CONSTANT_1]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_3]], %[[CONSTANT_2]] : tensor<i64>
// CHECK-NEXT:       %[[ADD_1:.*]] = stablehlo.add %[[VAL_4]], %[[VAL_4]] : tensor<?xf32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[ADD_1]] : tensor<i64>, tensor<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[WHILE_0]]#1 : tensor<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// Keep loops with an IV use, a host memory write, or a result on the host.
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

// CHECK-LABEL: module {
// CHECK-NEXT:   llvm.func @affine_iv_used(%[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: !llvm.ptr) {
// CHECK-NEXT:     %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i64 to index
// CHECK-NEXT:     affine.for %[[VAL_2:.*]] = 0 to %[[INDEX_CAST_0]] {
// CHECK-NEXT:       %[[INDEX_CAST_1:.*]] = arith.index_cast %[[VAL_2]] : index to i64
// CHECK-NEXT:       %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[INDEX_CAST_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:       %[[VAL_3:.*]] = "enzymexla.pointer2memref"(%[[GETELEMENTPTR_0]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:       enzymexla.xla_wrapper @body (%[[VAL_3]]) : (memref<?xf32>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.func @affine_host_effect(%[[VAL_4:.*]]: i64, %[[VAL_5:.*]]: !llvm.ptr, %[[VAL_6:.*]]: !llvm.ptr) {
// CHECK-NEXT:     %[[INDEX_CAST_2:.*]] = arith.index_cast %[[VAL_4]] : i64 to index
// CHECK-NEXT:     affine.for %[[VAL_7:.*]] = 0 to %[[INDEX_CAST_2]] {
// CHECK-NEXT:       llvm.store %[[VAL_4]], %[[VAL_6]] : i64, !llvm.ptr
// CHECK-NEXT:       %[[VAL_8:.*]] = "enzymexla.pointer2memref"(%[[VAL_5]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:       enzymexla.xla_wrapper @body (%[[VAL_8]]) : (memref<?xf32>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.func @affine_iter_arg(%[[VAL_9:.*]]: i64, %[[VAL_10:.*]]: !llvm.ptr) -> i32 {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK-NEXT:     %[[INDEX_CAST_3:.*]] = arith.index_cast %[[VAL_9]] : i64 to index
// CHECK-NEXT:     %[[FOR_0:.*]] = affine.for %[[VAL_11:.*]] = 0 to %[[INDEX_CAST_3]] iter_args(%[[VAL_12:.*]] = %[[CONSTANT_1]]) -> (i32) {
// CHECK-NEXT:       %[[VAL_13:.*]] = "enzymexla.pointer2memref"(%[[VAL_10]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:       enzymexla.xla_wrapper @body (%[[VAL_13]]) : (memref<?xf32>) -> ()
// CHECK-NEXT:       %[[ADDI_0:.*]] = arith.addi %[[VAL_12]], %[[CONSTANT_0]] : i32
// CHECK-NEXT:       affine.yield %[[ADDI_0]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     llvm.return %[[FOR_0]] : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.func @affine_may_alias(%[[VAL_14:.*]]: i64, %[[VAL_15:.*]]: !llvm.ptr, %[[VAL_16:.*]]: !llvm.ptr) {
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 8 : index
// CHECK-NEXT:     %[[ALLOCA_0:.*]] = memref.alloca() : memref<i64>
// CHECK-NEXT:     memref.store %[[VAL_14]], %[[ALLOCA_0]][] : memref<i64>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @pair_body_bound_1 : memref<i64, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_0]], %[[CONSTANT_2]] : memref<i64, 1>, memref<i64>
// CHECK-NEXT:     %[[VAL_17:.*]] = "enzymexla.pointer2memref"(%[[VAL_15]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     %[[VAL_18:.*]] = "enzymexla.pointer2memref"(%[[VAL_16]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     enzymexla.xla_wrapper @pair_body (%[[GET_GLOBAL_TEMP_0]], %[[VAL_17]], %[[VAL_18]]) : (memref<i64, 1>, memref<?xf32>, memref<?xf32>) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @body(%[[VAL_19:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK-NEXT:     %[[ADD_0:.*]] = stablehlo.add %[[VAL_19]], %[[VAL_19]] : tensor<?xf32>
// CHECK-NEXT:     return %[[ADD_0]] : tensor<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @pair_body(%[[VAL_20:.*]]: tensor<i64>, %[[VAL_21:.*]]: tensor<?xf32>, %[[VAL_22:.*]]: tensor<?xf32>) -> (tensor<i64>, tensor<?xf32>, tensor<?xf32>) {
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %[[WHILE_0:.*]]:3 = stablehlo.while(%[[VAL_23:.*]] = %[[CONSTANT_3]], %[[VAL_24:.*]] = %[[VAL_21]], %[[VAL_25:.*]] = %[[VAL_22]]) : tensor<i64>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_23]], %[[VAL_20]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_1:.*]] = stablehlo.add %[[VAL_23]], %[[CONSTANT_4]] : tensor<i64>
// CHECK-NEXT:       %[[ADD_2:.*]] = stablehlo.add %[[VAL_24]], %[[VAL_25]] : tensor<?xf32>
// CHECK-NEXT:       stablehlo.return %[[ADD_1]], %[[VAL_24]], %[[ADD_2]] : tensor<i64>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_20]], %[[WHILE_0]]#1, %[[WHILE_0]]#2 : tensor<i64>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @pair_body_bound_1 : memref<i64, 1>
// CHECK-NEXT: }
