// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s --match-full-lines --check-prefixes=CHECK,LOOP --implicit-check-not=gpu.alloc --implicit-check-not=gpu.dealloc
// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce --enzyme-hlo-unroll=max-num-iterations=4 | FileCheck %s --match-full-lines --check-prefixes=CHECK,UNROLL

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

// CHECK-LABEL: module {
// CHECK-NEXT:   llvm.func @lift_dynamic_loop(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !llvm.ptr {llvm.noalias}, %[[VAL_4:.*]]: !llvm.ptr {llvm.noalias}) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[ALLOCA_0:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     %[[ALLOCA_1:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     %[[ALLOCA_2:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %[[VAL_0]], %[[ALLOCA_2]][] : memref<i32>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @loop_body_bound_0 : memref<i32, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_2]], %[[CONSTANT_0]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT:     memref.store %[[VAL_1]], %[[ALLOCA_1]][] : memref<i32>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_1:.*]] = enzymexla.get_global_temp @loop_body_bound_1 : memref<i32, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_1]], %[[ALLOCA_1]], %[[CONSTANT_0]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT:     memref.store %[[VAL_2]], %[[ALLOCA_0]][] : memref<i32>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_2:.*]] = enzymexla.get_global_temp @loop_body_bound_2 : memref<i32, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_2]], %[[ALLOCA_0]], %[[CONSTANT_0]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT:     %[[VAL_5:.*]] = "enzymexla.pointer2memref"(%[[VAL_3]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     %[[VAL_6:.*]] = "enzymexla.pointer2memref"(%[[VAL_4]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     enzymexla.xla_wrapper @loop_body (%[[GET_GLOBAL_TEMP_0]], %[[GET_GLOBAL_TEMP_1]], %[[GET_GLOBAL_TEMP_2]], %[[VAL_5]], %[[VAL_6]]) : (memref<i32, 1>, memref<i32, 1>, memref<i32, 1>, memref<?xf32>, memref<?xf32>) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @loop_body(%[[VAL_7:.*]]: tensor<i32>, %[[VAL_8:.*]]: tensor<i32>, %[[VAL_9:.*]]: tensor<i32>, %[[VAL_10:.*]]: tensor<?xf32>, %[[VAL_11:.*]]: tensor<?xf32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<?xf32>, tensor<?xf32>) {
// CHECK-NEXT:     %[[WHILE_0:.*]]:3 = stablehlo.while(%[[VAL_12:.*]] = %[[VAL_7]], %[[VAL_13:.*]] = %[[VAL_10]], %[[VAL_14:.*]] = %[[VAL_11]]) : tensor<i32>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_12]], %[[VAL_8]], UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_12]], %[[VAL_9]] : tensor<i32>
// CHECK-NEXT:       %[[ADD_1:.*]] = stablehlo.add %[[VAL_13]], %[[VAL_14]] : tensor<?xf32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[VAL_13]], %[[ADD_1]] : tensor<i32>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[WHILE_0]]#1, %[[WHILE_0]]#2 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @loop_body_bound_0 : memref<i32, 1>
// CHECK-NEXT:   enzymexla.temp_alloc "private" @loop_body_bound_1 : memref<i32, 1>
// CHECK-NEXT:   enzymexla.temp_alloc "private" @loop_body_bound_2 : memref<i32, 1>
// CHECK-NEXT: }

// -----

module {
  llvm.func @lift_static_loop(%arg0: !llvm.ptr {llvm.noalias},
                             %arg1: !llvm.ptr {llvm.noalias}) {
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c2_i32 = arith.constant 2 : i32
    scf.for %iv = %c1_i32 to %c5_i32 step %c2_i32 : i32 {
      %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<4xf32>
      %1 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<4xf32>
      enzymexla.xla_wrapper @static_loop_body (%0, %1) :
          (memref<4xf32>, memref<4xf32>) -> ()
    }
    llvm.return
  }

  func.func private @static_loop_body(%arg0: tensor<4xf32>,
                                       %arg1: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xf32>
    return %0, %arg1 : tensor<4xf32>, tensor<4xf32>
  }
}

// LOOP-LABEL: module {
// LOOP-NEXT:   llvm.func @lift_static_loop(%[[VAL_0:.*]]: !llvm.ptr {llvm.noalias}, %[[VAL_1:.*]]: !llvm.ptr {llvm.noalias}) {
// LOOP-NEXT:     %[[VAL_2:.*]] = "enzymexla.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr) -> memref<4xf32>
// LOOP-NEXT:     %[[VAL_3:.*]] = "enzymexla.pointer2memref"(%[[VAL_1]]) : (!llvm.ptr) -> memref<4xf32>
// LOOP-NEXT:     enzymexla.xla_wrapper @static_loop_body (%[[VAL_2]], %[[VAL_3]]) : (memref<4xf32>, memref<4xf32>) -> ()
// LOOP-NEXT:     llvm.return
// LOOP-NEXT:   }
// LOOP-NEXT:   func.func private @static_loop_body(%[[VAL_4:.*]]: tensor<4xf32>, %[[VAL_5:.*]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// LOOP-NEXT:     %[[CONSTANT_0:.*]] = stablehlo.constant dense<1> : tensor<i32>
// LOOP-NEXT:     %[[CONSTANT_1:.*]] = stablehlo.constant dense<5> : tensor<i32>
// LOOP-NEXT:     %[[CONSTANT_2:.*]] = stablehlo.constant dense<2> : tensor<i32>
// LOOP-NEXT:     %[[WHILE_0:.*]]:3 = stablehlo.while(%[[VAL_6:.*]] = %[[CONSTANT_0]], %[[VAL_7:.*]] = %[[VAL_4]], %[[VAL_8:.*]] = %[[VAL_5]]) : tensor<i32>, tensor<4xf32>, tensor<4xf32>
// LOOP-NEXT:     cond {
// LOOP-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_6]], %[[CONSTANT_1]], SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
// LOOP-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// LOOP-NEXT:     } do {
// LOOP-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_6]], %[[CONSTANT_2]] : tensor<i32>
// LOOP-NEXT:       %[[SUBTRACT_0:.*]] = stablehlo.subtract %[[VAL_7]], %[[VAL_8]] : tensor<4xf32>
// LOOP-NEXT:       stablehlo.return %[[ADD_0]], %[[SUBTRACT_0]], %[[VAL_8]] : tensor<i32>, tensor<4xf32>, tensor<4xf32>
// LOOP-NEXT:     }
// LOOP-NEXT:     return %[[WHILE_0]]#1, %[[WHILE_0]]#2 : tensor<4xf32>, tensor<4xf32>
// LOOP-NEXT:   }
// LOOP-NEXT: }

// The constant loop has two iterations. Each update uses the previous result.
// UNROLL-LABEL: module {
// UNROLL-NEXT:   llvm.func @lift_static_loop(%[[VAL_0:.*]]: !llvm.ptr {llvm.noalias}, %[[VAL_1:.*]]: !llvm.ptr {llvm.noalias}) {
// UNROLL-NEXT:     %[[VAL_2:.*]] = "enzymexla.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr) -> memref<4xf32>
// UNROLL-NEXT:     %[[VAL_3:.*]] = "enzymexla.pointer2memref"(%[[VAL_1]]) : (!llvm.ptr) -> memref<4xf32>
// UNROLL-NEXT:     enzymexla.xla_wrapper @static_loop_body (%[[VAL_2]], %[[VAL_3]]) : (memref<4xf32>, memref<4xf32>) -> ()
// UNROLL-NEXT:     llvm.return
// UNROLL-NEXT:   }
// UNROLL-NEXT:   func.func private @static_loop_body(%[[VAL_4:.*]]: tensor<4xf32>, %[[VAL_5:.*]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// UNROLL-NEXT:     %[[SUBTRACT_0:.*]] = stablehlo.subtract %[[VAL_4]], %[[VAL_5]] : tensor<4xf32>
// UNROLL-NEXT:     %[[SUBTRACT_1:.*]] = stablehlo.subtract %[[SUBTRACT_0]], %[[VAL_5]] : tensor<4xf32>
// UNROLL-NEXT:     return %[[SUBTRACT_1]], %[[VAL_5]] : tensor<4xf32>, tensor<4xf32>
// UNROLL-NEXT:   }
// UNROLL-NEXT: }

// -----

// Loop lifting reuses the existing wrapper's buffer arguments.
// It needs no additional alias proof.
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

// CHECK-LABEL: module {
// CHECK-NEXT:   llvm.func @lift_without_noalias(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !llvm.ptr, %[[VAL_4:.*]]: !llvm.ptr) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[ALLOCA_0:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     %[[ALLOCA_1:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     %[[ALLOCA_2:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %[[VAL_0]], %[[ALLOCA_2]][] : memref<i32>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @may_alias_loop_body_bound_0 : memref<i32, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_2]], %[[CONSTANT_0]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT:     memref.store %[[VAL_1]], %[[ALLOCA_1]][] : memref<i32>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_1:.*]] = enzymexla.get_global_temp @may_alias_loop_body_bound_1 : memref<i32, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_1]], %[[ALLOCA_1]], %[[CONSTANT_0]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT:     memref.store %[[VAL_2]], %[[ALLOCA_0]][] : memref<i32>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_2:.*]] = enzymexla.get_global_temp @may_alias_loop_body_bound_2 : memref<i32, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_2]], %[[ALLOCA_0]], %[[CONSTANT_0]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT:     %[[VAL_5:.*]] = "enzymexla.pointer2memref"(%[[VAL_3]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     %[[VAL_6:.*]] = "enzymexla.pointer2memref"(%[[VAL_4]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     enzymexla.xla_wrapper @may_alias_loop_body (%[[GET_GLOBAL_TEMP_0]], %[[GET_GLOBAL_TEMP_1]], %[[GET_GLOBAL_TEMP_2]], %[[VAL_5]], %[[VAL_6]]) : (memref<i32, 1>, memref<i32, 1>, memref<i32, 1>, memref<?xf32>, memref<?xf32>) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @may_alias_loop_body(%[[VAL_7:.*]]: tensor<i32>, %[[VAL_8:.*]]: tensor<i32>, %[[VAL_9:.*]]: tensor<i32>, %[[VAL_10:.*]]: tensor<?xf32>, %[[VAL_11:.*]]: tensor<?xf32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<?xf32>, tensor<?xf32>) {
// CHECK-NEXT:     %[[WHILE_0:.*]]:3 = stablehlo.while(%[[VAL_12:.*]] = %[[VAL_7]], %[[VAL_13:.*]] = %[[VAL_10]], %[[VAL_14:.*]] = %[[VAL_11]]) : tensor<i32>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_12]], %[[VAL_8]], SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_12]], %[[VAL_9]] : tensor<i32>
// CHECK-NEXT:       %[[SUBTRACT_0:.*]] = stablehlo.subtract %[[VAL_13]], %[[VAL_14]] : tensor<?xf32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[VAL_13]], %[[SUBTRACT_0]] : tensor<i32>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[WHILE_0]]#1, %[[WHILE_0]]#2 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @may_alias_loop_body_bound_0 : memref<i32, 1>
// CHECK-NEXT:   enzymexla.temp_alloc "private" @may_alias_loop_body_bound_1 : memref<i32, 1>
// CHECK-NEXT:   enzymexla.temp_alloc "private" @may_alias_loop_body_bound_2 : memref<i32, 1>
// CHECK-NEXT: }

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

// CHECK-LABEL: module {
// CHECK-NEXT:   llvm.func @first_call_site(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: !llvm.ptr) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[ALLOCA_0:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %[[VAL_0]], %[[ALLOCA_0]][] : memref<i32>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @shared_body_bound_1 : memref<i32, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_0]], %[[CONSTANT_0]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT:     %[[VAL_2:.*]] = "enzymexla.pointer2memref"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     enzymexla.xla_wrapper @shared_body (%[[GET_GLOBAL_TEMP_0]], %[[VAL_2]]) : (memref<i32, 1>, memref<?xf32>) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.func @second_call_site(%[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: !llvm.ptr) {
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[ALLOCA_1:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %[[VAL_3]], %[[ALLOCA_1]][] : memref<i32>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_1:.*]] = enzymexla.get_global_temp @rxla$megakernel_0_bound_1 : memref<i32, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_1]], %[[ALLOCA_1]], %[[CONSTANT_1]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT:     %[[VAL_5:.*]] = "enzymexla.pointer2memref"(%[[VAL_4]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     enzymexla.xla_wrapper @rxla$megakernel_0 (%[[GET_GLOBAL_TEMP_1]], %[[VAL_5]]) : (memref<i32, 1>, memref<?xf32>) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @shared_body(%[[VAL_6:.*]]: tensor<i32>, %[[VAL_7:.*]]: tensor<?xf32>) -> (tensor<i32>, tensor<?xf32>) {
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %[[WHILE_0:.*]]:2 = stablehlo.while(%[[VAL_8:.*]] = %[[CONSTANT_2]], %[[VAL_9:.*]] = %[[VAL_7]]) : tensor<i32>, tensor<?xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_8]], %[[VAL_6]], SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_8]], %[[CONSTANT_3]] : tensor<i32>
// CHECK-NEXT:       %[[ADD_1:.*]] = stablehlo.add %[[VAL_9]], %[[VAL_9]] : tensor<?xf32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[ADD_1]] : tensor<i32>, tensor<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_6]], %[[WHILE_0]]#1 : tensor<i32>, tensor<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @rxla$megakernel_0(%[[VAL_10:.*]]: tensor<i32>, %[[VAL_11:.*]]: tensor<?xf32>) -> (tensor<i32>, tensor<?xf32>) {
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %[[WHILE_1:.*]]:2 = stablehlo.while(%[[VAL_12:.*]] = %[[CONSTANT_4]], %[[VAL_13:.*]] = %[[VAL_11]]) : tensor<i32>, tensor<?xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_1:.*]] = stablehlo.compare LT, %[[VAL_12]], %[[VAL_10]], SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_1]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_2:.*]] = stablehlo.add %[[VAL_12]], %[[CONSTANT_5]] : tensor<i32>
// CHECK-NEXT:       %[[ADD_3:.*]] = stablehlo.add %[[VAL_13]], %[[VAL_13]] : tensor<?xf32>
// CHECK-NEXT:       stablehlo.return %[[ADD_2]], %[[ADD_3]] : tensor<i32>, tensor<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_10]], %[[WHILE_1]]#1 : tensor<i32>, tensor<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @rxla$megakernel_0_bound_1 : memref<i32, 1>
// CHECK-NEXT:   enzymexla.temp_alloc "private" @shared_body_bound_1 : memref<i32, 1>
// CHECK-NEXT: }

// -----

// Reuse the private function. Move the data attribute with its argument when
// the new bound buffer is inserted before it.
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

// CHECK-LABEL: module {
// CHECK-NEXT:   llvm.func @keep_argument_attributes(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: !llvm.ptr) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[ALLOCA_0:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %[[VAL_0]], %[[ALLOCA_0]][] : memref<i32>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @annotated_body_bound_1 : memref<i32, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_0]], %[[CONSTANT_0]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT:     %[[VAL_2:.*]] = "enzymexla.pointer2memref"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     enzymexla.xla_wrapper @annotated_body (%[[GET_GLOBAL_TEMP_0]], %[[VAL_2]]) : (memref<i32, 1>, memref<?xf32>) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @annotated_body(%[[VAL_3:.*]]: tensor<i32>, %[[VAL_4:.*]]: tensor<?xf32> {test.keep}) -> (tensor<i32>, tensor<?xf32>) {
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %[[WHILE_0:.*]]:2 = stablehlo.while(%[[VAL_5:.*]] = %[[CONSTANT_1]], %[[VAL_6:.*]] = %[[VAL_4]]) : tensor<i32>, tensor<?xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_5]], %[[VAL_3]], SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_5]], %[[CONSTANT_2]] : tensor<i32>
// CHECK-NEXT:       %[[ADD_1:.*]] = stablehlo.add %[[VAL_6]], %[[VAL_6]] : tensor<?xf32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[ADD_1]] : tensor<i32>, tensor<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_3]], %[[WHILE_0]]#1 : tensor<i32>, tensor<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @annotated_body_bound_1 : memref<i32, 1>
// CHECK-NEXT: }
