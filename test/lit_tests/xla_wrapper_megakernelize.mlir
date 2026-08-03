// RUN: enzymexlamlir-opt %s --xla-wrapper-megakernelize --symbol-dce | FileCheck %s

module {
  llvm.func @fuse_ping_pong(%arg0: !llvm.ptr {llvm.noalias},
                            %arg1: !llvm.ptr {llvm.noalias}) {
    %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
    %1 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @update_b (%0, %1) : (memref<?xf32>, memref<?xf32>) -> ()
    %2 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
    %3 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @update_a (%2, %3) : (memref<?xf32>, memref<?xf32>) -> ()
    llvm.return
  }

  func.func private @update_b(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<?xf32>
    return %arg0, %0 : tensor<?xf32>, tensor<?xf32>
  }

  func.func private @update_a(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<?xf32>
    return %arg0, %0 : tensor<?xf32>, tensor<?xf32>
  }

  // Distinct SSA pointer values without noalias remain MayAlias.
  llvm.func @do_not_fuse_may_alias(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
    %1 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @may_update_b (%0, %1) : (memref<?xf32>, memref<?xf32>) -> ()
    %2 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
    %3 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @may_update_a (%2, %3) : (memref<?xf32>, memref<?xf32>) -> ()
    llvm.return
  }

  func.func private @may_update_b(%arg0: tensor<?xf32>,
                                  %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    return %arg0, %arg1 : tensor<?xf32>, tensor<?xf32>
  }

  func.func private @may_update_a(%arg0: tensor<?xf32>,
                                  %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    return %arg0, %arg1 : tensor<?xf32>, tensor<?xf32>
  }

  // Passing one physical buffer as two logical arguments is also rejected,
  // even though the cross-wrapper correspondence itself is unambiguous.
  llvm.func @do_not_fuse_duplicate_buffer(%arg0: !llvm.ptr {llvm.noalias}) {
    %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
    %1 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @duplicate_first (%0, %1) : (memref<?xf32>, memref<?xf32>) -> ()
    %2 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
    %3 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @duplicate_second (%2, %3) : (memref<?xf32>, memref<?xf32>) -> ()
    llvm.return
  }

  func.func private @duplicate_first(%arg0: tensor<?xf32>,
                                     %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    return %arg0, %arg1 : tensor<?xf32>, tensor<?xf32>
  }

  func.func private @duplicate_second(%arg0: tensor<?xf32>,
                                      %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    return %arg0, %arg1 : tensor<?xf32>, tensor<?xf32>
  }

  // Partial overlap is not enough to prove a safe buffer-threading order.
  llvm.func @do_not_fuse_partial_overlap(
      %arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias},
      %arg2: !llvm.ptr {llvm.noalias}) {
    %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
    %1 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @partial_first (%0, %1) : (memref<?xf32>, memref<?xf32>) -> ()
    %2 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
    %3 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @partial_second (%2, %3) : (memref<?xf32>, memref<?xf32>) -> ()
    llvm.return
  }

  func.func private @partial_first(%arg0: tensor<?xf32>,
                                   %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    return %arg0, %arg1 : tensor<?xf32>, tensor<?xf32>
  }

  func.func private @partial_second(%arg0: tensor<?xf32>,
                                    %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    return %arg0, %arg1 : tensor<?xf32>, tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func @fuse_ping_pong
// CHECK:         %[[A:.*]] = "enzymexla.pointer2memref"
// CHECK:         %[[B:.*]] = "enzymexla.pointer2memref"
// CHECK:         enzymexla.xla_wrapper @rxla$megakernel_0 (%[[A]], %[[B]])
// CHECK-NOT:     enzymexla.xla_wrapper
// CHECK:         llvm.return

// CHECK-NOT:     func.func private @update_b
// CHECK-NOT:     func.func private @update_a

// CHECK-LABEL: llvm.func @do_not_fuse_may_alias
// CHECK:         enzymexla.xla_wrapper @may_update_b
// CHECK:         enzymexla.xla_wrapper @may_update_a

// CHECK-LABEL: llvm.func @do_not_fuse_duplicate_buffer
// CHECK:         enzymexla.xla_wrapper @duplicate_first
// CHECK:         enzymexla.xla_wrapper @duplicate_second

// CHECK-LABEL: llvm.func @do_not_fuse_partial_overlap
// CHECK:         enzymexla.xla_wrapper @partial_first
// CHECK:         enzymexla.xla_wrapper @partial_second

// CHECK-LABEL: func.func private @rxla$megakernel_0(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>
// CHECK:         %[[UPDATED_B:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]]
// CHECK:         %[[UPDATED_A:.*]] = stablehlo.multiply %[[UPDATED_B]], %[[ARG0]]
// CHECK:         return %[[UPDATED_A]], %[[UPDATED_B]]
