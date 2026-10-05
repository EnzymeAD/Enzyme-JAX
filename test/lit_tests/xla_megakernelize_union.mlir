// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s

// {a, b} followed by {b, c}: share b's updated state and preserve a and c.
module {
  llvm.func @fuse_overlapping_sets(%a: !llvm.ptr {llvm.noalias},
                                   %b: !llvm.ptr {llvm.noalias},
                                   %c: !llvm.ptr {llvm.noalias}) {
    %ma = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %mb = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
    %mc = "enzymexla.pointer2memref"(%c) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @first (%ma, %mb) : (memref<?xf32>, memref<?xf32>) -> ()
    %mb2 = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @second (%mb2, %mc) : (memref<?xf32>, memref<?xf32>) -> ()
    llvm.return
  }
  func.func private @first(%a: tensor<?xf32>, %b: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %sum = stablehlo.add %a, %b : tensor<?xf32>
    return %a, %sum : tensor<?xf32>, tensor<?xf32>
  }
  func.func private @second(%b: tensor<?xf32>, %c: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %product = stablehlo.multiply %b, %c : tensor<?xf32>
    %difference = stablehlo.subtract %c, %b : tensor<?xf32>
    return %product, %difference : tensor<?xf32>, tensor<?xf32>
  }
}
// CHECK-LABEL: llvm.func @fuse_overlapping_sets
// CHECK: %[[MA:.*]] = "enzymexla.pointer2memref"
// CHECK: %[[MB:.*]] = "enzymexla.pointer2memref"
// CHECK: %[[MC:.*]] = "enzymexla.pointer2memref"
// CHECK: enzymexla.xla_wrapper @rxla$megakernel_0 (%[[MA]], %[[MB]], %[[MC]])
// CHECK-NOT: enzymexla.xla_wrapper
// CHECK: llvm.return
// CHECK-LABEL: func.func private @rxla$megakernel_0(
// CHECK-SAME: %[[A:.*]]: tensor<?xf32>, %[[B:.*]]: tensor<?xf32>, %[[C:.*]]: tensor<?xf32>
// CHECK: %[[SUM:.*]] = stablehlo.add %[[A]], %[[B]]
// CHECK: %[[PRODUCT:.*]] = stablehlo.multiply %[[SUM]], %[[C]]
// CHECK: %[[DIFF:.*]] = stablehlo.subtract %[[C]], %[[SUM]]
// CHECK: return %[[A]], %[[PRODUCT]], %[[DIFF]]

// -----

// The second wrapper uses only a subset; retain the first wrapper's other result.
module {
  llvm.func @fuse_subset(%a: !llvm.ptr {llvm.noalias},
                         %b: !llvm.ptr {llvm.noalias}) {
    %ma = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %mb = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @first (%ma, %mb) : (memref<?xf32>, memref<?xf32>) -> ()
    enzymexla.xla_wrapper @second (%ma) : (memref<?xf32>) -> ()
    llvm.return
  }
  func.func private @first(%a: tensor<?xf32>, %b: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %sum = stablehlo.add %a, %b : tensor<?xf32>
    %difference = stablehlo.subtract %a, %b : tensor<?xf32>
    return %sum, %difference : tensor<?xf32>, tensor<?xf32>
  }
  func.func private @second(%a: tensor<?xf32>) -> tensor<?xf32> {
    %negated = stablehlo.negate %a : tensor<?xf32>
    return %negated : tensor<?xf32>
  }
}
// CHECK-LABEL: llvm.func @fuse_subset
// CHECK: %[[MA:.*]] = "enzymexla.pointer2memref"
// CHECK: %[[MB:.*]] = "enzymexla.pointer2memref"
// CHECK: enzymexla.xla_wrapper @rxla$megakernel_0 (%[[MA]], %[[MB]])
// CHECK-NOT: enzymexla.xla_wrapper
// CHECK: llvm.return
// CHECK-LABEL: func.func private @rxla$megakernel_0(
// CHECK-SAME: %[[A:.*]]: tensor<?xf32>, %[[B:.*]]: tensor<?xf32>
// CHECK: %[[SUM:.*]] = stablehlo.add %[[A]], %[[B]]
// CHECK: %[[DIFF:.*]] = stablehlo.subtract %[[A]], %[[B]]
// CHECK: %[[NEG:.*]] = stablehlo.negate %[[SUM]]
// CHECK: return %[[NEG]], %[[DIFF]]

// -----

// The second wrapper adds a new argument before the shared one. Its argument
// order is mapped into the union instead of assuming a shared prefix.
module {
  llvm.func @fuse_superset(%a: !llvm.ptr {llvm.noalias},
                           %b: !llvm.ptr {llvm.noalias}) {
    %ma = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %mb = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @first (%ma) : (memref<?xf32>) -> ()
    enzymexla.xla_wrapper @second (%mb, %ma) : (memref<?xf32>, memref<?xf32>) -> ()
    llvm.return
  }
  func.func private @first(%a: tensor<?xf32>) -> tensor<?xf32> {
    %negated = stablehlo.negate %a : tensor<?xf32>
    return %negated : tensor<?xf32>
  }
  func.func private @second(%b: tensor<?xf32>, %a: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %sum = stablehlo.add %b, %a : tensor<?xf32>
    return %sum, %a : tensor<?xf32>, tensor<?xf32>
  }
}
// CHECK-LABEL: llvm.func @fuse_superset
// CHECK: %[[MA:.*]] = "enzymexla.pointer2memref"
// CHECK: %[[MB:.*]] = "enzymexla.pointer2memref"
// CHECK: enzymexla.xla_wrapper @rxla$megakernel_0 (%[[MA]], %[[MB]])
// CHECK-NOT: enzymexla.xla_wrapper
// CHECK: llvm.return
// CHECK-LABEL: func.func private @rxla$megakernel_0(
// CHECK-SAME: %[[A:.*]]: tensor<?xf32>, %[[B:.*]]: tensor<?xf32>
// CHECK: %[[NEG:.*]] = stablehlo.negate %[[A]]
// CHECK: %[[SUM:.*]] = stablehlo.add %[[B]], %[[NEG]]
// CHECK: return %[[NEG]], %[[SUM]]

// -----

// Completely disjoint wrappers may also have different element types.
module {
  llvm.func @fuse_disjoint_sets(%a: !llvm.ptr {llvm.noalias},
                                %b: !llvm.ptr {llvm.noalias}) {
    %ma = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %mb = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xi32>
    enzymexla.xla_wrapper @first (%ma) : (memref<?xf32>) -> ()
    enzymexla.xla_wrapper @second (%mb) : (memref<?xi32>) -> ()
    llvm.return
  }
  func.func private @first(%a: tensor<?xf32>) -> tensor<?xf32> {
    %negated = stablehlo.negate %a : tensor<?xf32>
    return %negated : tensor<?xf32>
  }
  func.func private @second(%b: tensor<?xi32>) -> tensor<?xi32> {
    %sum = stablehlo.add %b, %b : tensor<?xi32>
    return %sum : tensor<?xi32>
  }
}
// CHECK-LABEL: llvm.func @fuse_disjoint_sets
// CHECK: %[[MA:.*]] = "enzymexla.pointer2memref"
// CHECK: %[[MB:.*]] = "enzymexla.pointer2memref"
// CHECK: enzymexla.xla_wrapper @rxla$megakernel_0 (%[[MA]], %[[MB]])
// CHECK-NOT: enzymexla.xla_wrapper
// CHECK: llvm.return
// CHECK-LABEL: func.func private @rxla$megakernel_0(
// CHECK-SAME: %[[A:.*]]: tensor<?xf32>, %[[B:.*]]: tensor<?xi32>
// CHECK: %[[NEG:.*]] = stablehlo.negate %[[A]]
// CHECK: %[[SUM:.*]] = stablehlo.add %[[B]], %[[B]]
// CHECK: return %[[NEG]], %[[SUM]] : tensor<?xf32>, tensor<?xi32>

// -----

// Distinct inputs need no alias proof at the StableHLO level.
module {
  llvm.func @fuse_distinct_pointer_values(%a: !llvm.ptr, %b: !llvm.ptr) {
    %ma = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %mb = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @update (%ma) : (memref<?xf32>) -> ()
    enzymexla.xla_wrapper @update (%mb) : (memref<?xf32>) -> ()
    llvm.return
  }
  func.func private @update(%a: tensor<?xf32>) -> tensor<?xf32> {
    %negated = stablehlo.negate %a : tensor<?xf32>
    return %negated : tensor<?xf32>
  }
}
// CHECK-LABEL: llvm.func @fuse_distinct_pointer_values
// CHECK: %[[MA:.*]] = "enzymexla.pointer2memref"
// CHECK: %[[MB:.*]] = "enzymexla.pointer2memref"
// CHECK: enzymexla.xla_wrapper @rxla$megakernel_0 (%[[MA]], %[[MB]])
// CHECK-NOT: enzymexla.xla_wrapper
// CHECK: llvm.return
// CHECK-LABEL: func.func private @rxla$megakernel_0(
// CHECK-SAME: %[[A:.*]]: tensor<?xf32>, %[[B:.*]]: tensor<?xf32>
// CHECK: %[[NA:.*]] = stablehlo.negate %[[A]]
// CHECK: %[[NB:.*]] = stablehlo.negate %[[B]]
// CHECK: return %[[NA]], %[[NB]]

// -----

// Distinct views keep separate tensor state.
module {
  llvm.func @fuse_distinct_views(%a: !llvm.ptr {llvm.noalias}) {
    %offset = llvm.getelementptr %a[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %ma = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %mb = "enzymexla.pointer2memref"(%offset) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @update (%ma) : (memref<?xf32>) -> ()
    enzymexla.xla_wrapper @update (%mb) : (memref<?xf32>) -> ()
    llvm.return
  }
  func.func private @update(%a: tensor<?xf32>) -> tensor<?xf32> {
    %negated = stablehlo.negate %a : tensor<?xf32>
    return %negated : tensor<?xf32>
  }
}
// CHECK-LABEL: llvm.func @fuse_distinct_views(
// CHECK-SAME: %[[BASE:[^:]+]]: !llvm.ptr
// CHECK: %[[OFFSET:[^ ]+]] = llvm.getelementptr %[[BASE]][1] : (!llvm.ptr) -> !llvm.ptr, f32
// CHECK: %[[FIRST:[^ ]+]] = "enzymexla.pointer2memref"(%[[BASE]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK: %[[SECOND:[^ ]+]] = "enzymexla.pointer2memref"(%[[OFFSET]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK: enzymexla.xla_wrapper @rxla$megakernel_0 (%[[FIRST]], %[[SECOND]]) : (memref<?xf32>, memref<?xf32>) -> ()
// CHECK-NOT: enzymexla.xla_wrapper
// CHECK: llvm.return
// CHECK-LABEL: func.func private @rxla$megakernel_0(
// CHECK-SAME: %[[A:[^:]+]]: tensor<?xf32>, %[[B:[^:]+]]: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
// CHECK-NEXT: %[[NEG_A:[^ ]+]] = stablehlo.negate %[[A]] : tensor<?xf32>
// CHECK-NEXT: %[[NEG_B:[^ ]+]] = stablehlo.negate %[[B]] : tensor<?xf32>
// CHECK-NEXT: return %[[NEG_A]], %[[NEG_B]] : tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT: }

// -----

// These subviews have the same type but start four elements apart.
// Keep two inputs and negate each input separately.
module {
  func.func @fuse_distinct_subviews(
      %base: memref<8xf32, strided<[1], offset: ?>>) {
    %first = memref.subview %base[0] [4] [1] :
        memref<8xf32, strided<[1], offset: ?>> to
        memref<4xf32, strided<[1], offset: ?>>
    %second = memref.subview %base[4] [4] [1] :
        memref<8xf32, strided<[1], offset: ?>> to
        memref<4xf32, strided<[1], offset: ?>>
    enzymexla.xla_wrapper @update (%first) :
        (memref<4xf32, strided<[1], offset: ?>>) -> ()
    enzymexla.xla_wrapper @update (%second) :
        (memref<4xf32, strided<[1], offset: ?>>) -> ()
    return
  }
  func.func private @update(%a: tensor<4xf32>) -> tensor<4xf32> {
    %negated = stablehlo.negate %a : tensor<4xf32>
    return %negated : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @fuse_distinct_subviews(
// CHECK-SAME: %[[BASE:[^:]+]]: memref<8xf32, strided<[1], offset: ?>>) {
// CHECK-NEXT: %[[FIRST:[^ ]+]] = memref.subview %[[BASE]][0] [4] [1] : memref<8xf32, strided<[1], offset: ?>> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT: %[[SECOND:[^ ]+]] = memref.subview %[[BASE]][4] [4] [1] : memref<8xf32, strided<[1], offset: ?>> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT: enzymexla.xla_wrapper @rxla$megakernel_0 (%[[FIRST]], %[[SECOND]]) : (memref<4xf32, strided<[1], offset: ?>>, memref<4xf32, strided<[1], offset: ?>>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @rxla$megakernel_0(
// CHECK-SAME: %[[A:[^:]+]]: tensor<4xf32>, %[[B:[^:]+]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT: %[[NEG_A:[^ ]+]] = stablehlo.negate %[[A]] : tensor<4xf32>
// CHECK-NEXT: %[[NEG_B:[^ ]+]] = stablehlo.negate %[[B]] : tensor<4xf32>
// CHECK-NEXT: return %[[NEG_A]], %[[NEG_B]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }

// -----

// Two casts of the same offset pointer refer to the same buffer.
// Pass the first update directly to the second update through one state slot.
module {
  llvm.func @fuse_same_offset_view(%base: !llvm.ptr) {
    %offset = llvm.getelementptr %base[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %first = "enzymexla.pointer2memref"(%offset) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @negate (%first) : (memref<?xf32>) -> ()
    %second = "enzymexla.pointer2memref"(%offset) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @double (%second) : (memref<?xf32>) -> ()
    llvm.return
  }
  func.func private @negate(%a: tensor<?xf32>) -> tensor<?xf32> {
    %negated = stablehlo.negate %a : tensor<?xf32>
    return %negated : tensor<?xf32>
  }
  func.func private @double(%a: tensor<?xf32>) -> tensor<?xf32> {
    %doubled = stablehlo.add %a, %a : tensor<?xf32>
    return %doubled : tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func @fuse_same_offset_view(
// CHECK-SAME: %[[BASE:[^:]+]]: !llvm.ptr) {
// CHECK-NEXT: %[[OFFSET:[^ ]+]] = llvm.getelementptr %[[BASE]][4] : (!llvm.ptr) -> !llvm.ptr, f32
// CHECK-NEXT: %[[BUFFER:[^ ]+]] = "enzymexla.pointer2memref"(%[[OFFSET]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT: enzymexla.xla_wrapper @rxla$megakernel_0 (%[[BUFFER]]) : (memref<?xf32>) -> ()
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @rxla$megakernel_0(
// CHECK-SAME: %[[A:[^:]+]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK-NEXT: %[[NEGATED:[^ ]+]] = stablehlo.negate %[[A]] : tensor<?xf32>
// CHECK-NEXT: %[[DOUBLED:[^ ]+]] = stablehlo.add %[[NEGATED]], %[[NEGATED]] : tensor<?xf32>
// CHECK-NEXT: return %[[DOUBLED]] : tensor<?xf32>
// CHECK-NEXT: }

// -----

// Equal identities must have the same logical buffer types.
module {
  llvm.func @do_not_fuse_mismatched_types(%a: !llvm.ptr {llvm.noalias}) {
    %ma = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %mb = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi32>
    enzymexla.xla_wrapper @first (%ma) : (memref<?xf32>) -> ()
    enzymexla.xla_wrapper @second (%mb) : (memref<?xi32>) -> ()
    llvm.return
  }
  func.func private @first(%a: tensor<?xf32>) -> tensor<?xf32> {
    %negated = stablehlo.negate %a : tensor<?xf32>
    return %negated : tensor<?xf32>
  }
  func.func private @second(%a: tensor<?xi32>) -> tensor<?xi32> {
    %negated = stablehlo.negate %a : tensor<?xi32>
    return %negated : tensor<?xi32>
  }
}
// CHECK-LABEL: llvm.func @do_not_fuse_mismatched_types
// CHECK: enzymexla.xla_wrapper @first
// CHECK: enzymexla.xla_wrapper @second
// CHECK: llvm.return
// CHECK-NOT: @rxla$megakernel

// -----

// Argument/result metadata has no composition rule and must remain on the
// original calls, even for otherwise compatible disjoint buffers.
module {
  llvm.func @do_not_fuse_metadata(%a: !llvm.ptr {llvm.noalias},
                                  %b: !llvm.ptr {llvm.noalias}) {
    %ma = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %mb = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @update (%ma) <{arg_attrs = [{test.tag = "first"}]}> : (memref<?xf32>) -> ()
    enzymexla.xla_wrapper @update (%mb) <{res_attrs = [{test.tag = "second"}]}> : (memref<?xf32>) -> ()
    llvm.return
  }
  func.func private @update(%a: tensor<?xf32>) -> tensor<?xf32> {
    %negated = stablehlo.negate %a : tensor<?xf32>
    return %negated : tensor<?xf32>
  }
}
// CHECK-LABEL: llvm.func @do_not_fuse_metadata
// CHECK: enzymexla.xla_wrapper @update ({{.*}}) <{arg_attrs = [{test.tag = "first"}]}>
// CHECK: enzymexla.xla_wrapper @update ({{.*}}) <{res_attrs = [{test.tag = "second"}]}>
// CHECK: llvm.return
// CHECK-NOT: @rxla$megakernel

// -----

// Metadata on the raised function's arguments/results must not be dropped when
// building a new function signature either.
module {
  llvm.func @do_not_fuse_callee_metadata(%a: !llvm.ptr {llvm.noalias},
                                         %b: !llvm.ptr {llvm.noalias}) {
    %ma = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %mb = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @first (%ma) : (memref<?xf32>) -> ()
    enzymexla.xla_wrapper @second (%mb) : (memref<?xf32>) -> ()
    llvm.return
  }
  func.func private @first(%a: tensor<?xf32> {test.tag = "argument"})
      -> tensor<?xf32> {
    %negated = stablehlo.negate %a : tensor<?xf32>
    return %negated : tensor<?xf32>
  }
  func.func private @second(%a: tensor<?xf32>)
      -> (tensor<?xf32> {test.tag = "result"}) {
    %negated = stablehlo.negate %a : tensor<?xf32>
    return %negated : tensor<?xf32>
  }
}
// CHECK-LABEL: llvm.func @do_not_fuse_callee_metadata
// CHECK: enzymexla.xla_wrapper @first
// CHECK: enzymexla.xla_wrapper @second
// CHECK: llvm.return
// CHECK-NOT: @rxla$megakernel

// -----

// Moving a nested callee's body to the wrapper's module would change the scope
// in which its relative @helper reference is resolved.
module {
  llvm.func @do_not_fuse_nested_callee(%a: !llvm.ptr {llvm.noalias},
                                       %b: !llvm.ptr {llvm.noalias}) {
    %ma = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %mb = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @nested::@update (%ma) : (memref<?xf32>) -> ()
    enzymexla.xla_wrapper @second (%mb) : (memref<?xf32>) -> ()
    llvm.return
  }
  module @nested {
    func.func @update(%a: tensor<?xf32>) -> tensor<?xf32> {
      %result = func.call @helper(%a) : (tensor<?xf32>) -> tensor<?xf32>
      return %result : tensor<?xf32>
    }
    func.func private @helper(%a: tensor<?xf32>) -> tensor<?xf32> {
      %negated = stablehlo.negate %a : tensor<?xf32>
      return %negated : tensor<?xf32>
    }
  }
  func.func private @second(%a: tensor<?xf32>) -> tensor<?xf32> {
    %negated = stablehlo.negate %a : tensor<?xf32>
    return %negated : tensor<?xf32>
  }
}
// CHECK-LABEL: llvm.func @do_not_fuse_nested_callee
// CHECK: enzymexla.xla_wrapper @nested::@update
// CHECK: enzymexla.xla_wrapper @second
// CHECK: llvm.return
// CHECK-NOT: @rxla$megakernel

// -----

// A new input defined between the calls must not move before its definition.
module {
  llvm.func @keep_intervening_input(%a: !llvm.ptr, %b: !llvm.ptr) {
    %ma = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @update (%ma) : (memref<?xf32>) -> ()
    %mb = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @update (%mb) : (memref<?xf32>) -> ()
    llvm.return
  }
  func.func private @update(%a: tensor<?xf32>) -> tensor<?xf32> {
    %negated = stablehlo.negate %a : tensor<?xf32>
    return %negated : tensor<?xf32>
  }
}
// CHECK-LABEL: llvm.func @keep_intervening_input
// CHECK: enzymexla.xla_wrapper @update
// CHECK: "enzymexla.pointer2memref"
// CHECK: enzymexla.xla_wrapper @update
// CHECK: llvm.return
// CHECK-NOT: @rxla$megakernel

// -----

// The first function's empty effect summary must not hide the second function's
// side effect. Preserve unrelated function attributes on the combined callee.
module {
  func.func @fuse_effects(%buffer: memref<4xf32>) {
    enzymexla.xla_wrapper @first (%buffer) : (memref<4xf32>) -> ()
    enzymexla.xla_wrapper @second (%buffer) : (memref<4xf32>) -> ()
    return
  }
  func.func private @first(%a: tensor<4xf32>) -> tensor<4xf32>
      attributes {enzymexla.memory_effects = [], test.tag = "keep"} {
    %negated = stablehlo.negate %a : tensor<4xf32>
    return %negated : tensor<4xf32>
  }
  func.func private @second(%a: tensor<4xf32>) -> tensor<4xf32> {
    %result = stablehlo.custom_call @effect(%a) {has_side_effect = true} : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
// CHECK-LABEL: func.func @fuse_effects
// CHECK: enzymexla.xla_wrapper @[[FUSED_EFFECTS:[^ ]+]] ({{.*}})
// CHECK-NOT: enzymexla.xla_wrapper
// CHECK: return
// CHECK: func.func private @[[FUSED_EFFECTS]](
// CHECK-SAME: %[[INPUT:.*]]: tensor<4xf32>) -> tensor<4xf32> attributes {test.tag = "keep"}
// CHECK: %[[NEGATED:.*]] = stablehlo.negate %[[INPUT]]
// CHECK: %[[RESULT:.*]] = stablehlo.custom_call @effect(%[[NEGATED]]) {has_side_effect = true}
// CHECK: return %[[RESULT]] : tensor<4xf32>
