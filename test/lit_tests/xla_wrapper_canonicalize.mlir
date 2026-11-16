// RUN: enzymexlamlir-opt %s --split-input-file --canonicalize --symbol-dce | FileCheck %s

// For buffer inputs, the function result at each position replaces that buffer.
// Remove a position only if the function returns its argument unchanged there
// and has no other use of the argument.
//
// One wrapper is the only user of this private function. Remove spare from
// both signatures. Keep the original function name and body.
// Before: wrapper @negate(data, spare); negate(data, spare) -> (-data, spare)
// After:  wrapper @negate(data); negate(data) -> -data
module {
  func.func @single_private_call(%data: memref<4xf32>, %spare: memref<4xf32>) {
    enzymexla.xla_wrapper @negate (%data, %spare) :
        (memref<4xf32>, memref<4xf32>) -> ()
    return
  }
  func.func private @negate(%data: tensor<4xf32>, %spare: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %negative = stablehlo.negate %data : tensor<4xf32>
    return %negative, %spare : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @single_private_call(
// CHECK-SAME: %[[DATA:[^:]+]]: memref<4xf32>, %[[SPARE:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @negate (%[[DATA]]) : (memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @negate(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[NEGATIVE:[^ ]+]] = stablehlo.negate %[[DATA]] : tensor<4xf32>
// CHECK-NEXT: return %[[NEGATIVE]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

// Both users are wrappers. They pass different buffers, but they can remove
// the same unused position. Update both calls and the original private function.
// Before: negate(data, spare) -> (-data, spare)
// After:  negate(data) -> -data
module {
  func.func @first_private_call(%data: memref<4xf32>, %spare: memref<4xf32>) {
    enzymexla.xla_wrapper @negate (%data, %spare) :
        (memref<4xf32>, memref<4xf32>) -> ()
    return
  }
  func.func @second_private_call(%other: memref<4xf32>, %other_spare: memref<4xf32>) {
    enzymexla.xla_wrapper @negate (%other, %other_spare) :
        (memref<4xf32>, memref<4xf32>) -> ()
    return
  }
  func.func private @negate(%data: tensor<4xf32>, %spare: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %negative = stablehlo.negate %data : tensor<4xf32>
    return %negative, %spare : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @first_private_call(
// CHECK-SAME: %[[DATA:[^:]+]]: memref<4xf32>, %[[SPARE:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @negate (%[[DATA]]) : (memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func @second_private_call(
// CHECK-SAME: %[[OTHER:[^:]+]]: memref<4xf32>, %[[OTHER_SPARE:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @negate (%[[OTHER]]) : (memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @negate(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[NEGATIVE:[^ ]+]] = stablehlo.negate %[[DATA]] : tensor<4xf32>
// CHECK-NEXT: return %[[NEGATIVE]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

// This public function can have callers outside the module. Keep its signature
// even though only one wrapper is visible. Give that wrapper a reduced clone.
// Before: wrapper @negate(data, spare)
// After:  wrapper @negate_without_unused(data)
module {
  func.func @single_public_call(%data: memref<4xf32>, %spare: memref<4xf32>) {
    enzymexla.xla_wrapper @negate (%data, %spare) :
        (memref<4xf32>, memref<4xf32>) -> ()
    return
  }
  func.func @negate(%data: tensor<4xf32>, %spare: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %negative = stablehlo.negate %data : tensor<4xf32>
    return %negative, %spare : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @single_public_call(
// CHECK-SAME: %[[DATA:[^:]+]]: memref<4xf32>, %[[SPARE:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @negate_without_unused (%[[DATA]]) : (memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func @negate(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<4xf32>, %[[SPARE:[^:]+]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT: %[[NEGATIVE:[^ ]+]] = stablehlo.negate %[[DATA]] : tensor<4xf32>
// CHECK-NEXT: return %[[NEGATIVE]], %[[SPARE]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @negate_without_unused(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[NEGATIVE:[^ ]+]] = stablehlo.negate %[[DATA]] : tensor<4xf32>
// CHECK-NEXT: return %[[NEGATIVE]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

// One wrapper is not the only symbol use: an ordinary call also needs the
// private function. Keep that call and the original signature unchanged.
// Before: wrapper @negate(data, spare)
// After:  wrapper @negate_without_unused(data)
module {
  func.func @wrapper_beside_direct_call(%data: memref<4xf32>, %spare: memref<4xf32>) {
    enzymexla.xla_wrapper @negate (%data, %spare) :
        (memref<4xf32>, memref<4xf32>) -> ()
    return
  }
  func.func @keep_direct_call(%data: tensor<4xf32>, %spare: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %results:2 = call @negate(%data, %spare) :
        (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    return %results#0, %results#1 : tensor<4xf32>, tensor<4xf32>
  }
  func.func private @negate(%data: tensor<4xf32>, %spare: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %negative = stablehlo.negate %data : tensor<4xf32>
    return %negative, %spare : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @wrapper_beside_direct_call(
// CHECK-SAME: %[[DATA:[^:]+]]: memref<4xf32>, %[[SPARE:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @negate_without_unused (%[[DATA]]) : (memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func @keep_direct_call(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<4xf32>, %[[SPARE:[^:]+]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT: %[[RESULTS:[^:]+]]:2 = call @negate(%[[DATA]], %[[SPARE]]) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-NEXT: return %[[RESULTS]]#0, %[[RESULTS]]#1 : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @negate(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<4xf32>, %[[SPARE:[^:]+]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT: %[[NEGATIVE:[^ ]+]] = stablehlo.negate %[[DATA]] : tensor<4xf32>
// CHECK-NEXT: return %[[NEGATIVE]], %[[SPARE]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @negate_without_unused(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[NEGATIVE:[^ ]+]] = stablehlo.negate %[[DATA]] : tensor<4xf32>
// CHECK-NEXT: return %[[NEGATIVE]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

// Remove the middle buffer: no operation reads or changes spare.
// Before: wrapper @scale_data(data, spare, scale)
//         scale_data(data, spare, scale) -> (data * scale, spare, scale)
// After:  wrapper @scale_data(data, scale)
//         scale_data(data, scale) -> (data * scale, scale)
//
// Keep scale: the multiplication reads it, even though it is returned unchanged.
// Update all four wrappers and the private function. Remove the attributes
// for spare from the wrapper that has argument attributes.
module {
  func.func @drop_unused_middle_buffer(%data: memref<4xf32>,
                                      %spare: memref<4xf32>,
                                      %scale: memref<4xf32>) {
    enzymexla.xla_wrapper @scale_data (%data, %spare, %scale) :
        (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return
  }
  func.func @share_reduced_function(%data: memref<4xf32>,
                                   %spare: memref<4xf32>,
                                   %scale: memref<4xf32>) {
    enzymexla.xla_wrapper @scale_data (%data, %spare, %scale) :
        (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    enzymexla.xla_wrapper @scale_data (%data, %spare, %scale) :
        (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    enzymexla.xla_wrapper @scale_data (%data, %spare, %scale)
        <{arg_attrs = [{}, {test.keep}, {}]}> :
        (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return
  }
  func.func private @scale_data(%data: tensor<4xf32>,
                               %spare: tensor<4xf32>,
                               %scale: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %scaled = stablehlo.multiply %data, %scale : tensor<4xf32>
    return %scaled, %spare, %scale : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}

// The wrapper passes data and scale, in that order. It no longer passes spare.
// CHECK-LABEL: func.func @drop_unused_middle_buffer(
// CHECK-SAME: %[[DATA:[^:]+]]: memref<4xf32>,
// CHECK-SAME: %[[SPARE:[^:]+]]: memref<4xf32>,
// CHECK-SAME: %[[SCALE:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @scale_data (%[[DATA]], %[[SCALE]]) : (memref<4xf32>, memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

// All wrappers now use the same reduced function.
// CHECK-LABEL: func.func @share_reduced_function(
// CHECK-SAME: %[[DATA:[^:]+]]: memref<4xf32>,
// CHECK-SAME: %[[SPARE:[^:]+]]: memref<4xf32>,
// CHECK-SAME: %[[SCALE:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @scale_data (%[[DATA]], %[[SCALE]]) : (memref<4xf32>, memref<4xf32>) -> ()
// CHECK-NEXT: enzymexla.xla_wrapper @scale_data (%[[DATA]], %[[SCALE]]) : (memref<4xf32>, memref<4xf32>) -> ()
// CHECK-NEXT: enzymexla.xla_wrapper @scale_data (%[[DATA]], %[[SCALE]]) <arg_attrs = [{}, {}]> : (memref<4xf32>, memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

// The private function has two arguments and two results.
// CHECK-LABEL: func.func private @scale_data(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<4xf32>,
// CHECK-SAME: %[[SCALE:[^:]+]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT: %[[SCALED:[^ ]+]] = stablehlo.multiply %[[DATA]], %[[SCALE]] : tensor<4xf32>
// CHECK-NEXT: return %[[SCALED]], %[[SCALE]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

// Keep a read-only input. Returning source unchanged does not make it unused:
// the addition reads source to compute the new destination.
// Before and after: add_to_destination(source, destination)
//                  -> (source, source + destination)
module {
  func.func @keep_readonly_source(%source: memref<4xf32>, %destination: memref<4xf32>) {
    enzymexla.xla_wrapper @add_to_destination (%source, %destination) :
        (memref<4xf32>, memref<4xf32>) -> ()
    return
  }
  func.func private @add_to_destination(%source: tensor<4xf32>, %destination: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %sum = stablehlo.add %source, %destination : tensor<4xf32>
    return %source, %sum : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @keep_readonly_source(
// CHECK-SAME: %[[SOURCE:[^:]+]]: memref<4xf32>, %[[DEST:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @add_to_destination (%[[SOURCE]], %[[DEST]]) : (memref<4xf32>, memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @add_to_destination(
// CHECK-SAME: %[[SOURCE:[^:]+]]: tensor<4xf32>, %[[DEST:[^:]+]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT: %[[SUM:[^ ]+]] = stablehlo.add %[[SOURCE]], %[[DEST]] : tensor<4xf32>
// CHECK-NEXT: return %[[SOURCE]], %[[SUM]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }

// -----

// Keep a buffer that receives a new value. The old destination is never read,
// but the second result writes source into it. Removing this position loses
// that write. The first input also remains: two result positions use it.
// Before and after: copy_to_destination(source, destination) -> (source, source)
module {
  func.func @keep_written_destination(%source: memref<4xf32>, %destination: memref<4xf32>) {
    enzymexla.xla_wrapper @copy_to_destination (%source, %destination) :
        (memref<4xf32>, memref<4xf32>) -> ()
    return
  }
  func.func private @copy_to_destination(%source: tensor<4xf32>, %destination: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    return %source, %source : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @keep_written_destination(
// CHECK-SAME: %[[SOURCE:[^:]+]]: memref<4xf32>, %[[DEST:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @copy_to_destination (%[[SOURCE]], %[[DEST]]) : (memref<4xf32>, memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @copy_to_destination(
// CHECK-SAME: %[[SOURCE:[^:]+]]: tensor<4xf32>, %[[DEST:[^:]+]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT: return %[[SOURCE]], %[[SOURCE]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }

// -----

// Remove every buffer, but keep the call and its side effect.
// Before: wrapper @tick(unused); tick(unused) runs side_effect and returns unused.
// After:  wrapper @tick(); tick() runs side_effect.
module {
  func.func @keep_effect(%unused: memref<4xf32>) {
    enzymexla.xla_wrapper @tick (%unused) : (memref<4xf32>) -> ()
    return
  }
  func.func private @tick(%a: tensor<4xf32>) -> tensor<4xf32> {
    stablehlo.custom_call @side_effect() {has_side_effect = true} : () -> ()
    return %a : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @keep_effect(
// CHECK-SAME: %[[UNUSED:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @tick () : () -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @tick() {
// CHECK-NEXT: stablehlo.custom_call @side_effect() {has_side_effect = true} : () -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

// -----

// Remove the only buffer and its attributes from each call and function.
// The specialized function does not use its scalar input, so remove it too.
module {
  func.func @remove_unused_metadata(%buffer: memref<4xf32>, %bound: i64) {
    enzymexla.xla_wrapper @identity (%buffer) <{arg_attrs = [{test.keep}]}> : (memref<4xf32>) -> ()
    enzymexla.xla_wrapper @identity (%buffer) <{res_attrs = [{test.keep}]}> : (memref<4xf32>) -> ()
    enzymexla.xla_wrapper @arg_metadata (%buffer) : (memref<4xf32>) -> ()
    enzymexla.xla_wrapper @result_metadata (%buffer) : (memref<4xf32>) -> ()
    enzymexla.xla_wrapper @specialized (%buffer, %bound) <{num_specialized = 1 : i64}> : (memref<4xf32>, i64) -> ()
    return
  }
  func.func private @identity(%a: tensor<4xf32>) -> tensor<4xf32> {
    return %a : tensor<4xf32>
  }
  func.func private @arg_metadata(%a: tensor<4xf32> {test.keep}) -> tensor<4xf32> {
    return %a : tensor<4xf32>
  }
  func.func private @result_metadata(%a: tensor<4xf32>) -> (tensor<4xf32> {test.keep}) {
    return %a : tensor<4xf32>
  }
  func.func private @specialized(%a: tensor<4xf32>, %bound: tensor<i64>) -> tensor<4xf32> {
    return %a : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @remove_unused_metadata(
// CHECK-SAME: %[[BUFFER:[^:]+]]: memref<4xf32>, %[[BOUND:[^:]+]]: i64) {
// CHECK-NEXT: enzymexla.xla_wrapper @identity () <arg_attrs = []> : () -> ()
// CHECK-NEXT: enzymexla.xla_wrapper @identity () <res_attrs = []> : () -> ()
// CHECK-NEXT: enzymexla.xla_wrapper @arg_metadata () : () -> ()
// CHECK-NEXT: enzymexla.xla_wrapper @result_metadata () : () -> ()
// CHECK-NEXT: enzymexla.xla_wrapper @specialized () : () -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

// -----

// Keep the nested callee unchanged. Moving its clone into another symbol table
// could change which symbols its body references.
module {
  func.func @keep_nested_callee(%buffer: memref<4xf32>) {
    enzymexla.xla_wrapper @nested::@identity (%buffer) : (memref<4xf32>) -> ()
    return
  }
  module @nested {
    func.func @identity(%a: tensor<4xf32>) -> tensor<4xf32> {
      return %a : tensor<4xf32>
    }
  }
}

// CHECK-LABEL: func.func @keep_nested_callee(
// CHECK-SAME: %[[BUFFER:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @nested::@identity (%[[BUFFER]]) : (memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: module @nested {
// CHECK-NEXT: func.func @identity(%[[BUFFER:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: return %[[BUFFER]] : tensor<4xf32>
// CHECK-NEXT: }

// -----

// Both inputs point to the same buffer. Both results write the same value.
// Keep one input and one result. Use its tensor twice in the addition.
// Before: wrapper @pair(a, a); pair(a, b) -> (a + b, a + b)
// After:  wrapper @pair_without_duplicates(a)
//         pair_without_duplicates(a) -> a + a
// Keep the original function for the caller that passes distinct buffers.
module {
  llvm.func @deduplicate_inputs(%a: !llvm.ptr) {
    %first = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %second = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @pair (%first, %second) :
        (memref<?xf32>, memref<?xf32>) -> ()
    llvm.return
  }
  llvm.func @keep_other_call(%a: !llvm.ptr, %b: !llvm.ptr) {
    %first = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %second = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @pair (%first, %second) :
        (memref<?xf32>, memref<?xf32>) -> ()
    llvm.return
  }
  func.func private @pair(%a: tensor<?xf32>, %b: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %sum = stablehlo.add %a, %b : tensor<?xf32>
    return %sum, %sum : tensor<?xf32>, tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func @deduplicate_inputs(
// CHECK-SAME: %[[A:[^:]+]]: !llvm.ptr) {
// CHECK-NEXT: %[[BUFFER:[^ ]+]] = "enzymexla.pointer2memref"(%[[A]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT: enzymexla.xla_wrapper @pair_without_duplicates (%[[BUFFER]]) : (memref<?xf32>) -> ()
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }

// CHECK-LABEL: llvm.func @keep_other_call(
// CHECK-SAME: %[[A:[^:]+]]: !llvm.ptr, %[[B:[^:]+]]: !llvm.ptr) {
// CHECK-NEXT: %[[FIRST:[^ ]+]] = "enzymexla.pointer2memref"(%[[A]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT: %[[SECOND:[^ ]+]] = "enzymexla.pointer2memref"(%[[B]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT: enzymexla.xla_wrapper @pair (%[[FIRST]], %[[SECOND]]) : (memref<?xf32>, memref<?xf32>) -> ()
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }

// CHECK-LABEL: func.func private @pair(
// CHECK-SAME: %[[A:[^:]+]]: tensor<?xf32>, %[[B:[^:]+]]: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
// CHECK-NEXT: %[[SUM:[^ ]+]] = stablehlo.add %[[A]], %[[B]] : tensor<?xf32>
// CHECK-NEXT: return %[[SUM]], %[[SUM]] : tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT: }

// CHECK-LABEL: func.func private @pair_without_duplicates(
// CHECK-SAME: %[[A:[^:]+]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK-NEXT: %[[SUM:[^ ]+]] = stablehlo.add %[[A]], %[[A]] : tensor<?xf32>
// CHECK-NEXT: return %[[SUM]] : tensor<?xf32>
// CHECK-NEXT: }

// -----

// Keep duplicate inputs when their results differ. Removing either result
// would remove one of the two different writes to the buffer.
// Before and after: wrapper @different_results(a, a)
//                   different_results(a, b) -> (a + b, -(a + b))
module {
  func.func @keep_conflicting_results(%buffer: memref<4xf32>) {
    enzymexla.xla_wrapper @different_results (%buffer, %buffer) :
        (memref<4xf32>, memref<4xf32>) -> ()
    return
  }
  func.func private @different_results(%a: tensor<4xf32>, %b: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %sum = stablehlo.add %a, %b : tensor<4xf32>
    %negative = stablehlo.negate %sum : tensor<4xf32>
    return %sum, %negative : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @keep_conflicting_results(
// CHECK-SAME: %[[BUFFER:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @different_results (%[[BUFFER]], %[[BUFFER]]) : (memref<4xf32>, memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @different_results(
// CHECK-SAME: %[[A:[^:]+]]: tensor<4xf32>, %[[B:[^:]+]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT: %[[SUM:[^ ]+]] = stablehlo.add %[[A]], %[[B]] : tensor<4xf32>
// CHECK-NEXT: %[[NEGATIVE:[^ ]+]] = stablehlo.negate %[[SUM]] : tensor<4xf32>
// CHECK-NEXT: return %[[SUM]], %[[NEGATIVE]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

// The first two results are different arguments for the same input buffer.
// They become the same value when those arguments are merged. The addition
// reads both arguments, so unused-input removal cannot remove either one.
// Before: wrapper @read_pair(a, a, destination)
//         read_pair(a, b, destination) -> (a, b, a + b)
// After:  wrapper @read_pair_without_duplicates(a, destination)
//         read_pair_without_duplicates(a, destination) -> (a, a + a)
module {
  func.func @merge_returned_arguments(%source: memref<4xf32>,
                                     %destination: memref<4xf32>) {
    enzymexla.xla_wrapper @read_pair (%source, %source, %destination) :
        (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return
  }
  func.func private @read_pair(%a: tensor<4xf32>, %b: tensor<4xf32>,
                               %destination: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %sum = stablehlo.add %a, %b : tensor<4xf32>
    return %a, %b, %sum : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @merge_returned_arguments(
// CHECK-SAME: %[[SOURCE:[^:]+]]: memref<4xf32>, %[[DESTINATION:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @read_pair_without_duplicates (%[[SOURCE]], %[[DESTINATION]]) : (memref<4xf32>, memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @read_pair_without_duplicates(
// CHECK-SAME: %[[SOURCE:[^:]+]]: tensor<4xf32>, %[[DESTINATION:[^:]+]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT: %[[SUM:[^ ]+]] = stablehlo.add %[[SOURCE]], %[[SOURCE]] : tensor<4xf32>
// CHECK-NEXT: return %[[SOURCE]], %[[SUM]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }
