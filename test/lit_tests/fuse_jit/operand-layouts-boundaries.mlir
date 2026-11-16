// RUN: enzymexlamlir-opt --fuse-jit %s | FileCheck %s

module {
  llvm.func @write(%out: !llvm.ptr) {
    %zero = llvm.mlir.constant(0.0 : f64) : f64
    llvm.store %zero, %out : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @update(%read: !llvm.ptr, %out: !llvm.ptr) {
    %value = llvm.load %read : !llvm.ptr -> f64
    llvm.store %value, %out : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @update_two(%a: !llvm.ptr, %b: !llvm.ptr, %out: !llvm.ptr) {
    %a_value = llvm.load %a : !llvm.ptr -> f64
    %b_value = llvm.load %b : !llvm.ptr -> f64
    %value = llvm.fadd %a_value, %b_value : f64
    llvm.store %value, %out : f64, !llvm.ptr
    llvm.return
  }

  // Deduplicating a load-only input must deduplicate its layout too.
  func.func @shared_read_only(%read: tensor<4x4xf64>, %out: tensor<4x4xf64>) -> tensor<4x4xf64> {
    %0 = enzymexla.jit_call @update (%read, %out) <operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]> : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
    %1 = enzymexla.jit_call @update (%read, %0) <operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]> : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
    return %1 : tensor<4x4xf64>
  }
  // CHECK-LABEL: func.func @shared_read_only(
  // CHECK-NEXT: %[[SHARED:.*]] = enzymexla.jit_call @fused__update_update (%{{[^ ,]+}}, %{{[^ )]+}})
  // CHECK-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]
  // CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]
  // CHECK-NEXT: return %[[SHARED]] : tensor<4x4xf64>

  // The consumer requires a different layout for the producer output.
  func.func @internal_conflict(%read: tensor<4x4xf64>, %out: tensor<4x4xf64>) -> tensor<4x4xf64> {
    %0 = enzymexla.jit_call @write (%out) <operand_layouts = [dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<4x4xf64>) -> tensor<4x4xf64>
    %1 = enzymexla.jit_call @write (%0) <operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<4x4xf64>) -> tensor<4x4xf64>
    return %1 : tensor<4x4xf64>
  }
  // CHECK-LABEL: func.func @internal_conflict(
  // CHECK-NEXT: %[[FIRST:.*]] = enzymexla.jit_call @write
  // CHECK-NEXT: %[[SECOND:.*]] = enzymexla.jit_call @write
  // CHECK-NEXT: return %[[SECOND]] : tensor<4x4xf64>

  // The deduplicated read-only input has conflicting layouts across calls.
  func.func @shared_conflict(%read: tensor<4x4xf64>, %out: tensor<4x4xf64>) -> tensor<4x4xf64> {
    %0 = enzymexla.jit_call @update (%read, %out) <operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]> : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
    %1 = enzymexla.jit_call @update (%read, %0) <operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]> : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
    return %1 : tensor<4x4xf64>
  }
  // CHECK-LABEL: func.func @shared_conflict(
  // CHECK-NEXT: %[[FIRST:.*]] = enzymexla.jit_call @update
  // CHECK-NEXT: %[[SECOND:.*]] = enzymexla.jit_call @update
  // CHECK-NEXT: return %[[SECOND]] : tensor<4x4xf64>

  // Repeated load-only inputs within a newly staged call must also agree.
  func.func @staged_conflict(%read: tensor<4x4xf64>, %out: tensor<4x4xf64>) -> tensor<4x4xf64> {
    %0 = enzymexla.jit_call @write (%out) <operand_layouts = [dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<4x4xf64>) -> tensor<4x4xf64>
    %1 = enzymexla.jit_call @update_two (%read, %read, %0) <operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]> : (tensor<4x4xf64>, tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
    return %1 : tensor<4x4xf64>
  }
  // CHECK-LABEL: func.func @staged_conflict(
  // CHECK-NEXT: %[[FIRST:.*]] = enzymexla.jit_call @write
  // CHECK-NEXT: %[[SECOND:.*]] = enzymexla.jit_call @update_two
  // CHECK-NEXT: return %[[SECOND]] : tensor<4x4xf64>

  // Do not infer a layout for a call with unspecified layouts.
  func.func @explicit_then_unspecified(%read: tensor<4x4xf64>, %out: tensor<4x4xf64>) -> tensor<4x4xf64> {
    %0 = enzymexla.jit_call @write (%out) <operand_layouts = [dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<4x4xf64>) -> tensor<4x4xf64>
    %1 = enzymexla.jit_call @write (%0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<4x4xf64>) -> tensor<4x4xf64>
    return %1 : tensor<4x4xf64>
  }
  // CHECK-LABEL: func.func @explicit_then_unspecified(
  // CHECK-NEXT: %[[FIRST:.*]] = enzymexla.jit_call @write
  // CHECK-NEXT: %[[SECOND:.*]] = enzymexla.jit_call @write
  // CHECK-NEXT: return %[[SECOND]] : tensor<4x4xf64>
  // The conservative mixed-layout boundary also applies in reverse.
}
