// RUN: enzymexlamlir-opt --canonicalize %s -split-input-file | FileCheck %s

// The callee never touches %buf, but declares that it is written
// e.g. when we want to preserve the alias of a buffer
module {
  llvm.func @wait(!llvm.ptr)
  llvm.func @declared_write(%req: !llvm.ptr, %buf: !llvm.ptr {enzymexla.memory_effects = ["read", "write"]}) {
    llvm.call @wait(%req) : (!llvm.ptr) -> ()
    llvm.return
  }
  func.func @keep_declared_write(%req: tensor<i64>, %buf: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.jit_call @declared_write (%req, %buf) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]}> : (tensor<i64>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: llvm.func @declared_write
// CHECK-SAME:      (%arg0: !llvm.ptr, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write"]})
// CHECK-LABEL: func.func @keep_declared_write
// CHECK-SAME:      (%[[REQ:.+]]: tensor<i64>, %[[BUF:.+]]: tensor<4xf32>)
// CHECK-NEXT:    %[[R:.+]] = enzymexla.jit_call @declared_write (%[[REQ]], %[[BUF]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]> : (tensor<i64>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %[[R]] : tensor<4xf32>

// -----

// The callee never touches %buf and declares no effects on it, so its result
// is %buf itself and the operand is erased.
module {
  llvm.func @wait(!llvm.ptr)
  llvm.func @undeclared(%req: !llvm.ptr, %buf: !llvm.ptr) {
    llvm.call @wait(%req) : (!llvm.ptr) -> ()
    llvm.return
  }
  func.func @drop_undeclared(%req: tensor<i64>, %buf: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.jit_call @undeclared (%req, %buf) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]}> : (tensor<i64>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: llvm.func @undeclared
// CHECK-SAME:      (%arg0: !llvm.ptr)
// CHECK-LABEL: func.func @drop_undeclared
// CHECK-SAME:      (%[[REQ:.+]]: tensor<i64>, %[[BUF:.+]]: tensor<4xf32>)
// CHECK-NEXT:    enzymexla.jit_call @undeclared (%[[REQ]]) : (tensor<i64>) -> ()
// CHECK-NEXT:    return %[[BUF]] : tensor<4xf32>

// -----

// An empty declaration, as `mark-func-memory-effects` writes for an unused
// argument, declares no effects.
module {
  llvm.func @wait(!llvm.ptr)
  llvm.func @declared_none(%req: !llvm.ptr, %buf: !llvm.ptr {enzymexla.memory_effects = []}) {
    llvm.call @wait(%req) : (!llvm.ptr) -> ()
    llvm.return
  }
  func.func @drop_declared_none(%req: tensor<i64>, %buf: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.jit_call @declared_none (%req, %buf) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]}> : (tensor<i64>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: llvm.func @declared_none
// CHECK-SAME:      (%arg0: !llvm.ptr)
// CHECK-LABEL: func.func @drop_declared_none
// CHECK-SAME:      (%[[REQ:.+]]: tensor<i64>, %[[BUF:.+]]: tensor<4xf32>)
// CHECK-NEXT:    enzymexla.jit_call @declared_none (%[[REQ]]) : (tensor<i64>) -> ()
// CHECK-NEXT:    return %[[BUF]] : tensor<4xf32>

// -----

// A declared read leaves %buf readonly, so its result is %buf itself, but the
// operand is kept as the callee may still read it.
module {
  llvm.func @wait(!llvm.ptr)
  llvm.func @declared_read(%req: !llvm.ptr, %buf: !llvm.ptr {enzymexla.memory_effects = ["read"]}) {
    llvm.call @wait(%req) : (!llvm.ptr) -> ()
    llvm.return
  }
  func.func @forward_declared_read(%req: tensor<i64>, %buf: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.jit_call @declared_read (%req, %buf) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]}> : (tensor<i64>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: llvm.func @declared_read
// CHECK-SAME:      (%arg0: !llvm.ptr, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read"]})
// CHECK-LABEL: func.func @forward_declared_read
// CHECK-SAME:      (%[[REQ:.+]]: tensor<i64>, %[[BUF:.+]]: tensor<4xf32>)
// CHECK-NEXT:    enzymexla.jit_call @declared_read (%[[REQ]], %[[BUF]]) : (tensor<i64>, tensor<4xf32>) -> ()
// CHECK-NEXT:    return %[[BUF]] : tensor<4xf32>

// -----

// As for a jit_call, the region never touches the buffer of %b, but its
// arg_attrs declare that it is written, so its result and alias are kept.
func.func @region_keep_declared_write(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %r:2 = "enzymexla.jit_region"(%a, %b) <{
    arg_attrs = [{}, {enzymexla.memory_effects = ["read", "write"]}],
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>,
      #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>
    ]
  }> ({
  ^bb0(%ma: memref<4xf32>, %mb: memref<4xf32>):
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.0 : f32
    memref.store %cst, %ma[%c0] : memref<4xf32>
    enzymexla.jit_region_yield
  }) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  return %r#0, %r#1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @region_keep_declared_write
// CHECK-SAME:      (%[[A:.+]]: tensor<4xf32>, %[[B:.+]]: tensor<4xf32>)
// CHECK:         %[[R:.+]]:2 = "enzymexla.jit_region"(%[[A]], %[[B]]) <{arg_attrs = [{}, {enzymexla.memory_effects = ["read", "write"]}], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>]}> ({
// CHECK-NEXT:    ^bb0(%{{.+}}: memref<4xf32>, %{{.+}}: memref<4xf32>):
// CHECK:         }) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-NEXT:    return %[[R]]#0, %[[R]]#1 : tensor<4xf32>, tensor<4xf32>
