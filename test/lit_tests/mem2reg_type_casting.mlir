// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s


// Same type – forwarded with no cast at all

llvm.func @forward_same_type(%val: i32) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %al : i32, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> i32
  llvm.return %0 : i32
}

// CHECK: llvm.func @forward_same_type(%[[VAL:.*]]: i32) -> i32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: llvm.return %[[VAL]] : i32
// CHECK-NEXT: }

// -----

// IntegerType (i64) stored, LLVMPointerType loaded

llvm.func @forward_int_to_ptr(%val: i64) -> !llvm.ptr {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x i64 : (i32) -> !llvm.ptr
  llvm.store %val, %al : i64, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK: llvm.func @forward_int_to_ptr(%[[VAL:.*]]: i64) -> !llvm.ptr {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[ITP:.*]] = llvm.inttoptr %[[VAL]] : i64 to !llvm.ptr
// CHECK-NEXT: llvm.return %[[ITP]] : !llvm.ptr
// CHECK-NEXT: }

// -----

// LLVMPointerType stored, IntegerType (i64) loaded

llvm.func @forward_ptr_to_int(%val: !llvm.ptr) -> i64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
  llvm.store %val, %al : !llvm.ptr, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> i64
  llvm.return %0 : i64
}

// CHECK: llvm.func @forward_ptr_to_int(%[[VAL:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[PTI:.*]] = llvm.ptrtoint %[[VAL]] : !llvm.ptr to i64
// CHECK-NEXT: llvm.return %[[PTI]] : i64
// CHECK-NEXT: }

// -----

// FloatType (f32) stored, IntegerType (i32) loaded

llvm.func @forward_float_to_int(%val: f32) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x f32 : (i32) -> !llvm.ptr
  llvm.store %val, %al : f32, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> i32
  llvm.return %0 : i32
}

// CHECK: llvm.func @forward_float_to_int(%[[VAL:.*]]: f32) -> i32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BC:.*]] = arith.bitcast %[[VAL]] : f32 to i32
// CHECK-NEXT: llvm.return %[[BC]] : i32
// CHECK-NEXT: }

// -----

// IntegerType (i32) stored, FloatType (f32) loaded

llvm.func @forward_int_to_float(%val: i32) -> f32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %al : i32, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> f32
  llvm.return %0 : f32
}

// CHECK: llvm.func @forward_int_to_float(%[[VAL:.*]]: i32) -> f32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BC:.*]] = arith.bitcast %[[VAL]] : i32 to f32
// CHECK-NEXT: llvm.return %[[BC]] : f32
// CHECK-NEXT: }

// -----

// Scalar (i32) stored, single-field struct { i32 } loaded

llvm.func @forward_scalar_to_single_field_struct(%val: i32) -> !llvm.struct<(i32)> {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %al : i32, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> !llvm.struct<(i32)>
  llvm.return %0 : !llvm.struct<(i32)>
}

// CHECK: llvm.func @forward_scalar_to_single_field_struct(%[[VAL:.*]]: i32) -> !llvm.struct<(i32)> {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[UD:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK-NEXT: %[[IV:.*]] = llvm.insertvalue %[[VAL]], %[[UD]][0] : !llvm.struct<(i32)> 
// CHECK-NEXT: llvm.return %[[IV]] : !llvm.struct<(i32)>
// CHECK-NEXT: }

// -----

// Single-field struct { i32 } stored, scalar i32 loaded

llvm.func @forward_single_field_struct_to_scalar(%val: !llvm.struct<(i32)>) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x !llvm.struct<(i32)> : (i32) -> !llvm.ptr
  llvm.store %val, %al : !llvm.struct<(i32)>, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> i32
  llvm.return %0 : i32
}

// CHECK: llvm.func @forward_single_field_struct_to_scalar(%[[VAL:.*]]: !llvm.struct<(i32)>) -> i32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EV:.*]] = llvm.extractvalue %[[VAL]][0] : !llvm.struct<(i32)> 
// CHECK-NEXT: llvm.return %[[EV]] : i32
// CHECK-NEXT: }

// -----

// Nested struct { struct { f32 } } stored, f32 loaded

llvm.func @forward_nested_struct_to_scalar(%val: !llvm.struct<(struct<(f32)>)>) -> f32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x !llvm.struct<(struct<(f32)>)> : (i32) -> !llvm.ptr
  llvm.store %val, %al : !llvm.struct<(struct<(f32)>)>, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> f32
  llvm.return %0 : f32
}

// CHECK: llvm.func @forward_nested_struct_to_scalar(%[[VAL:.*]]: !llvm.struct<(struct<(f32)>)>) -> f32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EV0:.*]] = llvm.extractvalue %[[VAL]][0] : !llvm.struct<(struct<(f32)>)> 
// CHECK-NEXT: %[[EV1:.*]] = llvm.extractvalue %[[EV0]][0] : !llvm.struct<(f32)> 
// CHECK-NEXT: llvm.return %[[EV1]] : f32
// CHECK-NEXT: }

// -----

// Uniform two-field struct { f32, f32 } stored, vector<2xf32> loaded

llvm.func @forward_uniform_struct_to_vector(%val: !llvm.struct<(f32, f32)>) -> vector<2xf32> {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x !llvm.struct<(f32, f32)> : (i32) -> !llvm.ptr
  llvm.store %val, %al : !llvm.struct<(f32, f32)>, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> vector<2xf32>
  llvm.return %0 : vector<2xf32>
}

// CHECK: llvm.func @forward_uniform_struct_to_vector(%[[VAL:.*]]: !llvm.struct<(f32, f32)>) -> vector<2xf32> {
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EX0:.*]] = llvm.extractvalue %[[VAL]][0] : !llvm.struct<(f32, f32)>
// CHECK-NEXT: %[[EX1:.*]] = llvm.extractvalue %[[VAL]][1] : !llvm.struct<(f32, f32)>
// CHECK-NEXT: %[[PSN:.*]] = llvm.mlir.poison : vector<2xf32>
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[V0:.*]] = llvm.insertelement %[[EX0]], %[[PSN]][%[[C0]] : i32] : vector<2xf32>
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[V1:.*]] = llvm.insertelement %[[EX1]], %[[V0]][%[[C1]] : i32] : vector<2xf32>
// CHECK-NEXT: llvm.return %[[V1]] : vector<2xf32>
// CHECK-NEXT: }

// -----

// LLVM array [2 x f32] stored, vector<2xf32> loaded

llvm.func @forward_array_to_vector(%val: !llvm.array<2 x f32>) -> vector<2xf32> {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x !llvm.array<2 x f32> : (i32) -> !llvm.ptr
  llvm.store %val, %al : !llvm.array<2 x f32>, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> vector<2xf32>
  llvm.return %0 : vector<2xf32>
}

// CHECK: llvm.func @forward_array_to_vector(%[[VAL:.*]]: !llvm.array<2 x f32>) -> vector<2xf32> {
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EV0:.*]] = llvm.extractvalue %[[VAL]][0] : !llvm.array<2 x f32> 
// CHECK-NEXT: %[[EV1:.*]] = llvm.extractvalue %[[VAL]][1] : !llvm.array<2 x f32> 
// CHECK-NEXT: %[[PSN:.*]] = llvm.mlir.poison : vector<2xf32>
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[V0:.*]] = llvm.insertelement %[[EV0]], %[[PSN]][%[[C0]] : i32] : vector<2xf32>
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[V1:.*]] = llvm.insertelement %[[EV1]], %[[V0]][%[[C1]] : i32] : vector<2xf32>
// CHECK-NEXT: llvm.return %[[V1]] : vector<2xf32>
// CHECK-NEXT: }

// -----

// VectorType (vector<4xi8>, 4 bytes) stored, IntegerType (i32, 4 bytes) loaded

llvm.func @forward_vector_to_int_bitcast(%val: vector<4xi8>) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x vector<4xi8> : (i32) -> !llvm.ptr
  llvm.store %val, %al : vector<4xi8>, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> i32
  llvm.return %0 : i32
}

// CHECK: llvm.func @forward_vector_to_int_bitcast(%[[VAL:.*]]: vector<4xi8>) -> i32 {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BC:.*]] = llvm.bitcast %[[VAL]] : vector<4xi8> to i32
// CHECK-NEXT: llvm.return %[[BC]] : i32
// CHECK-NEXT: }