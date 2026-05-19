// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s


// Same type – forwarded with no cast at all

llvm.func @forward_same_type(%val: i32) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %al : i32, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @forward_same_type
// CHECK-NOT:   llvm.inttoptr
// CHECK-NOT:   llvm.ptrtoint
// CHECK-NOT:   arith.bitcast
// CHECK-NOT:   llvm.bitcast
// CHECK-NOT:   llvm.store
// CHECK-NOT:   llvm.load
// CHECK:       llvm.return %[[VAL:.+]] : i32

// -----

// IntegerType (i64) stored, LLVMPointerType loaded

llvm.func @forward_int_to_ptr(%val: i64) -> !llvm.ptr {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x i64 : (i32) -> !llvm.ptr
  llvm.store %val, %al : i64, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: llvm.func @forward_int_to_ptr
// CHECK-NOT:   llvm.store
// CHECK-NOT:   llvm.load
// CHECK:       %[[ITP:.+]] = llvm.inttoptr %[[VAL:.+]] : i64 to !llvm.ptr
// CHECK:       llvm.return %[[ITP]] : !llvm.ptr

// -----

// LLVMPointerType stored, IntegerType (i64) loaded

llvm.func @forward_ptr_to_int(%val: !llvm.ptr) -> i64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
  llvm.store %val, %al : !llvm.ptr, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> i64
  llvm.return %0 : i64
}

// CHECK-LABEL: llvm.func @forward_ptr_to_int
// CHECK-NOT:   llvm.store
// CHECK-NOT:   llvm.load
// CHECK:       %[[PTI:.+]] = llvm.ptrtoint %[[VAL:.+]] : !llvm.ptr to i64
// CHECK:       llvm.return %[[PTI]] : i64

// -----

// FloatType (f32) stored, IntegerType (i32) loaded

llvm.func @forward_float_to_int(%val: f32) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x f32 : (i32) -> !llvm.ptr
  llvm.store %val, %al : f32, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @forward_float_to_int
// CHECK-NOT:   llvm.store
// CHECK-NOT:   llvm.load
// CHECK:       %[[BC:.+]] = arith.bitcast %[[VAL:.+]] : f32 to i32
// CHECK:       llvm.return %[[BC]] : i32

// -----

// IntegerType (i32) stored, FloatType (f32) loaded

llvm.func @forward_int_to_float(%val: i32) -> f32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %al : i32, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> f32
  llvm.return %0 : f32
}

// CHECK-LABEL: llvm.func @forward_int_to_float
// CHECK-NOT:   llvm.store
// CHECK-NOT:   llvm.load
// CHECK:       %[[BC:.+]] = arith.bitcast %[[VAL:.+]] : i32 to f32
// CHECK:       llvm.return %[[BC]] : f32

// -----

// Scalar (i32) stored, single-field struct { i32 } loaded

llvm.func @forward_scalar_to_single_field_struct(%val: i32) -> !llvm.struct<(i32)> {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %al : i32, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> !llvm.struct<(i32)>
  llvm.return %0 : !llvm.struct<(i32)>
}

// CHECK-LABEL: llvm.func @forward_scalar_to_single_field_struct
// CHECK-NOT:   llvm.store
// CHECK-NOT:   llvm.load
// CHECK:       %[[UD:.+]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK:       %[[INS:.+]] = llvm.insertvalue %[[VAL:.+]], %[[UD]][0] : !llvm.struct<(i32)>
// CHECK:       llvm.return %[[INS]] : !llvm.struct<(i32)>

// -----

// Single-field struct { i32 } stored, scalar i32 loaded

llvm.func @forward_single_field_struct_to_scalar(%val: !llvm.struct<(i32)>) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x !llvm.struct<(i32)> : (i32) -> !llvm.ptr
  llvm.store %val, %al : !llvm.struct<(i32)>, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @forward_single_field_struct_to_scalar
// CHECK-NOT:   llvm.store
// CHECK-NOT:   llvm.load
// CHECK:       %[[EX:.+]] = llvm.extractvalue %[[VAL:.+]][0] : !llvm.struct<(i32)>
// CHECK:       llvm.return %[[EX]] : i32

// -----

// Nested struct { struct { f32 } } stored, f32 loaded

llvm.func @forward_nested_struct_to_scalar(%val: !llvm.struct<(struct<(f32)>)>) -> f32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x !llvm.struct<(struct<(f32)>)> : (i32) -> !llvm.ptr
  llvm.store %val, %al : !llvm.struct<(struct<(f32)>)>, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> f32
  llvm.return %0 : f32
}

// CHECK-LABEL: llvm.func @forward_nested_struct_to_scalar
// CHECK-NOT:   llvm.store
// CHECK-NOT:   llvm.load
// CHECK:       %[[EX0:.+]] = llvm.extractvalue %[[VAL:.+]][0] : !llvm.struct<(struct<(f32)>)>
// CHECK:       %[[EX1:.+]] = llvm.extractvalue %[[EX0]][0] : !llvm.struct<(f32)>
// CHECK:       llvm.return %[[EX1]] : f32

// -----

// Uniform two-field struct { f32, f32 } stored, vector<2xf32> loaded

llvm.func @forward_uniform_struct_to_vector(%val: !llvm.struct<(f32, f32)>) -> vector<2xf32> {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x !llvm.struct<(f32, f32)> : (i32) -> !llvm.ptr
  llvm.store %val, %al : !llvm.struct<(f32, f32)>, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> vector<2xf32>
  llvm.return %0 : vector<2xf32>
}

// CHECK-LABEL: llvm.func @forward_uniform_struct_to_vector
// CHECK-NOT:   llvm.store
// CHECK-NOT:   llvm.load
// CHECK:       %[[EX0:.+]] = llvm.extractvalue %[[VAL:.+]][0] : !llvm.struct<(f32, f32)>
// CHECK:       %[[EX1:.+]] = llvm.extractvalue %[[VAL:.+]][1] : !llvm.struct<(f32, f32)>
// CHECK:       %[[PSN:.+]] = llvm.mlir.poison : vector<2xf32>
// CHECK:       %[[C0:.+]]  = llvm.mlir.constant(0 : i32) : i32
// CHECK:       %[[V0:.+]]  = llvm.insertelement %[[EX0]], %[[PSN]][%[[C0]] : i32] : vector<2xf32>
// CHECK:       %[[C1:.+]]  = llvm.mlir.constant(1 : i32) : i32
// CHECK:       %[[V1:.+]]  = llvm.insertelement %[[EX1]], %[[V0]][%[[C1]] : i32] : vector<2xf32>
// CHECK:       llvm.return %[[V1]] : vector<2xf32>

// -----

// LLVM array [2 x f32] stored, vector<2xf32> loaded

llvm.func @forward_array_to_vector(%val: !llvm.array<2 x f32>) -> vector<2xf32> {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x !llvm.array<2 x f32> : (i32) -> !llvm.ptr
  llvm.store %val, %al : !llvm.array<2 x f32>, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> vector<2xf32>
  llvm.return %0 : vector<2xf32>
}

// CHECK-LABEL: llvm.func @forward_array_to_vector
// CHECK-NOT:   llvm.store
// CHECK-NOT:   llvm.load
// CHECK:       %[[EX0:.+]] = llvm.extractvalue %[[VAL:.+]][0] : !llvm.array<2 x f32>
// CHECK:       %[[EX1:.+]] = llvm.extractvalue %[[VAL:.+]][1] : !llvm.array<2 x f32>
// CHECK:       %[[PSN:.+]] = llvm.mlir.poison : vector<2xf32>
// CHECK:       %[[C0:.+]]  = llvm.mlir.constant(0 : i32) : i32
// CHECK:       %[[V0:.+]]  = llvm.insertelement %[[EX0]], %[[PSN]][%[[C0]] : i32] : vector<2xf32>
// CHECK:       %[[C1:.+]]  = llvm.mlir.constant(1 : i32) : i32
// CHECK:       %[[V1:.+]]  = llvm.insertelement %[[EX1]], %[[V0]][%[[C1]] : i32] : vector<2xf32>
// CHECK:       llvm.return %[[V1]] : vector<2xf32>

// -----

// VectorType (vector<4xi8>, 4 bytes) stored, IntegerType (i32, 4 bytes) loaded

llvm.func @forward_vector_to_int_bitcast(%val: vector<4xi8>) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %al = llvm.alloca %c1 x vector<4xi8> : (i32) -> !llvm.ptr
  llvm.store %val, %al : vector<4xi8>, !llvm.ptr
  %0 = llvm.load %al : !llvm.ptr -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @forward_vector_to_int_bitcast
// CHECK-NOT:   llvm.store
// CHECK-NOT:   llvm.load
// CHECK:       %[[BC:.+]] = llvm.bitcast %[[VAL:.+]] : vector<4xi8> to i32
// CHECK:       llvm.return %[[BC]] : i32