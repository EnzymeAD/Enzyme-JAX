// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" --split-input-file | FileCheck %s
// A whole-aggregate load only read field by field is read one field at a
// time instead: how a by-reference lambda capture reaches its kernel.
func.func @split_struct_load(%p: !llvm.ptr) -> (i32, f64) {
  %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
  %s = affine.load %m[0] : memref<?x!llvm.struct<(i32, f64)>>
  %f0 = llvm.extractvalue %s[0] : !llvm.struct<(i32, f64)>
  %f1 = llvm.extractvalue %s[1] : !llvm.struct<(i32, f64)>
  return %f0, %f1 : i32, f64
}

// CHECK-LABEL: func.func @split_struct_load(
// CHECK-SAME: %[[P:[a-z0-9]+]]: !llvm.ptr
// CHECK-NOT: llvm.struct
// CHECK-DAG: %[[MI:.+]] = "enzymexla.pointer2memref"(%[[P]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-DAG: %[[VI:.+]] = affine.load %[[MI]][0] : memref<?xi32>
// CHECK-DAG: %[[MF:.+]] = "enzymexla.pointer2memref"(%[[P]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-DAG: %[[VF:.+]] = affine.load %[[MF]][1] : memref<?xf64>
// CHECK: return %[[VI]], %[[VF]] : i32, f64
