// RUN: enzymexlamlir-opt --canonicalize-scf-for --split-input-file %s | FileCheck %s

func.func @select_extract_scalar(%cond: i1, %a: vector<4xf32>, %b: vector<4xf32>, %idx: i32) -> f32 {
  %sel = arith.select %cond, %a, %b : vector<4xf32>
  %result = llvm.extractelement %sel[%idx : i32] : vector<4xf32>
  return %result : f32
}
// CHECK-LABEL: func @select_extract_scalar
// CHECK: [[A_EXTRACT:%[a-zA-Z0-9]+]] = llvm.extractelement %{{.*}}[%{{.*}}] : vector<4xf32>
// CHECK: [[B_EXTRACT:%[a-zA-Z0-9]+]] = llvm.extractelement %{{.*}}[%{{.*}}] : vector<4xf32>
// CHECK: arith.select %{{.*}}, [[A_EXTRACT]], [[B_EXTRACT]] : f32

// -----

func.func @select_extract_multiple_indices(
  %cond: i1, %a: vector<8xi64>, %b: vector<8xi64>, %idx1: i32, %idx2: i32) -> (i64, i64) {
  
  %sel = arith.select %cond, %a, %b : vector<8xi64>
  %e1 = llvm.extractelement %sel[%idx1 : i32] : vector<8xi64>
  %e2 = llvm.extractelement %sel[%idx2 : i32] : vector<8xi64>
  return %e1, %e2 : i64, i64
} 
// CHECK-LABEL: func @select_extract_multiple_indices
// CHECK: llvm.extractelement %{{.*}}[%{{.*}}] : vector<8xi64>
// CHECK: llvm.extractelement %{{.*}}[%{{.*}}] : vector<8xi64>
// CHECK: arith.select %{{.*}}, {{.*}} : i64
// CHECK: llvm.extractelement %{{.*}}[%{{.*}}] : vector<8xi64>
// CHECK: llvm.extractelement %{{.*}}[%{{.*}}] : vector<8xi64>
// CHECK: arith.select %{{.*}}, {{.*}} : i64

// -----

func.func @negative_test(%a: vector<2xf32>, %idx: i32) -> f32 {
  %result = llvm.extractelement %a[%idx : i32] : vector<2xf32>
  return %result : f32
}
// CHECK-LABEL: func @negative_test
// Should NOT trigger the transformation
// CHECK: llvm.extractelement {{.*}} : vector<2xf32>
// CHECK-NOT: arith.select
