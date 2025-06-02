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

// ----

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

// ----

func.func @negative_test(%a: vector<2xf32>, %idx: i32) -> f32 {
  %result = llvm.extractelement %a[%idx : i32] : vector<2xf32>
  return %result : f32
}
// CHECK-LABEL: func @negative_test
// Should NOT trigger the transformation
// CHECK: llvm.extractelement {{.*}} : vector<2xf32>
// CHECK-NOT: arith.select

// ----

func.func @test_select_trunc(%arg0: i1, %arg1: i32, %arg2: i32) -> i8 {
  %select = arith.select %arg0, %arg1, %arg2 : i32
  %trunc = arith.trunci %select : i32 to i8
  return %trunc : i8
}
// CHECK-LABEL: func @test_select_trunc
// CHECK: %[[TRUE:.*]] = arith.trunci %arg1 : i32 to i8
// CHECK: %[[FALSE:.*]] = arith.trunci %arg2 : i32 to i8
// CHECK: arith.select %arg0, %[[TRUE]], %[[FALSE]] : i8

// ----

func.func @test_extui_select_trunci(%arg0: i1, %arg1: i8, %arg2: i8) -> i8 {
  
  // Original operations
  %ext1 = arith.extui %arg1 : i8 to i32
  %ext2 = arith.extui %arg2 : i8 to i32
  %select = arith.select %arg0, %ext1, %ext2 : i32
  %result = arith.trunci %select : i32 to i8
  return %result : i8
}
// CHECK-LABEL: func @test_extui_select_trunci
// CHECK-NOT: arith.extui
// CHECK-NOT: arith.trunci
// CHECK: arith.select %arg0, %arg1, %arg2 : i8

// ----
func.func @select_extractvalue_simple(%arg0: i1, %arg1: !llvm.array<2 x f64>, %arg2: !llvm.array<2 x f64>) -> f64 {
  %0 = llvm.extractvalue %arg1[0] : !llvm.array<2 x f64> 
  %1 = llvm.extractvalue %arg2[0] : !llvm.array<2 x f64> 
  %2 = arith.select %arg0, %0, %1 : f64
  return %2 : f64
}
// CHECK-LABEL:   func.func @select_extractvalue_simple(
// CHECK-SAME:                                          %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                          %[[VAL_1:.*]]: !llvm.array<2 x f64>,
// CHECK-SAME:                                          %[[VAL_2:.*]]: !llvm.array<2 x f64>) -> f64 {
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.array<2 x f64>
// CHECK:           %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_2]][0] : !llvm.array<2 x f64>
// CHECK:           %[[VAL_5:.*]] = arith.select %[[VAL_0]], %[[VAL_3]], %[[VAL_4]] : f64
// CHECK:           return %[[VAL_5]] : f64
// CHECK:         }

// ----
func.func @select_extractvalue_array(%cond: i1, %a: !llvm.array<2 x f64>, %b: !llvm.array<2 x f64>) -> (f64, f64) {
  %sel = arith.select %cond, %a, %b {fastmathFlags = #llvm.fastmath<none>} : !llvm.array<2 x f64>
  %e0 = llvm.extractvalue %sel[0] : !llvm.array<2 x f64>
  %e1 = llvm.extractvalue %sel[1] : !llvm.array<2 x f64>
  return %e0, %e1 : f64, f64
}
// CHECK-LABEL:   func.func @select_extractvalue_array(
// CHECK-SAME:                                         %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                         %[[VAL_1:.*]]: !llvm.array<2 x f64>,
// CHECK-SAME:                                         %[[VAL_2:.*]]: !llvm.array<2 x f64>) -> (f64, f64) {
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.array<2 x f64>
// CHECK:           %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_2]][0] : !llvm.array<2 x f64>
// CHECK:           %[[VAL_5:.*]] = arith.select %[[VAL_0]], %[[VAL_3]], %[[VAL_4]] : f64
// CHECK:           %[[VAL_6:.*]] = llvm.extractvalue %[[VAL_1]][1] : !llvm.array<2 x f64>
// CHECK:           %[[VAL_7:.*]] = llvm.extractvalue %[[VAL_2]][1] : !llvm.array<2 x f64>
// CHECK:           %[[VAL_8:.*]] = arith.select %[[VAL_0]], %[[VAL_6]], %[[VAL_7]] : f64
// CHECK:           return %[[VAL_5]], %[[VAL_8]] : f64, f64
// CHECK:         }
