// RUN: enzymexlamlir-opt --mark-gridkit-sparse-jacobian-llvm %s | FileCheck %s

module {
  llvm.func @_ZN7GridKit6Enzyme6Sparse16__enzyme_todenseIPdEET_Pvz(!llvm.ptr, !llvm.ptr, i64, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.func @_ZN7GridKit6Enzyme6Sparse16__enzyme_fwddiffIvEEv(!llvm.ptr, i32, i32, i32)
  llvm.func @_ZN7GridKit6Enzyme6SparseL12sparse_storeIdmEEv()
  llvm.func @residual_wrapper(!llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.mlir.global external @enzyme_const(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64} : i32
  llvm.mlir.global external @enzyme_dup(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64} : i32
  llvm.mlir.global external @enzyme_dupnoneed(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64} : i32

  // CHECK: module attributes {gridkit.jacobian.marked_sparse_helpers = 1 : i64}
  // CHECK: enzymexla.jacobian_materialization materializer = @{{.*4DfDy.*}} residual = @residual_wrapper method = <one_hot_forward> storage = <sparse_callback> fwddiff_calls = 1 todense_calls = 1 sparse_store_callbacks = 1
  // CHECK-SAME: active_input_dimension_arg = 2
  // CHECK-SAME: active_input_index = 1
  // CHECK-SAME: active_input_index_map_arg = 4
  // CHECK-SAME: active_output_index = 0
  // CHECK-SAME: enzyme_activity = ["enzyme_const", "enzyme_dup", "enzyme_dupnoneed"]
  // CHECK-SAME: input_activity = ["enzyme_const", "enzyme_dup"]
  // CHECK-SAME: input_count = 2
  // CHECK-SAME: output_activity = ["enzyme_dupnoneed"]
  // CHECK-SAME: output_count = 1
  // CHECK-SAME: output_dimension_arg = 1
  // CHECK-SAME: output_index_map_arg = 3
  // CHECK-SAME: seed_loop_dimension_arg = 2
  // CHECK-SAME: source = "DfDy"
  // CHECK-SAME: sparse_assembly = "coo_column_seeded_callback"
  // CHECK-SAME: sparse_cols_arg = 9
  // CHECK-SAME: sparse_nnz_arg = 11
  // CHECK-SAME: sparse_rows_arg = 8
  // CHECK-SAME: sparse_values_arg = 10

  // CHECK-LABEL: llvm.func @_ZN7GridKit6Enzyme6Sparse4DfDyINS_5ModelE
  // CHECK-SAME: gridkit.jacobian.fwddiff_calls = 1 : i64
  // CHECK-SAME: gridkit.jacobian.materialization = "enzyme_sparse_one_hot"
  // CHECK-SAME: gridkit.jacobian.source = "DfDy"
  // CHECK-SAME: gridkit.jacobian.sparse_store_addresses = 1 : i64
  // CHECK-SAME: gridkit.jacobian.todense_calls = 1 : i64
  // CHECK-SAME: gridkit.solver = "ida_jac_times"
  llvm.func @_ZN7GridKit6Enzyme6Sparse4DfDyINS_5ModelE(%arg0: !llvm.ptr, %arg1: i64, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: !llvm.ptr) {
    // CHECK: llvm.mlir.addressof @_ZN7GridKit6Enzyme6SparseL12sparse_storeIdmEEv
    %store = llvm.mlir.addressof @_ZN7GridKit6Enzyme6SparseL12sparse_storeIdmEEv : !llvm.ptr
    // CHECK: llvm.call @_ZN7GridKit6Enzyme6Sparse16__enzyme_todenseIPdEET_Pvz
    // CHECK-SAME: gridkit.jacobian.role = "sparse_todense"
    %scale = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %dense = llvm.call @_ZN7GridKit6Enzyme6Sparse16__enzyme_todenseIPdEET_Pvz(%store, %store, %arg2, %scale, %arg3, %arg4, %arg8, %arg9, %arg10, %arg11) : (!llvm.ptr, !llvm.ptr, i64, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %residual = llvm.mlir.addressof @residual_wrapper : !llvm.ptr
    %const_addr = llvm.mlir.addressof @enzyme_const : !llvm.ptr
    %dup_addr = llvm.mlir.addressof @enzyme_dup : !llvm.ptr
    %dupnoneed_addr = llvm.mlir.addressof @enzyme_dupnoneed : !llvm.ptr
    %const = llvm.load %const_addr : !llvm.ptr -> i32
    %dup = llvm.load %dup_addr : !llvm.ptr -> i32
    %dupnoneed = llvm.load %dupnoneed_addr : !llvm.ptr -> i32
    // CHECK: llvm.call @_ZN7GridKit6Enzyme6Sparse16__enzyme_fwddiffIvEEv(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {gridkit.jacobian.action = "residual_jvp_candidate", gridkit.jacobian.source = "DfDy", gridkit.solver = "ida_jac_times"}
    llvm.call @_ZN7GridKit6Enzyme6Sparse16__enzyme_fwddiffIvEEv(%residual, %const, %dup, %dupnoneed) : (!llvm.ptr, i32, i32, i32) -> ()
    llvm.return
  }

  // CHECK-LABEL: llvm.func @_ZN7GridKit6Enzyme6Sparse5DfDypINS_5ModelE
  // CHECK-NOT: gridkit.jacobian.materialization
  llvm.func @_ZN7GridKit6Enzyme6Sparse5DfDypINS_5ModelE(%arg0: !llvm.ptr, %arg1: i64, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: !llvm.ptr) {
    %scale = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %dense = llvm.call @_ZN7GridKit6Enzyme6Sparse16__enzyme_todenseIPdEET_Pvz(%arg0, %arg0, %arg2, %scale, %arg3, %arg4, %arg8, %arg9, %arg10, %arg11) : (!llvm.ptr, !llvm.ptr, i64, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %residual = llvm.mlir.addressof @residual_wrapper : !llvm.ptr
    %const_addr = llvm.mlir.addressof @enzyme_const : !llvm.ptr
    %const = llvm.load %const_addr : !llvm.ptr -> i32
    llvm.call @_ZN7GridKit6Enzyme6Sparse16__enzyme_fwddiffIvEEv(%residual, %const, %const, %const) : (!llvm.ptr, i32, i32, i32) -> ()
    llvm.return
  }
}
