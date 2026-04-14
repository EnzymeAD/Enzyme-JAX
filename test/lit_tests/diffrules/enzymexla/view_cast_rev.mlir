// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s

// Verify that reverse-mode AD does not crash on enzymexla.pointer2memref
module {
  func.func @load_via_p2m(%ptr: !llvm.ptr, %i: index) -> f64 {
    %mem = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xf64>
    %v = memref.load %mem[%i] : memref<?xf64>
    return %v : f64
  }

  func.func @diff_load_via_p2m(%ptr: !llvm.ptr, %d_ptr: !llvm.ptr,
                                %i: index, %seed: f64) {
    enzyme.autodiff @load_via_p2m(%ptr, %d_ptr, %i, %seed) {
      activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (!llvm.ptr, !llvm.ptr, index, f64) -> ()
    return
  }

  func.func @load_via_m2p(%mem: memref<?xf64>) -> f64 {
    %ptr = "enzymexla.memref2pointer"(%mem) : (memref<?xf64>) -> !llvm.ptr
    %v = llvm.load %ptr : !llvm.ptr -> f64
    return %v : f64
  }

  func.func @diff_load_via_m2p(%mem: memref<?xf64>, %d_mem: memref<?xf64>,
                                %seed: f64) {
    enzyme.autodiff @load_via_m2p(%mem, %d_mem, %seed) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (memref<?xf64>, memref<?xf64>, f64) -> ()
    return
  }
}

// CHECK-LABEL: func.func private @diffeload_via_p2m(
// CHECK-SAME:    %[[PTR:[^,]+]]: !llvm.ptr,
// CHECK-SAME:    %[[DPTR:[^,]+]]: !llvm.ptr,
// CHECK-SAME:    %[[I:[^,]+]]: index,
// CHECK-SAME:    %[[SEED:[^)]+]]: f64)
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %[[DMEM:.*]] = "enzymexla.pointer2memref"(%[[DPTR]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK:         %[[ACC:.*]] = arith.addf %[[SEED]], %[[ZERO]] : f64
// CHECK:         %[[OLD:.*]] = memref.load %[[DMEM]][%[[I]]] : memref<?xf64>
// CHECK:         %[[NEW:.*]] = arith.addf %[[OLD]], %[[ACC]] : f64
// CHECK:         memref.store %[[NEW]], %[[DMEM]][%[[I]]] : memref<?xf64>

// CHECK-LABEL: func.func private @diffeload_via_m2p(
// CHECK-SAME:    %[[MEM:[^,]+]]: memref<?xf64>,
// CHECK-SAME:    %[[DMEM:[^,]+]]: memref<?xf64>,
// CHECK-SAME:    %[[SEED:[^)]+]]: f64)
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %[[DPTR:.*]] = "enzymexla.memref2pointer"(%[[DMEM]]) : (memref<?xf64>) -> !llvm.ptr
// CHECK:         %[[ACC:.*]] = arith.addf %[[SEED]], %[[ZERO]] : f64
// CHECK:         %[[OLD:.*]] = llvm.load %[[DPTR]] : !llvm.ptr -> f64
// CHECK:         %[[NEW:.*]] = arith.addf %[[OLD]], %[[ACC]] : f64
// CHECK:         llvm.store %[[NEW]], %[[DPTR]] : f64, !llvm.ptr
