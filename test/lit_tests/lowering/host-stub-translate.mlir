// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=cuda})" | FileCheck %s

// A kernel pointer that flows into a cuda runtime query (occupancy,
// attributes) as a value cannot be rewritten to the registered stub at
// compile time; the query receives the original host function whose
// registration was removed and fails with invalid resource handle. The
// translation function maps each raised kernel's host function to the stub
// this pass registers for it, and leaves unknown pointers unchanged.

module attributes {gpu.container_module} {
  llvm.func @_Z4stepv() {
    llvm.return
  }
  llvm.func @"__reactant$get_device_from_host"(!llvm.ptr) -> !llvm.ptr
  llvm.func @query(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @"__reactant$get_device_from_host"(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @launch(%arg0: i64) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %stream = llvm.inttoptr %arg0 : i64 to !llvm.ptr
    %e = "enzymexla.gpu_error"() ({
      gpu.launch_func <%stream : !llvm.ptr> @_Z4stepv_kernel_1::@_Z4stepv_kernel blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1)
      "enzymexla.polygeist_yield"() : () -> ()
    }) : () -> index
    llvm.return
  }
  gpu.module @_Z4stepv_kernel_1 [#nvvm.target] {
    gpu.func @_Z4stepv_kernel() kernel {
      gpu.return
    }
  }
}

// CHECK-LABEL: llvm.func internal @__reactant$get_device_from_host(%arg0: !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT: %[[ORIG:.+]] = llvm.mlir.addressof @_Z4stepv : !llvm.ptr
// CHECK-NEXT: %[[STUB:.+]] = llvm.mlir.addressof @__polygeist__Z4stepv_kernel_1__Z4stepv_kernel_device_stub : !llvm.ptr
// CHECK-NEXT: %[[EQ:.+]] = llvm.icmp "eq" %arg0, %[[ORIG]] : !llvm.ptr
// CHECK-NEXT: %[[RES:.+]] = llvm.select %[[EQ]], %[[STUB]], %arg0 : i1, !llvm.ptr
// CHECK-NEXT: llvm.return %[[RES]] : !llvm.ptr

// CHECK-LABEL: llvm.func @query(%arg0: !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT: %[[V:.+]] = llvm.call @__reactant$get_device_from_host(%arg0) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT: llvm.return %[[V]] : !llvm.ptr
