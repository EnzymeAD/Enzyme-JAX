// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=xla-gpu})" | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=xla-cpu})" | FileCheck %s
// RUN: enzymexlamlir-opt %s --cse | FileCheck %s --check-prefix=CSE
// RUN: not enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=cuda})" 2>&1 | FileCheck %s --check-prefix=BACKEND
// RUN: not enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=xla-gpu use-c-style-memref=false})" 2>&1 | FileCheck %s --check-prefix=DESCRIPTOR

// No megakernelize pass is involved. Two accesses to a symbol share storage;
// separate declarations own separate allocations. The copy remains in update.
module {
  enzymexla.temp_alloc "private" @bound : memref<i32, 1>
  enzymexla.temp_alloc "private" @other : memref<1xi64, 1>

  func.func @update(%value: i32) {
    %zero = arith.constant 0 : index
    %four = arith.constant 4 : index
    %host = memref.alloca() : memref<1xi32>
    memref.store %value, %host[%zero] : memref<1xi32>
    %device = enzymexla.get_global_temp @bound : memref<i32, 1>
    enzymexla.memcpy %device, %host, %four : memref<i32, 1>, memref<1xi32>
    return
  }

  func.func @get_bound() -> memref<i32, 1> {
    %first = enzymexla.get_global_temp @bound : memref<i32, 1>
    %second = enzymexla.get_global_temp @bound : memref<i32, 1>
    func.call @consume(%first, %second) : (memref<i32, 1>, memref<i32, 1>) -> ()
    return %second : memref<i32, 1>
  }
  func.func private @consume(memref<i32, 1>, memref<i32, 1>)
}

// CHECK-DAG: llvm.mlir.global internal @[[BOUND:__reactant_temp_bound]]
// CHECK-DAG: llvm.mlir.global internal @[[OTHER:__reactant_temp_other]]
// CHECK-LABEL: llvm.func @update(
// CHECK-NOT: llvm.call @reactantXLAMalloc
// CHECK: llvm.mlir.addressof @[[BOUND]]
// CHECK: llvm.load
// CHECK: llvm.call @reactantXLAMemcpy
// CHECK-NOT: llvm.call @reactantXLAFree
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @get_bound(
// CHECK-NOT: llvm.call @reactantXLAMalloc
// CHECK: llvm.mlir.addressof @[[BOUND]]
// CHECK: llvm.load
// CHECK-NOT: llvm.call @reactantXLAFree
// CHECK: llvm.return
// CHECK-LABEL: llvm.func internal @__reactant_temps_init()
// CHECK-DAG: %[[INIT_BOUND:.*]] = llvm.mlir.addressof @[[BOUND]]
// CHECK-DAG: %[[INIT_OTHER:.*]] = llvm.mlir.addressof @[[OTHER]]
// CHECK: llvm.call @reactantXLAMalloc
// CHECK: llvm.store %{{.*}}, %[[INIT_BOUND]] : !llvm.ptr<1>, !llvm.ptr
// CHECK: llvm.call @reactantXLAMalloc
// CHECK: llvm.store %{{.*}}, %[[INIT_OTHER]] : !llvm.ptr<1>, !llvm.ptr
// CHECK: llvm.return
// CHECK-LABEL: llvm.func internal @__reactant_temps_deinit()
// CHECK-DAG: %[[FINI_BOUND:.*]] = llvm.mlir.addressof @[[BOUND]]
// CHECK-DAG: %[[FINI_OTHER:.*]] = llvm.mlir.addressof @[[OTHER]]
// CHECK-DAG: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr<1>
// CHECK: llvm.load %[[FINI_OTHER]]
// CHECK: llvm.call @reactantXLAFree
// CHECK: llvm.store %[[NULL]], %[[FINI_OTHER]]
// CHECK: llvm.load %[[FINI_BOUND]]
// CHECK: llvm.call @reactantXLAFree
// CHECK: llvm.store %[[NULL]], %[[FINI_BOUND]]
// CHECK: llvm.return
// CHECK-DAG: llvm.mlir.global_ctors ctors = [@__reactant_temps_init], priorities = [65535 : i32]
// CHECK-DAG: llvm.mlir.global_dtors dtors = [@__reactant_temps_deinit], priorities = [65535 : i32]
// CHECK-DAG: llvm.mlir.global_ctors ctors = [@__reactant_xla_init], priorities = [65534 : i32]
// CHECK-DAG: llvm.mlir.global_dtors dtors = [@__reactant_xla_deinit], priorities = [65534 : i32]

// CSE-LABEL: func.func @get_bound
// CSE: %[[HANDLE:.*]] = enzymexla.get_global_temp @bound
// CSE-NOT: enzymexla.get_global_temp
// CSE: call @consume(%[[HANDLE]], %[[HANDLE]])

// BACKEND: persistent temporaries require an XLA backend
// DESCRIPTOR: persistent temporaries require C-style memref lowering
