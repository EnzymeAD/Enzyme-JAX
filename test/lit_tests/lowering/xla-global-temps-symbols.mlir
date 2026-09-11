// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(symbol-dce,convert-polygeist-to-llvm{backend=xla-gpu})" | FileCheck %s

module {
  // Generated LLVM symbols must not collide with existing symbols. Symbol DCE
  // must retain the declaration referenced by get_global_temp.
  llvm.func @__reactant_temps_init()
  llvm.func @__reactant_temps_deinit()
  llvm.mlir.global external @__reactant_temp_bound() : i32
  enzymexla.temp_alloc "private" @bound : memref<i32, 1>

  func.func @get_bound() -> memref<i32, 1> {
    %bound = enzymexla.get_global_temp @bound : memref<i32, 1>
    return %bound : memref<i32, 1>
  }
}

// CHECK: llvm.mlir.global external @__reactant_temp_bound()
// CHECK: llvm.mlir.global internal @[[SLOT:__reactant_temp_bound_[0-9]+]]()
// CHECK-LABEL: llvm.func @get_bound()
// CHECK: llvm.mlir.addressof @[[SLOT]]
// CHECK: llvm.load
// CHECK: llvm.func internal @[[CTOR:__reactant_temps_init_[0-9]+]]()
// CHECK: llvm.mlir.addressof @[[SLOT]]
// CHECK: llvm.call @reactantXLAMalloc
// CHECK: llvm.func internal @[[DTOR:__reactant_temps_deinit_[0-9]+]]()
// CHECK: llvm.mlir.addressof @[[SLOT]]
// CHECK: llvm.call @reactantXLAFree
// CHECK: llvm.mlir.global_ctors ctors = [@[[CTOR]]]
// CHECK: llvm.mlir.global_dtors dtors = [@[[DTOR]]]
