// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1{emit_single_alternative=true})" | FileCheck %s

// CHECK-LABEL: func.func @same_space(
func.func @same_space(%in: memref<3x2x4xi32, 1>, %out: memref<3x2x4xi32, 1>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %one = arith.constant 1 : i32
  // CHECK: gpu.launch
  // CHECK: %[[BLOCK:.+]] = gpu.block_id x
  // CHECK: %[[X:.+]] = gpu.thread_id x
  // CHECK: %[[Y:.+]] = gpu.thread_id y
  %r = "enzymexla.gpu_wrapper"(%c3, %c1, %c1, %c2, %c4, %c1) ({
    scf.parallel (%b) = (%c0) to (%c3) step (%c1) {
      scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c4) step (%c1, %c1) {
        %value = arith.index_cast %j : index to i32
        // CHECK: memref.store %{{.+}}, %{{.+}}[%[[BLOCK]], %[[X]], %[[Y]]]
        memref.store %value, %in[%b, %i, %j] : memref<3x2x4xi32, 1>
        scf.reduce
      }
      // CHECK-NEXT: gpu.barrier
      scf.parallel (%x, %y) = (%c0, %c0) to (%c2, %c4) step (%c1, %c1) {
        // CHECK-NEXT: %[[NEIGHBOR:.+]] = arith.xori %[[Y]], %{{.+}} : index
        %neighbor = arith.xori %y, %c1 : index
        // CHECK-NEXT: %[[VALUE:.+]] = memref.load %{{.+}}[%[[BLOCK]], %[[X]], %[[NEIGHBOR]]]
        %value = memref.load %in[%b, %x, %neighbor] : memref<3x2x4xi32, 1>
        // CHECK-NEXT: memref.store %[[VALUE]], %{{.+}}[%[[BLOCK]], %[[X]], %[[Y]]]
        memref.store %value, %out[%b, %x, %y] : memref<3x2x4xi32, 1>
        scf.reduce
      }
      // CHECK-NEXT: gpu.barrier
      scf.parallel (%x, %y) = (%c0, %c0) to (%c2, %c4) step (%c1, %c1) {
        // CHECK-NEXT: %[[VALUE2:.+]] = memref.load %{{.+}}[%[[BLOCK]], %[[X]], %[[Y]]]
        %value = memref.load %out[%b, %x, %y] : memref<3x2x4xi32, 1>
        // CHECK-NEXT: %[[RESULT:.+]] = arith.addi %[[VALUE2]], %{{.+}} : i32
        %result = arith.addi %value, %one : i32
        // CHECK-NEXT: memref.store %[[RESULT]], %{{.+}}[%[[BLOCK]], %[[X]], %[[Y]]]
        memref.store %result, %out[%b, %x, %y] : memref<3x2x4xi32, 1>
        scf.reduce
      }
      scf.reduce
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  // CHECK-NEXT: gpu.terminator
  return %r : index
}

// CHECK-LABEL: func.func @column(
func.func @column(%a: memref<3x45xf32, 1>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c44 = arith.constant 44 : index
  %c45 = arith.constant 45 : index
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  // CHECK: gpu.launch
  %r = "enzymexla.gpu_wrapper"(%c3, %c1, %c1, %c44, %c1, %c1) ({
    scf.parallel (%b) = (%c0) to (%c3) step (%c1) {
      // CHECK: memref.store
      memref.store %one, %a[%b, %c0] : memref<3x45xf32, 1>
      // CHECK: gpu.barrier
      scf.parallel (%k) = (%c0) to (%c44) step (%c1) {
        %next = arith.addi %k, %c1 : index
        // CHECK: memref.store
        memref.store %one, %a[%b, %next] : memref<3x45xf32, 1>
        scf.reduce
      }
      // CHECK: gpu.barrier
      // CHECK: scf.for
      scf.for %k = %c1 to %c45 step %c1 {
        %previous = arith.subi %k, %c1 : index
        %value = memref.load %a[%b, %previous] : memref<3x45xf32, 1>
        %result = arith.addf %value, %one : f32
        // CHECK: memref.store
        memref.store %result, %a[%b, %k] : memref<3x45xf32, 1>
      }
      // CHECK: gpu.barrier
      scf.parallel (%k) = (%c0) to (%c44) step (%c1) {
        %next = arith.addi %k, %c1 : index
        // CHECK: memref.load
        %value = memref.load %a[%b, %next] : memref<3x45xf32, 1>
        %result = arith.mulf %value, %two : f32
        // CHECK: memref.store
        memref.store %result, %a[%b, %next] : memref<3x45xf32, 1>
        scf.reduce
      }
      scf.reduce
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  // CHECK: gpu.terminator
  return %r : index
}

// CHECK-LABEL: func.func @different_space(
func.func @different_space(%out: memref<3x8xi32, 1>, %value: i32) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  // CHECK: enzymexla.gpu_wrapper
  %r = "enzymexla.gpu_wrapper"(%c3, %c1, %c1, %c8, %c1, %c1) ({
    // CHECK: scf.parallel
    scf.parallel (%b) = (%c0) to (%c3) step (%c1) {
      // CHECK: scf.parallel
      scf.parallel (%k) = (%c0) to (%c4) step (%c1) {
        memref.store %value, %out[%b, %k] : memref<3x8xi32, 1>
        scf.reduce
      }
      // CHECK: scf.parallel
      scf.parallel (%k) = (%c1) to (%c4) step (%c1) {
        memref.store %value, %out[%b, %k] : memref<3x8xi32, 1>
        scf.reduce
      }
      // CHECK: scf.parallel
      scf.parallel (%k) = (%c1) to (%c8) step (%c1) {
        memref.store %value, %out[%b, %k] : memref<3x8xi32, 1>
        scf.reduce
      }
      // CHECK: scf.parallel
      scf.parallel (%k) = (%c1) to (%c8) step (%c2) {
        memref.store %value, %out[%b, %k] : memref<3x8xi32, 1>
        scf.reduce
      }
      scf.reduce
    }
    // CHECK-NOT: enzymexla.barrier
    // CHECK: enzymexla.polygeist_yield
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return %r : index
}
