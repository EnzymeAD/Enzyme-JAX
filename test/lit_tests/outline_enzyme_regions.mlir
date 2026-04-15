// RUN: enzymexlamlir-opt %s --outline-enzyme-regions | FileCheck %s

// outline-enzyme-regions must pull the autodiff_region out

module {
  func.func @type_a(%d0: index, %d1: index, %d2: index,
                    %d3: index, %d4: index, %d5: index,
                    %mem: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f64

    %w = "enzymexla.gpu_wrapper"(%d0, %d1, %d2, %d3, %d4, %d5) ({
      scf.parallel (%i0, %i1, %i2, %i3, %i4, %i5) =
          (%c0, %c0, %c0, %c0, %c0, %c0)
          to (%d0, %d1, %d2, %d3, %d4, %d5)
          step (%c1, %c1, %c1, %c1, %c1, %c1) {
        %alloca = memref.alloca() : memref<1xf64>
        %ptr0 = llvm.alloca %c1_i32 x !llvm.array<4 x f64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
        %ptr1 = llvm.alloca %c1_i64 x !llvm.struct<(i32)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
        %as_memref = "enzymexla.pointer2memref"(%ptr1) : (!llvm.ptr) -> memref<?xi32>
        memref.store %cst, %alloca[%c0] : memref<1xf64>
        enzyme.autodiff_region(%ptr0) {
        ^bb0(%arg: !llvm.ptr):
          scf.parallel (%j) = (%c0) to (%c4) step (%c1) {
            memref.store %cst, %mem[%j] : memref<?xf64>
            scf.reduce
          }
          enzyme.yield
        } attributes {
          activity = [#enzyme<activity enzyme_const>],
          ret_activity = []
        } : (!llvm.ptr) -> ()
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @type_a
// CHECK:         "enzymexla.gpu_wrapper"
// CHECK:         scf.parallel
// CHECK:         "enzymexla.pointer2memref"
// CHECK-NOT:     scf.parallel
// CHECK-NOT:     enzyme.autodiff_region
// CHECK:         enzyme.autodiff @type_a_to_diff0
// CHECK-SAME:    activity = [#enzyme<activity enzyme_const>
// CHECK:         "enzymexla.polygeist_yield"
// CHECK-LABEL: func.func @type_a_to_diff0
// CHECK:         scf.parallel
// CHECK:         memref.store
// CHECK:         return
