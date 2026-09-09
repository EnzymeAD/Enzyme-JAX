// RUN: enzymexlamlir-opt %s -polygeist-mem2reg | FileCheck %s

// An access through a branch between two allocations names neither of them,
// and keeps both from being promoted. The access is done in each arm instead,
// on the buffer that arm chose, and the forwarding then sees an access of
// each allocation on its own. This is MFEM's H(div) mass kernel choosing
// between two shared arrays by the thread's component,
// `const real_t *B = (c == 0) ? sBo : sBc`, which arrives as an affine.if
// yielding a cast of one alloca or the other.

#set = affine_set<(d0) : (d0 == 0)>

module {
  func.func @load_through_if(%a: f64, %b: f64) -> f64 {
    %s = memref.alloca() : memref<8xf64>
    %t = memref.alloca() : memref<12xf64>
    affine.store %a, %s[1] : memref<8xf64>
    affine.store %b, %t[1] : memref<12xf64>
    %r = affine.for %i = 0 to 2 iter_args(%acc = %a) -> f64 {
      %buf = affine.if #set(%i) -> memref<?xf64> {
        %cs = memref.cast %s : memref<8xf64> to memref<?xf64>
        affine.yield %cs : memref<?xf64>
      } else {
        %ct = memref.cast %t : memref<12xf64> to memref<?xf64>
        affine.yield %ct : memref<?xf64>
      }
      %v = affine.load %buf[1] : memref<?xf64>
      %n = arith.addf %acc, %v : f64
      affine.yield %n : f64
    }
    return %r : f64
  }

  func.func @store_through_select(%c: i1, %v: f64, %a: f64) -> (f64, f64) {
    %s = memref.alloca() : memref<4xf64>
    %t = memref.alloca() : memref<4xf64>
    affine.store %a, %s[2] : memref<4xf64>
    affine.store %a, %t[2] : memref<4xf64>
    %buf = arith.select %c, %s, %t : memref<4xf64>
    affine.store %v, %buf[2] : memref<4xf64>
    %rs = affine.load %s[2] : memref<4xf64>
    %rt = affine.load %t[2] : memref<4xf64>
    return %rs, %rt : f64, f64
  }

  // The access is reached through a cast of the branch's result: the cast
  // goes into the arms first.
  func.func @load_through_cast(%c: i1, %a: f64, %b: f64) -> f64 {
    %s = memref.alloca() : memref<4xf64>
    %t = memref.alloca() : memref<4xf64>
    affine.store %a, %s[0] : memref<4xf64>
    affine.store %b, %t[0] : memref<4xf64>
    %buf = scf.if %c -> memref<4xf64> {
      scf.yield %s : memref<4xf64>
    } else {
      scf.yield %t : memref<4xf64>
    }
    %p = "enzymexla.memref2pointer"(%buf) : (memref<4xf64>) -> !llvm.ptr
    %r = llvm.load %p : !llvm.ptr -> f64
    return %r : f64
  }

  // A branch that does more than choose stays, even once nothing asks it,
  // and the allocation it still names is not promoted.
  func.func @arm_with_effect(%c: i1, %a: f64, %out: memref<f64>) -> f64 {
    %s = memref.alloca() : memref<4xf64>
    %t = memref.alloca() : memref<4xf64>
    affine.store %a, %s[0] : memref<4xf64>
    affine.store %a, %t[0] : memref<4xf64>
    %buf = scf.if %c -> memref<4xf64> {
      affine.store %a, %out[] : memref<f64>
      scf.yield %s : memref<4xf64>
    } else {
      scf.yield %t : memref<4xf64>
    }
    %r = affine.load %buf[0] : memref<4xf64>
    return %r : f64
  }

  // A branch between buffers that are not allocations is left alone.
  func.func @not_allocations(%c: i1, %s: memref<4xf64>, %t: memref<4xf64>) -> f64 {
    %buf = arith.select %c, %s, %t : memref<4xf64>
    %r = affine.load %buf[0] : memref<4xf64>
    return %r : f64
  }

  // The same choice between two buffers the kernel was handed, not allocated:
  // `const real_t wx = (c == 0) ? Bct(dx,qx) : Bot(dx,qx)` in MFEM's H(curl)/H(div)
  // kernels arrives as an affine.if yielding one of two pointer arguments, and
  // the access sits inside a GPU wrapper. Nothing owns those buffers, but the
  // kernel's raising has no tensor standing for one of two buffers, so the
  // access is done in each arm here as well.
  func.func @args_in_wrapper(%bc: !llvm.ptr, %bo: !llvm.ptr, %out: memref<?xf64>, %n: index) {
    %c1 = arith.constant 1 : index
    "enzymexla.gpu_wrapper"(%n, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%e) = (0) to (symbol(%n)) {
        %acc = affine.for %c = 0 to 3 iter_args(%a = %e) -> index {
          %p = affine.if #set(%c) -> !llvm.ptr {
            affine.yield %bc : !llvm.ptr
          } else {
            affine.yield %bo : !llvm.ptr
          }
          %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
          %v = affine.load %m[%c + %e * 3] : memref<?xf64>
          affine.store %v, %out[%c + %e * 3] : memref<?xf64>
          affine.yield %a : index
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @load_through_if(
// CHECK-NOT:     memref.alloca
// CHECK:         affine.for
// CHECK-NOT:       memref.cast
// CHECK:           %[[v:.+]] = affine.if #set(%{{.+}}) -> f64 {
// CHECK-NEXT:        affine.yield %arg0 : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        affine.yield %arg1 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      arith.addf %{{.+}}, %[[v]]

// CHECK-LABEL: func.func @store_through_select(
// CHECK-NOT:     memref.alloca
// CHECK:         %[[r:.+]]:2 = scf.if %arg0 -> (f64, f64) {
// CHECK-NEXT:      scf.yield %arg1, %arg2 : f64, f64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %arg2, %arg1 : f64, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[r]]#0, %[[r]]#1 : f64, f64

// CHECK-LABEL: func.func @load_through_cast(
// CHECK-NOT:     memref.alloca
// CHECK:         %[[r:.+]] = scf.if %arg0 -> (f64) {
// CHECK-NEXT:      scf.yield %arg1 : f64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %arg2 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[r]] : f64

// CHECK-LABEL: func.func @arm_with_effect(
// CHECK:         %[[s:.+]] = memref.alloca() : memref<4xf64>
// CHECK:         %[[t:.+]] = memref.alloca() : memref<4xf64>
// CHECK:         scf.if %arg0 -> (memref<4xf64>) {
// CHECK-NEXT:      affine.store %arg1, %arg2[]
// CHECK:         %[[r:.+]] = scf.if %arg0 -> (f64) {
// CHECK-NEXT:      affine.load %[[s]][0]
// CHECK:         } else {
// CHECK-NEXT:      affine.load %[[t]][0]
// CHECK:         return %[[r]] : f64

// CHECK-LABEL: func.func @not_allocations(
// CHECK:         %[[buf:.+]] = arith.select %arg0, %arg1, %arg2
// CHECK-NEXT:    affine.load %[[buf]][0]

// CHECK:    func.func @args_in_wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: memref<?xf64>, %arg3: index) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%arg3, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      affine.parallel (%arg4) = (0) to (symbol(%arg3)) {
// CHECK-NEXT:        %1 = affine.for %arg5 = 0 to 3 iter_args(%arg6 = %arg4) -> (index) {
// CHECK-NEXT:          %2 = affine.if #set(%arg5) -> f64 {
// CHECK-NEXT:            %3 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:            %4 = affine.load %3[%arg5 + %arg4 * 3] : memref<?xf64>
// CHECK-NEXT:            affine.yield %4 : f64
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %3 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:            %4 = affine.load %3[%arg5 + %arg4 * 3] : memref<?xf64>
// CHECK-NEXT:            affine.yield %4 : f64
// CHECK-NEXT:          }
// CHECK-NEXT:          affine.store %2, %arg2[%arg5 + %arg4 * 3] : memref<?xf64>
// CHECK-NEXT:          affine.yield %arg6 : index
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
