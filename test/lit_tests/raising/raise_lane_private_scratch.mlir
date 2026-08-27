// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="prefer_while_raising=false err_if_not_fully_raised=true" | FileCheck %s

module {
  llvm.func @kern(%outp: !llvm.ptr, %inp: !llvm.ptr, %d: i32, %q: i32, %nei: i32, %bdi: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c8_i32 = arith.constant 8 : i32
    %cst0 = arith.constant 0.0 : f64
    %ne = arith.index_cast %nei : i32 to index
    %bdc = arith.minsi %bdi, %c8_i32 : i32
    %bd = arith.index_cast %bdc : i32 to index
    %out = "enzymexla.pointer2memref"(%outp) : (!llvm.ptr) -> memref<?xf64>
    %in = "enzymexla.pointer2memref"(%inp) : (!llvm.ptr) -> memref<?xf64>
    %0 = "enzymexla.gpu_wrapper"(%ne, %c1, %c1, %bd, %bd, %c1) ({
      affine.parallel (%e) = (0) to (symbol(%ne)) {
        affine.parallel (%ty, %tx) = (0, 0) to (symbol(%bd), symbol(%bd)) {
          %loc = memref.alloca() : memref<2xf64>
          %di = arith.index_cast %d : i32 to index
          %qi = arith.index_cast %q : i32 to index
          %bdI = arith.index_cast %bdc : i32 to index
          scf.for %i = %ty to %qi step %bdI {
            scf.for %j = %tx to %qi step %bdI {
              affine.store %cst0, %loc[0] : memref<2xf64>
              affine.store %cst0, %loc[1] : memref<2xf64>
              %qq = arith.muli %qi, %qi : index
              %eoff = arith.muli %e, %qq : index
              %roff = arith.muli %i, %qi : index
              %idx0 = arith.addi %eoff, %roff : index
              %idx = arith.addi %idx0, %j : index
              %v = memref.load %in[%idx] : memref<?xf64>
              %a0 = affine.load %loc[0] : memref<2xf64>
              %s0 = arith.addf %a0, %v : f64
              affine.store %s0, %loc[0] : memref<2xf64>
              %two = arith.constant 2.0 : f64
              %v2 = arith.mulf %v, %two : f64
              %a1 = affine.load %loc[1] : memref<2xf64>
              %s1 = arith.addf %a1, %v2 : f64
              affine.store %s1, %loc[1] : memref<2xf64>
              %r0 = affine.load %loc[0] : memref<2xf64>
              %r1 = affine.load %loc[1] : memref<2xf64>
              %sum = arith.addf %r0, %r1 : f64
              memref.store %sum, %out[%idx] : memref<?xf64>
            }
          }
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    llvm.return
  }
}

// A per-thread local array inside the lane-batched parallel must not
// collapse to one lane's value: the buffer gains one dimension per lane
// axis and every access is indexed by the lane IVs.
// CHECK-LABEL: llvm.func @kern(
// CHECK-NOT: failed to raise
// CHECK: enzymexla.xla_wrapper @rxla$raised
