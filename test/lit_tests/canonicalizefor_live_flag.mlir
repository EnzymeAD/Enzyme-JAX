// RUN: enzymexlamlir-opt %s --cse --canonicalize-scf-for | FileCheck %s

// The while-converted grid-stride loop carries an i1 "live" flag: true on
// entry, re-computed as `live ? f(counter) < N : false` with the counter
// advancing once per live iteration. Every earlier test held while the flag
// is true and f is nondecreasing in the counter, so the flag is just the
// previous iteration's test: `iv == lb || f(cnt0 + iv - lb - 1) < N`, a pure
// function of the induction variable, and the flag and counter fold away.
func.func @gridstride(%N: i32, %ipt: i32, %tid: i32, %woff: i32, %buf: memref<?xf64>) -> f64 {
  %bs = arith.constant 256 : i32
  %cst = arith.constant 0.0 : f64
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %r:2 = scf.while (%acc = %cst, %idx = %c0_i32) : (f64, i32) -> (f64, i32) {
    %t0 = arith.addi %idx, %woff overflow<nsw> : i32
    %t1 = arith.muli %t0, %bs : i32
    %i = arith.addi %t1, %tid : i32
    %inb = arith.cmpi slt, %i, %N : i32
    %next = arith.addi %idx, %c1_i32 overflow<nsw, nuw> : i32
    %ne = arith.cmpi ne, %next, %ipt : i32
    %acc2 = scf.if %inb -> (f64) {
      %ii = arith.index_cast %i : i32 to index
      %v = memref.load %buf[%ii] : memref<?xf64>
      %s = arith.addf %acc, %v : f64
      scf.yield %s : f64
    } else {
      scf.yield %acc : f64
    }
    %cond = arith.andi %inb, %ne : i1
    scf.condition(%cond) %acc2, %next : f64, i32
  } do {
  ^bb0(%a: f64, %i2: i32):
    scf.yield %a, %i2 : f64, i32
  }
  return %r#0 : f64
}

// CHECK:    func.func @gridstride(%[[a1:.+]]: i32, %[[a2:.+]]: i32, %[[a3:.+]]: i32, %[[a4:.+]]: i32, %[[a5:.+]]: memref<?xf64>) -> f64 {
// CHECK-NEXT:    %[[a6:.+]] = arith.constant false
// CHECK-NEXT:    %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:    %[[a7:.+]] = arith.constant 256 : i32
// CHECK-NEXT:    %[[a8:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[a9:.+]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[a10:.+]] = ub.poison : f64
// CHECK-NEXT:    %[[a11:.+]] = ub.poison : i32
// CHECK-NEXT:    %[[a12:.+]] = arith.maxsi %[[a2]], %[[a9]] : i32
// CHECK-NEXT:    %[[a13:.+]] = arith.addi %[[a12]], %[[a9]] : i32
// CHECK-NEXT:    %[[a14:.+]]:3 = scf.for %[[a15:.+]] = %[[a9]] to %[[a13]] step %[[a9]] iter_args(%[[a16:.+]] = %[[a8]], %[[a17:.+]] = %[[a10]], %[[a18:.+]] = %[[a11]]) -> (f64, f64, i32)  : i32 {
// CHECK-NEXT:      %[[a19:.+]] = arith.addi %[[a15]], %c-1_i32 : i32
// CHECK-NEXT:      %[[a20:.+]] = arith.addi %[[a19]], %c-1_i32 : i32
// CHECK-NEXT:      %[[a21:.+]] = arith.addi %[[a20]], %[[a4]] : i32
// CHECK-NEXT:      %[[a22:.+]] = arith.muli %[[a21]], %[[a7]] : i32
// CHECK-NEXT:      %[[a23:.+]] = arith.addi %[[a22]], %[[a3]] : i32
// CHECK-NEXT:      %[[a24:.+]] = arith.cmpi slt, %[[a23]], %[[a1]] : i32
// CHECK-NEXT:      %[[a25:.+]] = arith.cmpi eq, %[[a15]], %[[a9]] : i32
// CHECK-NEXT:      %[[a26:.+]] = arith.ori %[[a25]], %[[a24]] : i1
// CHECK-NEXT:      %[[a27:.+]]:3 = scf.if %[[a26]] -> (f64, i32, i1) {
// CHECK-NEXT:        %[[a28:.+]] = arith.addi %[[a19]], %[[a4]] overflow<nsw> : i32
// CHECK-NEXT:        %[[a29:.+]] = arith.muli %[[a28]], %[[a7]] : i32
// CHECK-NEXT:        %[[a30:.+]] = arith.addi %[[a29]], %[[a3]] : i32
// CHECK-NEXT:        %[[a31:.+]] = arith.cmpi slt, %[[a30]], %[[a1]] : i32
// CHECK-NEXT:        %[[a32:.+]] = arith.addi %[[a19]], %[[a9]] overflow<nsw, nuw> : i32
// CHECK-NEXT:        %[[a33:.+]] = scf.if %[[a31]] -> (f64) {
// CHECK-NEXT:          %[[a34:.+]] = arith.index_cast %[[a30]] : i32 to index
// CHECK-NEXT:          %[[a35:.+]] = memref.load %[[a5]][%[[a34]]] : memref<?xf64>
// CHECK-NEXT:          %[[a36:.+]] = arith.addf %[[a16]], %[[a35]] : f64
// CHECK-NEXT:          scf.yield %[[a36]] : f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %[[a16]] : f64
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[a33]], %[[a32]], %[[a31]] : f64, i32, i1
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %[[a17]], %[[a18]], %[[a6]] : f64, i32, i1
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[a37:.+]] = arith.cmpi slt, %[[a15]], %[[a2]] : i32
// CHECK-NEXT:      %[[a38:.+]] = arith.andi %[[a37]], %[[a27]]#2 : i1
// CHECK-NEXT:      %[[a39:.+]] = scf.if %[[a38]] -> (f64) {
// CHECK-NEXT:        scf.yield %[[a27]]#0 : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %[[a10]] : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[a39]], %[[a27]]#0, %[[a27]]#1 : f64, f64, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a14]]#1 : f64
// CHECK-NEXT:  }
