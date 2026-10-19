// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

module {
  func.func @select_yield_dim(%ptr: !llvm.ptr, %n: i64, %v: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0_i32 = arith.constant 0 : i32
    scf.parallel (%tid) = (%c0) to (%c256) step (%c1) {
      %tidi = arith.index_castui %tid : index to i32
      %inner = arith.cmpi eq, %tidi, %c0_i32 : i32
      %if:2 = scf.if %inner -> (i32, i32) {
        scf.yield %c0_i32, %v : i32, i32
      } else {
        scf.yield %v, %c0_i32 : i32, i32
      }
      %ni = arith.index_castui %n : i64 to index
      %outer = arith.cmpi eq, %ni, %c0 : index
      %sel = arith.select %outer, %if#0, %if#1 : i32
      %selz = arith.cmpi eq, %sel, %c0_i32 : i32
      %res = scf.if %selz -> (i32) {
        %l = llvm.load %ptr : !llvm.ptr -> i32
        scf.yield %l : i32
      } else {
        scf.yield %v : i32
      }
      llvm.store %res, %ptr : i32, !llvm.ptr
    }
    return
  }
}

// %sel yields out of an if keyed off the induction variable, so it is a dim and
// never a valid affine symbol; the conditional using it must stay an scf.if
// rather than be raised and hoisted.

// CHECK:       #set = affine_set<(d0) : (d0 == 0)>
// CHECK:       #set1 = affine_set<()[s0] : (s0 == 0)>
// CHECK-LABEL:   func.func @select_yield_dim(
// CHECK-SAME:                                %[[PTR:.+]]: !llvm.ptr, %[[N:.+]]: i64, %[[V:.+]]: i32) {
// CHECK-NEXT:      %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[NIDX:.+]] = arith.index_cast %[[N]] : i64 to index
// CHECK-NEXT:      affine.parallel (%[[TID:.+]]) = (0) to (256) {
// CHECK-NEXT:        %[[IF:.+]]:2 = affine.if #set(%[[TID]]) -> (i32, i32) {
// CHECK-NEXT:          affine.yield %[[C0_I32]], %[[V]] : i32, i32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          affine.yield %[[V]], %[[C0_I32]] : i32, i32
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[SEL:.+]] = affine.if #set1()[%[[NIDX]]] -> i32 {
// CHECK-NEXT:          affine.yield %[[IF]]#0 : i32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          affine.yield %[[IF]]#1 : i32
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[SELZ:.+]] = arith.cmpi eq, %[[SEL]], %[[C0_I32]] : i32
// CHECK-NEXT:        %[[RES:.+]] = scf.if %[[SELZ]] -> (i32) {
// CHECK-NEXT:          %[[LOAD:.+]] = llvm.load %[[PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:          scf.yield %[[LOAD]] : i32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %[[V]] : i32
// CHECK-NEXT:        }
// CHECK-NEXT:        llvm.store %[[RES]], %[[PTR]] : i32, !llvm.ptr
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
