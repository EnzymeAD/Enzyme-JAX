// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(affine-cfg)" | FileCheck %s

#setblk = affine_set<(d0) : (d0 == 0)>
#set256 = affine_set<(d0) : (255 - d0 >= 0)>

module {
  func.func @red(%blk: index) {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %alloca = memref.alloca() : memref<1024xf64>
    affine.parallel (%tid) = (0) to (1024) {
      %t64 = arith.index_castui %tid : index to i64
      %sel = affine.if #setblk(%blk) -> i64 {
        affine.yield %t64 : i64
      } else {
        affine.yield %t64 : i64
      }
      "enzymexla.barrier"(%tid, %c0, %c0) : (index, index, index) -> ()
      affine.if #set256(%tid) {
        %j = arith.index_cast %sel : i64 to index
        %k = arith.addi %j, %c256 : index
        %b = memref.load %alloca[%k] : memref<1024xf64>
        memref.store %b, %alloca[%j] : memref<1024xf64>
      }
      affine.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @red
// CHECK:         affine.parallel (%[[TID:.+]]) = (0) to (1024) {
// CHECK:           "enzymexla.barrier"
// CHECK:           affine.if
// CHECK-NEXT:        %[[V:.+]] = affine.load %{{.+}}[%[[TID]] + 256]
// CHECK-NEXT:        affine.store %[[V]], %{{.+}}[%[[TID]]]
// CHECK-NEXT:      }
