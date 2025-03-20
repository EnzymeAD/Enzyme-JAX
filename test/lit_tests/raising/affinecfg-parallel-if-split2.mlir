// RUN: enzymexlamlir-opt %s --affine-cfg -allow-unregistered-dialect | FileCheck %s

#set = affine_set<(d0) : (d0 - 10 >= 0)>
#set1 = affine_set<(d0) : (d0 - 20 == 0)>
#set2 = affine_set<(d0) : (d0 - 30 <= 0)>

module {
  func.func private @foo() {
    affine.parallel (%i) = (0) to (100) {
      "test.test1"(%i): (index) -> (index)
      affine.if #set(%i) {
        "test.test_if_ge_10"(%i): (index) -> (index)
      }
      affine.if #set1(%i) {
        "test.test_if_eq_20"(%i): (index) -> (index)
      }
      affine.if #set2(%i) {
        "test.test_if_le_30"(%i): (index) -> (index)
      }
      "test.test2"(%i): (index) -> (index)
    }
    return
  }

  // TODO This can also work but needs special handling for else blocks
  //
  // func.func private @foo(%array: memref<85x180x18xf64, 1>, %sum_res: memref<85x180xf64, 1>) {
  //   affine.parallel (%i) = (0) to (100) {
  //     "test.test1"(%i): (index) -> (index)
  //     affine.if #set(%i) {
  //       "test.test_if_ge_10"(%i): (index) -> (index)
  //     } else {
  //       "test.test_if_lt_10"(%i): (index) -> (index)
  //     }
  //     affine.if #set1(%i) {
  //       "test.test_if_eq_20"(%i): (index) -> (index)
  //     } else {
  //       "test.test_if_ne_20"(%i): (index) -> (index)
  //     }
  //     affine.if #set2(%i) {
  //       "test.test_if_le_30"(%i): (index) -> (index)
  //     } else {
  //       "test.test_if_gt_30"(%i): (index) -> (index)
  //     }
  //     "test.test2"(%i): (index) -> (index)
  //   }
  //   return
  // }
}


// CHECK-LABEL:   func.func private @foo() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 20 : index
// CHECK:           affine.parallel (%[[VAL_1:.*]]) = (0) to (10) {
// CHECK:             %[[VAL_2:.*]] = "test.test1"(%[[VAL_1]]) : (index) -> index
// CHECK:             %[[VAL_3:.*]] = "test.test_if_le_30"(%[[VAL_1]]) : (index) -> index
// CHECK:             %[[VAL_4:.*]] = "test.test2"(%[[VAL_1]]) : (index) -> index
// CHECK:           }
// CHECK:           affine.parallel (%[[VAL_5:.*]]) = (10) to (20) {
// CHECK:             %[[VAL_6:.*]] = "test.test1"(%[[VAL_5]]) : (index) -> index
// CHECK:             %[[VAL_7:.*]] = "test.test_if_ge_10"(%[[VAL_5]]) : (index) -> index
// CHECK:             %[[VAL_8:.*]] = "test.test_if_le_30"(%[[VAL_5]]) : (index) -> index
// CHECK:             %[[VAL_9:.*]] = "test.test2"(%[[VAL_5]]) : (index) -> index
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = "test.test1"(%[[VAL_0]]) : (index) -> index
// CHECK:           %[[VAL_11:.*]] = "test.test_if_ge_10"(%[[VAL_0]]) : (index) -> index
// CHECK:           %[[VAL_12:.*]] = "test.test_if_eq_20"(%[[VAL_0]]) : (index) -> index
// CHECK:           %[[VAL_13:.*]] = "test.test_if_le_30"(%[[VAL_0]]) : (index) -> index
// CHECK:           %[[VAL_14:.*]] = "test.test2"(%[[VAL_0]]) : (index) -> index
// CHECK:           affine.parallel (%[[VAL_15:.*]]) = (21) to (31) {
// CHECK:             %[[VAL_16:.*]] = "test.test1"(%[[VAL_15]]) : (index) -> index
// CHECK:             %[[VAL_17:.*]] = "test.test_if_ge_10"(%[[VAL_15]]) : (index) -> index
// CHECK:             %[[VAL_18:.*]] = "test.test_if_le_30"(%[[VAL_15]]) : (index) -> index
// CHECK:             %[[VAL_19:.*]] = "test.test2"(%[[VAL_15]]) : (index) -> index
// CHECK:           }
// CHECK:           affine.parallel (%[[VAL_20:.*]]) = (31) to (100) {
// CHECK:             %[[VAL_21:.*]] = "test.test1"(%[[VAL_20]]) : (index) -> index
// CHECK:             %[[VAL_22:.*]] = "test.test_if_ge_10"(%[[VAL_20]]) : (index) -> index
// CHECK:             %[[VAL_23:.*]] = "test.test2"(%[[VAL_20]]) : (index) -> index
// CHECK:           }
// CHECK:           return
// CHECK:         }

