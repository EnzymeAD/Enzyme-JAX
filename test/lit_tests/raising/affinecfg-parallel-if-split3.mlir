// RUN: enzymexlamlir-opt %s --affine-cfg -allow-unregistered-dialect | FileCheck %s

#set = affine_set<(d0) : (d0 - 10 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 - 20 >= 0, 30 - d1 >= 0)>

module {
  func.func private @foo() {
    affine.parallel (%i, %j) = (0, 0) to (100, 100) {
      affine.if #set(%i) {
        "test.test_if_i_ge_10_and_j__all_"(%i, %j): (index, index) -> (index)
      }
      affine.if #set1(%i, %j) {
        "test.test_if_i_ge_20_and_j_le_30"(%i, %j): (index, index) -> (index)
      }
    }
    return
  }

  // TODO This can also work but needs special handling for else blocks
  //
  // func.func private @foo() {
  //   affine.parallel (%i, %j) = (0, 0) to (100, 100) {
  //     affine.if #set(%i) {
  //       "test.test_if_i_ge_10_and_j__all_"(%i, %j): (index, index) -> (index)
  //     } else {
  //       "test.test_if_i_lt_10_and_j__all_"(%i, %j): (index, index) -> (index)
  //     }
  //     affine.if #set1(%i, %j) {
  //       "test.test_if_i_ge_20_and_j_le_30"(%i, %j): (index, index) -> (index)
  //     } else {
  //       "test.test_if_not_i_lt_20_and_j_gt_30"(%i, %j): (index, index) -> (index)
  //     }
  //   }
  //   return
  // }
}

// CHECK-LABEL:   func.func private @foo() {
// CHECK:           affine.parallel (%[[VAL_0:.*]], %[[VAL_1:.*]]) = (10, 0) to (20, 100) {
// CHECK:             %[[VAL_2:.*]] = "test.test_if_i_ge_10_and_j__all_"(%[[VAL_0]], %[[VAL_1]]) : (index, index) -> index
// CHECK:           }
// CHECK:           affine.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]]) = (20, 0) to (100, 31) {
// CHECK:             %[[VAL_5:.*]] = "test.test_if_i_ge_10_and_j__all_"(%[[VAL_3]], %[[VAL_4]]) : (index, index) -> index
// CHECK:             %[[VAL_6:.*]] = "test.test_if_i_ge_20_and_j_le_30"(%[[VAL_3]], %[[VAL_4]]) : (index, index) -> index
// CHECK:           }
// CHECK:           affine.parallel (%[[VAL_7:.*]], %[[VAL_8:.*]]) = (20, 31) to (100, 100) {
// CHECK:             %[[VAL_9:.*]] = "test.test_if_i_ge_10_and_j__all_"(%[[VAL_7]], %[[VAL_8]]) : (index, index) -> index
// CHECK:           }
// CHECK:           return
// CHECK:         }

