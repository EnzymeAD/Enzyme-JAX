// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

#set = affine_set<(d0) : (d0 - 10 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 - 20 >= 0, 30 - d1 >= 0)>

module {
  func.func private @foo(%array: memref<85x180x18xf64, 1>, %sum_res: memref<85x180xf64, 1>) {
    affine.parallel (%i, %j) = (0, 0) to (100, 100) {
      affine.if #set(%i) {
        "test.test_if_i_ge_10_and_j__all_"(%i, %j): (index, index) -> (index)
      } else {
        "test.test_if_i_lt_10_and_j__all_"(%i, %j): (index, index) -> (index)
      }
      affine.if #set1(%i, %j) {
        "test.test_if_i_ge_20_and_j_le_30"(%i, %j): (index, index) -> (index)
      } else {
        "test.test_if_i_lt_20_and_j_gt_30"(%i, %j): (index, index) -> (index)
      }
    }
    return
  }
}
