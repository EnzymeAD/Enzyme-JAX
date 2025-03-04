// RUN: enzymexlamlir-opt --canonicalize-scf-for %s | FileCheck %s

func.func @test_if_yield_movement(%cond: i1, %a: i32, %b: i32) -> i32 {
  %0 = scf.if %cond -> (i32) {
    %1 = arith.addi %a, %b : i32
    %2 = arith.muli %1, %1 : i32  // This should be moved outside the if
    scf.yield %2 : i32
  } else {
    %1 = arith.subi %a, %b : i32
    %2 = arith.muli %1, %1 : i32  // This should be moved outside the if
    scf.yield %2 : i32
  }
  
  return %0 : i32
}
// CHECK-LABEL:   func.func @test_if_yield_movement(
// CHECK-SAME:                                      %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                      %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                      %[[VAL_2:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_3:.*]]:2 = scf.if %[[VAL_0]] -> (i32, i32) {
// CHECK:             %[[VAL_4:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             scf.yield %[[VAL_4]], %[[VAL_4]] : i32, i32
// CHECK:           } else {
// CHECK:             %[[VAL_5:.*]] = arith.subi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             scf.yield %[[VAL_5]], %[[VAL_5]] : i32, i32
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = arith.muli %[[VAL_7:.*]]#0, %[[VAL_7]]#1 : i32
// CHECK:           return %[[VAL_6]] : i32
// CHECK:         }

func.func @test_if_yield_movement2(%cond: i1, %a: i32, %b: i32) -> i32 {
  %0 = scf.if %cond -> (i32) {
    %1 = arith.addi %a, %b : i32
    %2 = arith.muli %1, %1 : i32  
    scf.yield %2 : i32
  } else {
    %0 = arith.subi %a, %b : i32
    %1 = arith.addi %a, %0 : i32
    %2 = arith.muli %1, %1 : i32
    scf.yield %2 : i32
  }
  
  return %0 : i32
}
// CHECK-LABEL:   func.func @test_if_yield_movement2(
// CHECK-SAME:                                       %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                       %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                       %[[VAL_2:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_3:.*]]:2 = scf.if %[[VAL_0]] -> (i32, i32) {
// CHECK:             scf.yield %[[VAL_1]], %[[VAL_2]] : i32, i32
// CHECK:           } else {
// CHECK:             %[[VAL_4:.*]] = arith.subi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             scf.yield %[[VAL_1]], %[[VAL_4]] : i32, i32
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_6:.*]]#0, %[[VAL_6]]#1 : i32
// CHECK:           %[[VAL_7:.*]] = arith.muli %[[VAL_5]], %[[VAL_5]] : i32
// CHECK:           return %[[VAL_7]] : i32
// CHECK:         }

func.func @test_if_yield_movement3(%cond: i1, %a: i32, %b: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %0 = scf.if %cond -> (i32) {
    %1 = arith.addi %a, %c1 : i32
    %2 = arith.muli %1, %1 : i32 
    scf.yield %2 : i32
  } else {
    %0 = arith.subi %a, %b : i32
    %1 = arith.addi %a, %0 : i32
    %2 = arith.muli %1, %1 : i32
    scf.yield %2 : i32
  }
  
  return %0 : i32
}
// CHECK-LABEL:   func.func @test_if_yield_movement3(
// CHECK-SAME:                                       %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                       %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                       %[[VAL_2:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]]:2 = scf.if %[[VAL_0]] -> (i32, i32) {
// CHECK:             scf.yield %[[VAL_1]], %[[VAL_3]] : i32, i32
// CHECK:           } else {
// CHECK:             %[[VAL_5:.*]] = arith.subi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             scf.yield %[[VAL_1]], %[[VAL_5]] : i32, i32
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_7:.*]]#0, %[[VAL_7]]#1 : i32
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_6]], %[[VAL_6]] : i32
// CHECK:           return %[[VAL_8]] : i32
// CHECK:         }

func.func @test_if_yield_movement4(%cond: i1, %a: i32, %b: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %0 = scf.if %cond -> (i32) {
    %0 = arith.subi %a, %b : i32
    %1 = arith.addi %a, %0 : i32
    %2 = arith.muli %1, %1 : i32 
    scf.yield %2 : i32
  } else {
    %1 = arith.addi %a, %c1 : i32
    %2 = arith.muli %1, %1 : i32 
    scf.yield %2 : i32
  }
  
  return %0 : i32
}
// CHECK-LABEL:   func.func @test_if_yield_movement4(
// CHECK-SAME:                                       %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                       %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                       %[[VAL_2:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]]:2 = scf.if %[[VAL_0]] -> (i32, i32) {
// CHECK:             %[[VAL_5:.*]] = arith.subi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             scf.yield %[[VAL_1]], %[[VAL_5]] : i32, i32
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_1]], %[[VAL_3]] : i32, i32
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_7:.*]]#0, %[[VAL_7]]#1 : i32
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_6]], %[[VAL_6]] : i32
// CHECK:           return %[[VAL_8]] : i32
// CHECK:         }

func.func @test_if_yield_movement5(%cond: i1, %a: i32, %b: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %0 = scf.if %cond -> (i32) {
    %1 = arith.addi %a, %c1 : i32
    %2 = arith.muli %1, %1 : i32 
    scf.yield %2 : i32
  } else {
    %1 = arith.addi %a, %c1 : i32
    %2 = arith.muli %1, %1 : i32 
    scf.yield %2 : i32
  }
  
  return %0 : i32
}
// CHECK-LABEL:   func.func @test_if_yield_movement5(
// CHECK-SAME:                                       %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                       %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                       %[[VAL_2:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]]:2 = scf.if %[[VAL_0]] -> (i32, i32) {
// CHECK:             scf.yield %[[VAL_1]], %[[VAL_3]] : i32, i32
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_1]], %[[VAL_3]] : i32, i32
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_6:.*]]#0, %[[VAL_6]]#1 : i32
// CHECK:           %[[VAL_7:.*]] = arith.muli %[[VAL_5]], %[[VAL_5]] : i32
// CHECK:           return %[[VAL_7]] : i32
// CHECK:         }

func.func @test_if_yield_movement6(%cond: i1, %a: i32, %b: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %buffer = memref.alloc() : memref<1xi32>
  %add = arith.addi %a, %c1 : i32
  memref.store %add, %buffer[%c0] : memref<1xi32>
    
  %0 = scf.if %cond -> (i32) {
    // Load instead of direct addi use
    %1 = memref.load %buffer[%c0] : memref<1xi32>
    %2 = arith.muli %1, %1 : i32
    scf.yield %2 : i32
  } else {
    %1 = memref.load %buffer[%c0] : memref<1xi32>
    %2 = arith.muli %1, %1 : i32
    scf.yield %2 : i32
  }
  return %0 : i32
}
// CHECK-LABEL:   func.func @test_if_yield_movement6(
// CHECK-SAME:                                       %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                       %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                       %[[VAL_2:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_1]], %[[VAL_3]] : i32
// CHECK:           memref.store %[[VAL_6]], %[[VAL_5]]{{\[}}%[[VAL_4]]] : memref<1xi32>
// CHECK:           %[[VAL_7:.*]]:2 = scf.if %[[VAL_0]] -> (i32, i32) {
// CHECK:             %[[VAL_8:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_4]]] : memref<1xi32>
// CHECK:             scf.yield %[[VAL_8]], %[[VAL_8]] : i32, i32
// CHECK:           } else {
// CHECK:             %[[VAL_9:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_4]]] : memref<1xi32>
// CHECK:             scf.yield %[[VAL_9]], %[[VAL_9]] : i32, i32
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = arith.muli %[[VAL_11:.*]]#0, %[[VAL_11]]#1 : i32
// CHECK:           return %[[VAL_10]] : i32
// CHECK:         }