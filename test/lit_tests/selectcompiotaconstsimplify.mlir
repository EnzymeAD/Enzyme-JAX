// RUN: enzymexlamlir-opt --enzyme-hlo-opt --split-input-file %s | FileCheck %s

// iota `op` const

// 0 1 2 3 4 5 6 7 8 9 
// + + + + + - - - - - ... +5/20 
module {
  func.func @comp_lt(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64>{
    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
    %c = stablehlo.constant dense<5> : tensor<20xi64>    
    %1 = stablehlo.compare  LT, %0, %c : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<20xi1>, tensor<20xi64>
    return %2 : tensor<20xi64>
  }
}

//CHECK:        func.func @comp_lt(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
//CHECK-NEXT:     %[[v0:[^ ]*]] = stablehlo.slice %arg0 [0:5] : (tensor<20xi64>) -> tensor<5xi64>
//CHECK-NEXT:     %[[v1:[^ ]*]] = stablehlo.slice %arg1 [5:20] : (tensor<20xi64>) -> tensor<15xi64>
//CHECK-NEXT:     %[[v2:[^ ]*]] = stablehlo.concatenate %[[v0]], %[[v1]], dim = 0 : (tensor<5xi64>, tensor<15xi64>) -> tensor<20xi64>
//CHECK-NEXT:     return %[[v2]] : tensor<20xi64>
//CHECK-NEXT:   }

// -----

// 0 1 2 3 4 5 6 7 8 9 
// - - - - - - + + + + ... -6/20

module {
  func.func @comp_gt(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
    %c = stablehlo.constant dense<5> : tensor<20xi64>    
    %1 = stablehlo.compare GT, %0, %c : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<20xi1>, tensor<20xi64>
    return %2 : tensor<20xi64>
  }
}

//CHECK:        func.func @comp_gt(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
//CHECK-NEXT:     %[[v0:[^ ]*]] = stablehlo.slice %arg1 [0:6] : (tensor<20xi64>) -> tensor<6xi64>
//CHECK-NEXT:     %[[v1:[^ ]*]] = stablehlo.slice %arg0 [6:20] : (tensor<20xi64>) -> tensor<14xi64>
//CHECK-NEXT:     %[[v2:[^ ]*]] = stablehlo.concatenate %[[v0]], %[[v1]], dim = 0 : (tensor<6xi64>, tensor<14xi64>) -> tensor<20xi64>
//CHECK-NEXT:     return %[[v2]] : tensor<20xi64>
//CHECK-NEXT:   }

// -----

// 0 1 2 3 4 5 6 7 8 9 
// + + + + + + - - - - ... +6/20 

module {
  func.func @comp_le(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
    %c = stablehlo.constant dense<5> : tensor<20xi64>    
    %1 = stablehlo.compare LE, %0, %c : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<20xi1>, tensor<20xi64>
    return %2 : tensor<20xi64>
  }
}

//CHECK:        func.func @comp_le(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
//CHECK-NEXT:     %[[v0:[^ ]*]] = stablehlo.slice %arg0 [0:6] : (tensor<20xi64>) -> tensor<6xi64>
//CHECK-NEXT:     %[[v1:[^ ]*]] = stablehlo.slice %arg1 [6:20] : (tensor<20xi64>) -> tensor<14xi64>
//CHECK-NEXT:     %[[v2:[^ ]*]] = stablehlo.concatenate %[[v0]], %[[v1]], dim = 0 : (tensor<6xi64>, tensor<14xi64>) -> tensor<20xi64>
//CHECK-NEXT:     return %[[v2]] : tensor<20xi64>
//CHECK-NEXT:   }

// -----

// 0 1 2 3 4 5 6 7 8 9 
// - - - - - + + + + + ... -5/20

module {
  func.func @comp_ge(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64>{
    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
    %c = stablehlo.constant dense<5> : tensor<20xi64>    
    %1 = stablehlo.compare  GE, %0, %c : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<20xi1>, tensor<20xi64>
    return %2 : tensor<20xi64>
  }
}

//CHECK:        func.func @comp_ge(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
//CHECK-NEXT:     %[[v0:[^ ]*]] = stablehlo.slice %arg1 [0:5] : (tensor<20xi64>) -> tensor<5xi64>
//CHECK-NEXT:     %[[v1:[^ ]*]] = stablehlo.slice %arg0 [5:20] : (tensor<20xi64>) -> tensor<15xi64>
//CHECK-NEXT:     %[[v2:[^ ]*]] = stablehlo.concatenate %[[v0]], %[[v1]], dim = 0 : (tensor<5xi64>, tensor<15xi64>) -> tensor<20xi64>
//CHECK-NEXT:     return %[[v2]] : tensor<20xi64>
//CHECK-NEXT:   }

// -----

// 0 1 2 3 4 5 6 7 8 9 
// - - - - - + - - - -... -5/20

module {
  func.func @comp_eq(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64>{
    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
    %c = stablehlo.constant dense<5> : tensor<20xi64>    
    %1 = stablehlo.compare  EQ, %0, %c : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<20xi1>, tensor<20xi64>
    return %2 : tensor<20xi64>
  }
}

//CHECK:        func.func @comp_eq(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
//CHECK-NEXT:     %[[v0:[^ ]*]] = stablehlo.slice %arg1 [0:5] : (tensor<20xi64>) -> tensor<5xi64>
//CHECK-NEXT:     %[[v1:[^ ]*]] = stablehlo.slice %arg0 [5:6] : (tensor<20xi64>) -> tensor<1xi64>
//CHECK-NEXT:     %[[v2:[^ ]*]] = stablehlo.slice %arg1 [6:20] : (tensor<20xi64>) -> tensor<14xi64>
//CHECK-NEXT:     %[[v3:[^ ]*]] = stablehlo.concatenate %[[v0]], %[[v1]], %[[v2]], dim = 0 : (tensor<5xi64>, tensor<1xi64>, tensor<14xi64>) -> tensor<20xi64>
//CHECK-NEXT:     return %[[v3]] : tensor<20xi64>
//CHECK-NEXT:   }

// -----

// 0 1 2 3 4 5 6 7 8 9 
// + + + + + - + + + + ... +5/20

module {
  func.func @comp_ne(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64>{
    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
    %c = stablehlo.constant dense<5> : tensor<20xi64>    
    %1 = stablehlo.compare  NE, %0, %c : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<20xi1>, tensor<20xi64>
    return %2 : tensor<20xi64>
  }
}

//CHECK:        func.func @comp_ne(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
//CHECK-NEXT:     %[[v0:[^ ]*]] = stablehlo.slice %arg0 [0:5] : (tensor<20xi64>) -> tensor<5xi64>
//CHECK-NEXT:     %[[v1:[^ ]*]] = stablehlo.slice %arg1 [5:6] : (tensor<20xi64>) -> tensor<1xi64>
//CHECK-NEXT:     %[[v2:[^ ]*]] = stablehlo.slice %arg0 [6:20] : (tensor<20xi64>) -> tensor<14xi64>
//CHECK-NEXT:     %[[v3:[^ ]*]] = stablehlo.concatenate %[[v0]], %[[v1]], %[[v2]], dim = 0 : (tensor<5xi64>, tensor<1xi64>, tensor<14xi64>) -> tensor<20xi64>
//CHECK-NEXT:     return %[[v3]] : tensor<20xi64>
//CHECK-NEXT:   }

// ----

// const `op` iota

// 0 1 2 3 4 5 6 7 8 9 
// - - - - - - + + + + ... -6/20 
module {
  func.func @comp_lt(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64>{
    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
    %c = stablehlo.constant dense<5> : tensor<20xi64>    
    %1 = stablehlo.compare  LT, %c, %0 : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<20xi1>, tensor<20xi64>
    return %2 : tensor<20xi64>
  }
}

//CHECK:        func.func @comp_lt(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
//CHECK-NEXT:     %[[v0:[^ ]*]] = stablehlo.slice %arg1 [0:6] : (tensor<20xi64>) -> tensor<6xi64>
//CHECK-NEXT:     %[[v1:[^ ]*]] = stablehlo.slice %arg0 [6:20] : (tensor<20xi64>) -> tensor<14xi64>
//CHECK-NEXT:     %[[v2:[^ ]*]] = stablehlo.concatenate %[[v0]], %[[v1]], dim = 0 : (tensor<6xi64>, tensor<14xi64>) -> tensor<20xi64>
//CHECK-NEXT:     return %[[v2]] : tensor<20xi64>
//CHECK-NEXT:   }

// -----

// 0 1 2 3 4 5 6 7 8 9 
// + + + + + - - - - - ... -6/20

module {
  func.func @comp_gt(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
    %c = stablehlo.constant dense<5> : tensor<20xi64>    
    %1 = stablehlo.compare GT, %c, %0 : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<20xi1>, tensor<20xi64>
    return %2 : tensor<20xi64>
  }
}

//CHECK:        func.func @comp_gt(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
//CHECK-NEXT:     %[[v0:[^ ]*]] = stablehlo.slice %arg0 [0:5] : (tensor<20xi64>) -> tensor<5xi64>
//CHECK-NEXT:     %[[v1:[^ ]*]] = stablehlo.slice %arg1 [5:20] : (tensor<20xi64>) -> tensor<15xi64>
//CHECK-NEXT:     %[[v2:[^ ]*]] = stablehlo.concatenate %[[v0]], %[[v1]], dim = 0 : (tensor<5xi64>, tensor<15xi64>) -> tensor<20xi64>
//CHECK-NEXT:     return %[[v2]] : tensor<20xi64>
//CHECK-NEXT:   }

// 0 1 2 3 4 5 6 7 8 9 
// - - - - - + + + + + ... -5/20 
module {
  func.func @comp_le(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64>{
    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
    %c = stablehlo.constant dense<5> : tensor<20xi64>    
    %1 = stablehlo.compare  LE, %c, %0 : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<20xi1>, tensor<20xi64>
    return %2 : tensor<20xi64>
  }
}

//CHECK:        func.func @comp_le(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
//CHECK-NEXT:     %[[v0:[^ ]*]] = stablehlo.slice %arg1 [0:5] : (tensor<20xi64>) -> tensor<5xi64>
//CHECK-NEXT:     %[[v1:[^ ]*]] = stablehlo.slice %arg0 [5:20] : (tensor<20xi64>) -> tensor<15xi64>
//CHECK-NEXT:     %[[v2:[^ ]*]] = stablehlo.concatenate %[[v0]], %[[v1]], dim = 0 : (tensor<5xi64>, tensor<15xi64>) -> tensor<20xi64>
//CHECK-NEXT:     return %[[v2]] : tensor<20xi64>
//CHECK-NEXT:   }

// -----

// 0 1 2 3 4 5 6 7 8 9 
// + + + + + + - - - - ... +6/20

module {
  func.func @comp_ge(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
    %c = stablehlo.constant dense<5> : tensor<20xi64>    
    %1 = stablehlo.compare GE, %c, %0 : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<20xi1>, tensor<20xi64>
    return %2 : tensor<20xi64>
  }
}

//CHECK:        func.func @comp_ge(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64> {
//CHECK-NEXT:     %[[v0:[^ ]*]] = stablehlo.slice %arg0 [0:6] : (tensor<20xi64>) -> tensor<6xi64>
//CHECK-NEXT:     %[[v1:[^ ]*]] = stablehlo.slice %arg1 [6:20] : (tensor<20xi64>) -> tensor<14xi64>
//CHECK-NEXT:     %[[v2:[^ ]*]] = stablehlo.concatenate %[[v0]], %[[v1]], dim = 0 : (tensor<6xi64>, tensor<14xi64>) -> tensor<20xi64>
//CHECK-NEXT:     return %[[v2]] : tensor<20xi64>
//CHECK-NEXT:   }
