// RUN: enzymexlamlir-opt --enzyme-hlo-opt --split-input-file %s | FileCheck %s

// iota `op` const

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

// -----

module {
  func.func @comp_eq(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64>{
    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
    %c = stablehlo.constant dense<5> : tensor<20xi64>    
    %1 = stablehlo.compare  EQ, %c, %0 : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
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

module {
  func.func @comp_ne(%arg0: tensor<20xi64>, %arg1: tensor<20xi64>) -> tensor<20xi64>{
    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
    %c = stablehlo.constant dense<5> : tensor<20xi64>    
    %1 = stablehlo.compare  NE, %c, %0 : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
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

// -----

// test in higher dimensions

module {
  func.func @comp_lt(%arg0: tensor<2x5xi64>, %arg1: tensor<2x5xi64>) -> tensor<2x5xi64>{
    %0 = stablehlo.iota dim = 1 : tensor<2x5xi64>
    %c = stablehlo.constant dense<3> : tensor<2x5xi64>    
    %1 = stablehlo.compare  LT, %0, %c : (tensor<2x5xi64>, tensor<2x5xi64>) -> tensor<2x5xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<2x5xi1>, tensor<2x5xi64>
    return %2 : tensor<2x5xi64>
  }
}

//CHECK:      func.func @comp_lt(%arg0: tensor<2x5xi64>, %arg1: tensor<2x5xi64>) -> tensor<2x5xi64> {
//CHECK-NEXT:   %[[v0:[^ ]*]] = stablehlo.slice %arg0 [0:2, 0:3] : (tensor<2x5xi64>) -> tensor<2x3xi64>
//CHECK-NEXT:   %[[v1:[^ ]*]] = stablehlo.slice %arg1 [0:2, 3:5] : (tensor<2x5xi64>) -> tensor<2x2xi64>
//CHECK-NEXT:   %[[v2:[^ ]*]] = stablehlo.concatenate %[[v0]], %[[v1]], dim = 1 : (tensor<2x3xi64>, tensor<2x2xi64>) -> tensor<2x5xi64>
//CHECK-NEXT:   return %[[v2]] : tensor<2x5xi64>
//CHECK-NEXT: }

// -----

// test for out of range indices

module {
  func.func @comp_lt(%arg0: tensor<2x5xi64>, %arg1: tensor<2x5xi64>) -> tensor<2x5xi64>{
    %0 = stablehlo.iota dim = 1 : tensor<2x5xi64>
    %c = stablehlo.constant dense<-1> : tensor<2x5xi64>    
    %1 = stablehlo.compare  LT, %0, %c : (tensor<2x5xi64>, tensor<2x5xi64>) -> tensor<2x5xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<2x5xi1>, tensor<2x5xi64>
    return %2 : tensor<2x5xi64>
  }
}

//CHECK:      func.func @comp_lt(%arg0: tensor<2x5xi64>, %arg1: tensor<2x5xi64>) -> tensor<2x5xi64> {
//CHECK-NEXT:   return %arg1 : tensor<2x5xi64>
//CHECK-NEXT: }

// -----

module {
  func.func @comp_lt(%arg0: tensor<2x5xi64>, %arg1: tensor<2x5xi64>) -> tensor<2x5xi64>{
    %0 = stablehlo.iota dim = 1 : tensor<2x5xi64>
    %c = stablehlo.constant dense<5> : tensor<2x5xi64>    
    %1 = stablehlo.compare  LT, %0, %c : (tensor<2x5xi64>, tensor<2x5xi64>) -> tensor<2x5xi1>
    %2 = stablehlo.select %1, %arg0, %arg1: tensor<2x5xi1>, tensor<2x5xi64>
    return %2 : tensor<2x5xi64>
  }
}

//CHECK:      func.func @comp_lt(%arg0: tensor<2x5xi64>, %arg1: tensor<2x5xi64>) -> tensor<2x5xi64> {
//CHECK-NEXT:   return %arg0 : tensor<2x5xi64>
//CHECK-NEXT: }