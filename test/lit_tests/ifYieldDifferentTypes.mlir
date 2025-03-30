// RUN: enzymexlamlir-opt --canonicalize-scf-for %s | FileCheck %s

module {
  func.func @type_mismatch_example(%arg0: i32, %arg1: i32, %arg2: i32, 
                                  %arg3: i32, %arg4: i32, %arg5: i32, 
                                  %arg6: i32, %arg7: i32, %arg8: i32, 
                                  %arg9: i32, %arg10: i32) -> (i32, i8) {
    %c32_i64 = arith.constant 32 : i64
    %cst_3 = arith.constant 0.000000e+00 : f32
    %cst_4 = arith.constant 0.000000e+00 : f64
    
    // Create some values for the condition and operands
    %141 = arith.cmpi eq, %arg0, %arg1 : i32
    %147 = arith.addi %arg2, %arg3 : i32
    %149 = arith.addi %arg4, %arg5 : i32
    
    // This is the problematic scf.if with type mismatch between branches
    %154:2 = scf.if %141 -> (i32, i8) {
      // Then branch - using f32
      %179 = arith.bitcast %147 : i32 to f32
      %180 = math.absf %179 : f32
      %181 = arith.cmpf olt, %cst_3, %180 {fastmathFlags = #llvm.fastmath<none>} : f32
      %182 = arith.extui %181 : i1 to i8
      scf.yield %arg10, %182 : i32, i8
    } else {
      // Else branch - using f64 (type mismatch)
      %179 = arith.extui %149 : i32 to i64
      %180 = arith.shli %179, %c32_i64 : i64
      %181 = arith.extui %147 : i32 to i64
      %182 = arith.ori %180, %181 : i64
      %183 = arith.bitcast %182 : i64 to f64
      %184 = math.absf %183 : f64
      %185 = arith.bitcast %184 : f64 to i64
      %186 = arith.trunci %185 : i64 to i32
      %187 = arith.shrui %185, %c32_i64 : i64
      %188 = arith.trunci %187 : i64 to i32
      %189 = arith.extui %188 : i32 to i64
      %190 = arith.shli %189, %c32_i64 : i64
      %191 = arith.extui %186 : i32 to i64
      %192 = arith.addi %190, %191 : i64
      %193 = arith.bitcast %192 : i64 to f64
      %194 = arith.cmpf olt, %cst_4, %193 {fastmathFlags = #llvm.fastmath<none>} : f64
      %195 = arith.extui %194 : i1 to i8
      scf.yield %188, %195 : i32, i8
    }
    
    return %154#0, %154#1 : i32, i8
  }
}

// CHECK-LABEL: func.func @type_mismatch_example(
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32, %[[ARG8:.*]]: i32, %[[ARG9:.*]]: i32, %[[ARG10:.*]]: i32) -> (i32, i8)
// CHECK: %[[C32:.*]] = arith.constant 32 : i64
// CHECK: %[[CST_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_F64:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[ARG0]], %[[ARG1]] : i32
// CHECK: %[[ADD1:.*]] = arith.addi %[[ARG2]], %[[ARG3]] : i32
// CHECK: %[[ADD2:.*]] = arith.addi %[[ARG4]], %[[ARG5]] : i32
// CHECK: %[[IF_RESULT:.*]]:2 = scf.if %[[CMP]] -> (i32, i1) {
// CHECK:   %[[BITCAST1:.*]] = arith.bitcast %[[ADD1]] : i32 to f32
// CHECK:   %[[ABSF1:.*]] = math.absf %[[BITCAST1]] : f32
// CHECK:   %[[CMPF1:.*]] = arith.cmpf olt, %[[CST_F32]], %[[ABSF1]] {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK:   scf.yield %[[ARG10]], %[[CMPF1]] : i32, i1
// CHECK: } else {
// CHECK:   %[[EXTUI1:.*]] = arith.extui %[[ADD2]] : i32 to i64
// CHECK:   %[[SHLI1:.*]] = arith.shli %[[EXTUI1]], %[[C32]] : i64
// CHECK:   %[[EXTUI2:.*]] = arith.extui %[[ADD1]] : i32 to i64
// CHECK:   %[[ORI:.*]] = arith.ori %[[SHLI1]], %[[EXTUI2]] : i64
// CHECK:   %[[BITCAST2:.*]] = arith.bitcast %[[ORI]] : i64 to f64
// CHECK:   %[[ABSF2:.*]] = math.absf %[[BITCAST2]] : f64
// CHECK:   %[[BITCAST3:.*]] = arith.bitcast %[[ABSF2]] : f64 to i64
// CHECK:   %[[TRUNCI1:.*]] = arith.trunci %[[BITCAST3]] : i64 to i32
// CHECK:   %[[SHRUI:.*]] = arith.shrui %[[BITCAST3]], %[[C32]] : i64
// CHECK:   %[[TRUNCI2:.*]] = arith.trunci %[[SHRUI]] : i64 to i32
// CHECK:   %[[CMPF2:.*]] = arith.cmpf olt, %[[CST_F64]], {{.*}} {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK:   scf.yield %[[TRUNCI2]], %[[CMPF2]] : i32, i1
// CHECK: }
// CHECK: %[[EXTUI_FINAL:.*]] = arith.extui %[[IF_RESULT]]#1 : i1 to i8
// CHECK: return %[[IF_RESULT]]#0, %[[EXTUI_FINAL]] : i32, i8