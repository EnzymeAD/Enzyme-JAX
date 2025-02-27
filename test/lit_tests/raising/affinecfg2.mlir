// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module @"reactant_run!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @foo() {
    %c4 = arith.constant 4 : index
    %c12 = arith.constant 12 : index
    %true = arith.constant true
    %c-12_i64 = arith.constant -12 : i64
    %c4_i64 = arith.constant 4 : i64
    %c181_i64 = arith.constant 181 : i64
    %c86_i64 = arith.constant 86 : i64
    affine.parallel (%arg0, %arg1) = (0, 0) to (72, 256) {

      %0 = arith.index_castui %arg1 : index to i64
      %1 = arith.index_castui %arg0 : index to i64
      
      %2 = arith.divui %arg0, %c12 : index       // floor(arg0 / 12)
      %3 = arith.index_castui %2 : index to i64  // floor(arg0 / 12)
      %4 = arith.muli %3, %c-12_i64 : i64        // -12 floor(arg0 / 12)

      %5 = arith.addi %4, %1 : i64               // arg0 - 12 floor(arg0 / 12)    [0, 12)
      
      %6 = arith.shrui %arg1, %c4 : index        // floor(arg1 / 16)
      %7 = arith.index_castui %6 : index to i64  // floor(arg1 / 16)
      
      %8 = arith.subi %5, %7 : i64               // arg0 - 12 floor(arg0 / 12) - floor(arg1 / 16)
      %9 = arith.shli %8, %c4_i64 : i64          // (arg0 - 12 floor(arg0 / 12) - floor(arg1 / 16)) * 16

      %10 = arith.addi %0, %9 : i64              // (arg0 - 12 floor(arg0 / 12) - floor(arg1 / 16)) * 16 + arg1

      %11 = arith.shli %3, %c4_i64 : i64         // floor(floor(arg0 / 12) * 16)
      %12 = arith.addi %7, %11 : i64             // floor(arg1 / 16) + floor(floor(arg0 / 12) * 16)
      %13 = arith.cmpi ugt, %10, %c181_i64 : i64 // (arg0 - 12 floor(arg0 / 12) - floor(arg1 / 16)) * 16 + arg1 u> 181
      %14 = arith.cmpi ugt, %12, %c86_i64 : i64  // floor(arg1 / 16) + floor(floor(arg0 / 12) * 16) u> 86
      %15 = arith.ori %14, %13 : i1              // [ (arg0 - 12 floor(arg0 / 12) - floor(arg1 / 16)) * 16 + arg1 u> 181 ] OR [ floor(arg1 / 16) + floor(floor(arg0 / 12) * 16) u> 86 ]
      %16 = arith.xori %15, %true : i1           // NOT ([ (arg0 - 12 floor(arg0 / 12) - floor(arg1 / 16)) * 16 + arg1 u> 181 ] OR [ floor(arg1 / 16) + floor(floor(arg0 / 12) * 16) u> 86 ])
						 // [ (arg0 - 12 floor(arg0 / 12) - floor(arg1 / 16)) * 16 + arg1 u<= 181 ] AND  [floor(arg1 / 16) + floor(floor(arg0 / 12) * 16) u<= 86 ]
						 // [ (arg0 - 12 floor(arg0 / 12) - floor(arg1 / 16)) * 16 + arg1 >= 0 ]
						 // AND
						 // [ 181 - ((arg0 - 12 floor(arg0 / 12) - floor(arg1 / 16)) * 16 + arg1 ) >= 0 ]
						 // AND
						 // floor(arg1 / 16) + floor(floor(arg0 / 12) / 16) >= 0 [ auto proven, and removed ]
						 // AND
						 // [ 86 - (floor(arg1 / 16) + floor(floor(arg0 / 12) * 16)) ] >= 0
      scf.if %16 {
        "test.op"() : () -> ()
      }
    }
    return
  }
}

// PRE-SIMPLE: #set = affine_set<(d0, d1) : (-(d0 floordiv 16) - (d1 floordiv 12) * 16 + 86 >= 0, d0 + d1 * 16 - (d1 floordiv 12) * 192 - (d0 floordiv 16) * 16 >= 0, -d0 - d1 * 16 + (d1 floordiv 12) * 192 + (d0 floordiv 16) * 16 + 181 >= 0)>
// OPREV-CHECK: #set = affine_set<(d0, d1) : (-(d0 floordiv 16) - (d1 floordiv 12) * 16 + 86 >= 0, d0 mod 16 + (d1 mod 12) * 16 >= 0, -(d0 mod 16) - (d1 mod 12) * 16 + 181 >= 0)> 
// CHECK-NEXT: module @"reactant_run!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
// CHECK-NEXT:   func.func @foo() {
// CHECK-NEXT:     affine.parallel (%arg0, %arg1) = (0, 0) to (87, 182) {
// CHECK-NEXT:       affine.if #set(%arg1, %arg0) {
// CHECK-NEXT:         "test.op"() : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

