// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(raise-affine-to-stablehlo)" | FileCheck %s

module {
  func.func @test_gpu_wrapper_alloca(%arg0: memref<?xi32>, %val: i32) {
    // CHECK: func.func @test_gpu_wrapper_alloca
    // CHECK: %[[ALLOCA:.+]] = memref.alloca() : memref<i32>
    // CHECK: cf.br ^bb1
    // CHECK: ^bb1
    // CHECK: cf.cond_br
    // CHECK: ^bb2
    // CHECK-NOT: memref.alloca
    
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    
    %c1_i32 = arith.constant 1 : i32
    
    cf.br ^bb1(%c0 : index)
    
  ^bb1(%i: index):
    %cmp = arith.cmpi slt, %i, %c10 : index
    cf.cond_br %cmp, ^bb2, ^bb3
    
  ^bb2:
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      // simulate use of %val
      %1 = arith.addi %val, %c1_i32 : i32
      memref.store %1, %arg0[%i] : memref<?xi32>
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    
    %next = arith.addi %i, %c1 : index
    cf.br ^bb1(%next : index)
    
  ^bb3:
    return
  }
}
