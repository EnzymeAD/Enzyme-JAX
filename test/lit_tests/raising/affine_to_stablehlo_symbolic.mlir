// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

module {
  func.func @test_symbolic_load(%memref: memref<?xf64>, %symbol_memref: memref<1xi64>, %res_memref: memref<1xf64>) {
    %cst = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %symbol_i64 = affine.load %symbol_memref[0] : memref<1xi64>
    %symbol = arith.index_cast %symbol_i64 : i64 to index
    %res = affine.for %i = 0 to 10 iter_args(%acc = %cst) -> (f64) {
      %val = "affine.load"(%memref, %symbol) <{map = affine_map<()[s0] -> (s0 + 1)>}> : (memref<?xf64>, index) -> f64
      %add = arith.addf %acc, %val : f64
      affine.yield %add : f64
    }
    affine.store %res, %res_memref[0] : memref<1xf64>
    return
  }
}

// CHECK-LABEL: func.func @test_symbolic_load
