// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --mlir-print-ir-after-failure --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

module {
  func.func @main(%arg0: tensor<100xf32>) -> tensor<100xf32> {

    %0 = "enzymexla.jit_region"(%arg0) <{
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    }> ({
    ^bb0(%arg1: memref<100xf32>):

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c100 = arith.constant 100 : index
      %cst0 = arith.constant 2.0 : f32

      %w = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c100, %c1, %c1) ({
        scf.parallel (%i) = (%c0) to (%c100) step (%c1) {
          %v1 = memref.load %arg1[%i] : memref<100xf32>
          %v2 = arith.mulf %v1, %cst0 : f32
          %v3 = math.exp %v2 : f32
          memref.store %v3, %arg1[%i] : memref<100xf32>
          scf.reduce
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : (index, index, index, index, index, index) -> index

      "enzymexla.jit_region_yield"() : () -> ()
    }) : (tensor<100xf32>) -> tensor<100xf32>

    return %0 : tensor<100xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
// CHECK-NEXT:      %cst = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:      %c100 = arith.constant 100 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %cst_1 = arith.constant dense<0.000000e+00> : tensor<100xf32>
// CHECK-NEXT:      %0 = "enzymexla.jit_region"(%arg0, %cst_1, %cst_1) <{operand_attrs = [{}, {enzymexla.reverse_operand_index = 0 : i64}, {}], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]}> ({
// CHECK-NEXT:      ^bb0(%arg2: memref<100xf32>, %arg3: memref<100xf32>, %arg4: memref<100xf32, 1>):
// CHECK-NEXT:        %2 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c100, %c1, %c1) ({
// CHECK-NEXT:          scf.parallel (%arg5) = (%c0) to (%c100) step (%c1) {
// CHECK-NEXT:            %3 = memref.load %arg2[%arg5] : memref<100xf32>
// CHECK-NEXT:            %4 = arith.mulf %3, %cst : f32
// CHECK-NEXT:            memref.store %4, %arg4[%arg5] : memref<100xf32, 1>
// CHECK-NEXT:            %5 = math.exp %4 : f32
// CHECK-NEXT:            memref.store %5, %arg2[%arg5] : memref<100xf32>
// CHECK-NEXT:            scf.reduce
// CHECK-NEXT:          }
// CHECK-NEXT:          "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:        }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:        enzymexla.jit_region_yield
// CHECK-NEXT:      }) : (tensor<100xf32>, tensor<100xf32>, tensor<100xf32>) -> tensor<100xf32>
// CHECK-NEXT:      %1 = "enzymexla.jit_region"(%arg1, %0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> ({
// CHECK-NEXT:      ^bb0(%arg2: memref<100xf32>, %arg3: memref<100xf32, 1>):
// CHECK-NEXT:        %2 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c100, %c1, %c1) ({
// CHECK-NEXT:          scf.parallel (%arg4) = (%c0) to (%c100) step (%c1) {
// CHECK-NEXT:            %3 = memref.load %arg3[%arg4] : memref<100xf32, 1>
// CHECK-NEXT:            %4 = memref.load %arg2[%arg4] : memref<100xf32>
// CHECK-NEXT:            memref.store %cst_0, %arg2[%arg4] : memref<100xf32>
// CHECK-NEXT:            %5 = math.exp %3 fastmath<fast> : f32
// CHECK-NEXT:            %6 = arith.mulf %4, %5 fastmath<fast> : f32
// CHECK-NEXT:            %7 = arith.mulf %6, %cst fastmath<fast> : f32
// CHECK-NEXT:            %8 = enzyme.atomic_rmw addf %7, %arg2[%arg4] monotonic fastmath<fast> : (f32, memref<100xf32>) -> f32
// CHECK-NEXT:            scf.reduce
// CHECK-NEXT:          }
// CHECK-NEXT:          "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:        }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:        enzymexla.jit_region_yield
// CHECK-NEXT:      }) : (tensor<100xf32>, tensor<100xf32>) -> tensor<100xf32>
// CHECK-NEXT:      return %1 : tensor<100xf32>
// CHECK-NEXT:    }
