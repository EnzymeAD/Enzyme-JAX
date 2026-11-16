// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize | FileCheck %s --check-prefix=AD

// The kernel runs on a view of the buffer: the cached view of its shadow is
// rebuilt on the buffer of the reverse region.

module {
  func.func @main(%arg0: tensor<100xf32>) -> tensor<100xf32> {
    %0 = "enzymexla.jit_region"(%arg0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> ({
    ^bb0(%m: memref<100xf32>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c50 = arith.constant 50 : index
      %s = memref.subview %m[25] [50] [1] : memref<100xf32> to memref<50xf32, strided<[1], offset: 25>>
      %w0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c50, %c1, %c1) ({
        scf.parallel (%i) = (%c0) to (%c50) step (%c1) {
          %v1 = memref.load %s[%i] : memref<50xf32, strided<[1], offset: 25>>
          %v2 = math.exp %v1 : f32
          memref.store %v2, %s[%i] : memref<50xf32, strided<[1], offset: 25>>
          scf.reduce
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : (index, index, index, index, index, index) -> index
      "enzymexla.jit_region_yield"() : () -> ()
    }) : (tensor<100xf32>) -> tensor<100xf32>
    return %0 : tensor<100xf32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK-NOT:     enzyme.init
// CHECK:         "enzymexla.jit_region"(%arg1, %{{.+}})
// CHECK-NEXT:    ^bb0(%[[DM:.+]]: memref<100xf32>, %{{.+}}: memref<50xf32, 1>):
// CHECK-NEXT:      %[[DS:.+]] = memref.subview %[[DM]][25] [50] [1]
// CHECK:           enzyme.atomic_rmw addf %{{.+}}, %[[DS]][%{{.+}}]
// CHECK-NOT:     enzyme.{{push|pop}}
// CHECK-NOT:     memref.alloc

// Right after differentiation, the augmented region has no input for the shadow,
// and no cache of it is left.
// AD-LABEL: func.func @main
// AD:         "enzymexla.jit_region"(%arg0) <{
// AD-NOT:     !enzyme.Cache<memref
// AD-NOT:     memref.alloc
