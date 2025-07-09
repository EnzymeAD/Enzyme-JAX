// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main1 outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD1
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main1 outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --verify-each=0 | FileCheck %s --check-prefix=REVERSE1
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main2 outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD2
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main2 outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --verify-each=0 | FileCheck %s --check-prefix=REVERSE2
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main3 outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD3
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main3 outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --verify-each=0 | FileCheck %s --check-prefix=REVERSE3

func.func @main1(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = enzymexla.ml.gelu %arg0, approximation = NONE : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
}

func.func @main2(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = enzymexla.ml.gelu %arg0, approximation = TANH : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
}

func.func @main3(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = enzymexla.ml.gelu %arg0, approximation = SIGMOID : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
}
