// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  %y = chlo.conj %x : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %y : tensor<2xcomplex<f32>>
}