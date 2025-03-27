// RUN: enzymexlamlir-opt %s --libdevice-funcs-raise  --raise-affine-to-stablehlo --arith-raise | FileCheck %s

module {
  func.func @tr(%a : f64) -> f32 {
    %trunc = llvm.fptrunc %a : f64 to f32
    return %trunc : f32
  }
  func.func @trmem(%in : memref<1xf64>, %out : memref<1xf32>) {
    affine.parallel (%i) = (0) to (1) {
      %a = affine.load %in[0] : memref<1xf64>
      %trunc = llvm.fptrunc %a : f64 to f32
      affine.store %trunc, %out[0] : memref<1xf32>
    }
    return
  }
  func.func @ctr(%a : f64) -> f32 {
     %trunc = llvm.intr.experimental.constrained.fptrunc %a towardzero ignore : f64 to f32
     return %trunc : f32
   }
}
