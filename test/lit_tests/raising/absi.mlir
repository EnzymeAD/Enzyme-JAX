// RUN: enzymexlamlir-opt --libdevice-funcs-raise %s | FileCheck %s --check-prefix=RAISE
// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s --check-prefix=HLO

module {
  // RAISE-LABEL: @intr_absi
  // RAISE: math.absi %arg0 : i32
  // RAISE-NOT: llvm.intr.abs
  func.func @intr_absi(%arg0: i32) -> i32 {
    %res = "llvm.intr.abs"(%arg0) <{is_int_min_poison = false}> : (i32) -> i32
    func.return %res : i32
  }

  // HLO-LABEL: @tensor_absi
  // HLO: stablehlo.abs %arg0 : tensor<20xi32>
  // HLO-NOT: math.absi
  func.func @tensor_absi(%arg0: tensor<20xi32>) -> tensor<20xi32> {
    %res = math.absi %arg0 : tensor<20xi32>
    func.return %res : tensor<20xi32>
  }
}
