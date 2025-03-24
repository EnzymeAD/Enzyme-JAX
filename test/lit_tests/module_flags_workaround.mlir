// RUN: enzymexlamlir-opt %s --sroa-wrappers

// Working around bug in upstream llvm which was fixed in 800593a0
module {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Debug Info Version", 3>]
}
