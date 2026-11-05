#ifndef PERFIFY_LEAN_EMITTER_H
#define PERFIFY_LEAN_EMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "llvm/ADT/StringMap.h"
#include <string>

namespace mlir {
namespace enzyme {
namespace perfify {

struct LeanTheorem {
  ConditionsOp cond;
  HypothesisOp hyp;
  std::string name;
  std::string statement;
};

struct LeanFile {
  std::string text;
  SmallVector<LeanTheorem> theorems;
};

/// `proofs` replaces `sorry` by theorem name; empty `imports` means the default.
LeanFile emitLean(ModuleOp module, ArrayRef<std::string> imports = {},
                  const llvm::StringMap<std::string> &proofs = {});

} // namespace perfify
} // namespace enzyme
} // namespace mlir

#endif
