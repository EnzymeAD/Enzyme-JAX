#ifndef PERFIFY_LEAN_EMITTER_H
#define PERFIFY_LEAN_EMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
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

LeanFile emitLean(ModuleOp module);

} // namespace perfify
} // namespace enzyme
} // namespace mlir

#endif
