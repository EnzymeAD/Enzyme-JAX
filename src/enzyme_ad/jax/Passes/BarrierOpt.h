#ifndef ENZYME_AD_JAX_PASSES_BARRIEROPT_H
#define ENZYME_AD_JAX_PASSES_BARRIEROPT_H

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace enzymexla {

extern llvm::cl::opt<bool> BarrierOpt;

} // namespace enzymexla
} // namespace mlir

#endif