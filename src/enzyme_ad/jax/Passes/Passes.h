//===- Passes.h - Enzyme pass include header  -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ENZYMEXLA_PASSES_H
#define ENZYMEXLA_PASSES_H

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class PatternRewriter;
class RewritePatternSet;
class DominanceInfo;
namespace enzyme {
std::unique_ptr<Pass> createArithRaisingPass();
std::unique_ptr<Pass> createConsumingInterpreterPass();
std::unique_ptr<Pass> createEnzymeHLOOptPass();
std::unique_ptr<Pass> createEnzymeHLOUnrollPass();
std::unique_ptr<Pass> createPrintPass();
std::unique_ptr<Pass> createLowerKernelPass();
std::unique_ptr<Pass> createLibDeviceFuncsRaisingPass();
std::unique_ptr<Pass> createSROAWrappersPass();

void populateLibDeviceFuncsToOpsPatterns(MLIRContext *context,
                                         RewritePatternSet &patterns);
} // namespace enzyme

// namespace enzymexla {
// class EnzymeXLADialect;
//}
} // namespace mlir

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

class DLTIDialect;

namespace mhlo {
class MhloDialect;
} // end namespace mhlo

namespace chlo {
class ChloDialect;
} // end namespace chlo

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace math {
class MathDialect;
} // namespace math

namespace vector {
class VectorDialect;
} // namespace vector

namespace nvgpu {
class NVGPUDialect;
} // namespace nvgpu

namespace NVVM {
class NVVMDialect;
} // namespace NVVM

namespace stablehlo {
class StablehloDialect;
} // namespace stablehlo

namespace arith {
class ArithDialect;
} // end namespace arith

namespace cf {
class ControlFlowDialect;
} // end namespace cf

namespace gpu {
class GPUDialect;
} // end namespace gpu

namespace scf {
class SCFDialect;
} // end namespace scf

namespace memref {
class MemRefDialect;
} // end namespace memref

namespace async {
class AsyncDialect;
} // namespace async

namespace func {
class FuncDialect;
}

class AffineDialect;
namespace LLVM {
class LLVMDialect;
}

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

} // end namespace mlir

#endif // ENZYMEXLA_PASSES_H
