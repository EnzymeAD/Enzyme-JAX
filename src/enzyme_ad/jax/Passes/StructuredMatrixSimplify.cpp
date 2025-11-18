#include "src/enzyme_ad/jax/Analysis/StructuredMatrixAnalysis.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "structured-matrix-simplify"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_STRUCTUREDMATRIXSIMPLIFYPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::enzyme;
using namespace mlir::structure_analysis;

namespace {

class StructuredMatrixSimplifyPass
    : public enzyme::impl::StructuredMatrixSimplifyPassBase<
          StructuredMatrixSimplifyPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    DataFlowSolver solver;

    solver.load<structure_analysis::StructuredMatrixAnalysis>();
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      return signalPassFailure();
    }

    // TODO: do things here
  }
};

} // namespace
