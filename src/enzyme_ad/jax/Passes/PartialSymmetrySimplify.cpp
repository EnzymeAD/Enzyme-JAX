#include "src/enzyme_ad/jax/Analysis/PartialSymmetryAnalysis.h"
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
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "partial-symmetry-simplify"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_PARTIALSYMMETRYSIMPLIFYPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::enzyme;

namespace {

class PartialSymmetrySimplifyPass
    : public enzyme::impl::PartialSymmetrySimplifyPassBase<
          PartialSymmetrySimplifyPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    DataFlowSolver solver;

    solver.load<enzyme::PartialSymmetryAnalysis>();
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      return signalPassFailure();
    }

    auto mod = getOperation();

    mod->walk([&](Operation *op) {
      SmallVector<Attribute> partialSymmetryAttrs;
      bool anyKnown = false;

      for (auto result : op->getResults()) {
        auto *state =
            solver.lookupState<enzyme::PartialSymmetryLattice>(result);
        if (!state) {
          continue;
        }

        auto dimensionSets = state->getValue().getDimensionSets();

        SmallVector<enzymexla::SymmetricDimensionSetAttr> dimensionSetAttrs;
        for (const auto &set : dimensionSets) {
          if (set.size() > 1) {
            anyKnown = true;
            auto denseAttr = DenseI64ArrayAttr::get(mod.getContext(), set);
            auto dimensionSetAttr = enzymexla::SymmetricDimensionSetAttr::get(
                mod.getContext(), denseAttr);
            dimensionSetAttrs.push_back(dimensionSetAttr);
          }
        }

        if (dimensionSetAttrs.empty()) {
          continue;
        }

        auto partialSymmetry =
            enzymexla::PartialSymmetryAnalysisResultAttr::get(
                mod.getContext(), dimensionSetAttrs);
        partialSymmetryAttrs.push_back(partialSymmetry);
      }

      if (anyKnown) {
        op->setAttr("enzymexla.partial_symmetry",
                    ArrayAttr::get(mod.getContext(), partialSymmetryAttrs));
      }

      return WalkResult::advance();
    });

    // TODO: do things here
  }
};

} // namespace
