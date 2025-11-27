#include "src/enzyme_ad/jax/Analysis/TensorSymmetricAnalysis.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_TENSORSYMMETRICSIMPLIFYPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

class TensorSymmetricSimplifyPass
    : public enzyme::impl::TensorSymmetricSimplifyPassBase<
          TensorSymmetricSimplifyPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    DataFlowSolver solver;

    solver.load<enzyme::TensorSymmetricAnalysis>();
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      return signalPassFailure();
    }

    auto mod = getOperation();

    mod->walk([&](Operation *op) {
      SmallVector<Attribute> symmetryAttrs;
      bool anyKnown = false;

      for (auto result : op->getResults()) {
        auto *state =
            solver.lookupState<enzyme::TensorSymmetricLattice>(result);

        if (!state) {
          symmetryAttrs.push_back(StringAttr::get(mod.getContext(), "unknown"));
          continue;
        }

        const auto &group = state->getValue();
        std::string str;
        llvm::raw_string_ostream os(str);
        group.print(os);

        symmetryAttrs.push_back(StringAttr::get(mod.getContext(), str));
        anyKnown = true;
      }

      if (anyKnown) {
        op->setAttr("enzymexla.symmetric_dims",
                    ArrayAttr::get(mod.getContext(), symmetryAttrs));
      }
    });
  }
};

} // namespace
