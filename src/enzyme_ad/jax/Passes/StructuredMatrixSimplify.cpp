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
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
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

    auto mod = getOperation();

    // TODO: make IR annotation optional via an option
    mod->walk([&](Operation *op) {
      SmallVector<Attribute> structuredSparsityAttrs;
      bool anyKnown = false;
      for (auto result : op->getResults()) {
        auto *state =
            solver.lookupState<structure_analysis::StructuredMatrixLattice>(
                result);
        if (!state) {
          structuredSparsityAttrs.push_back(
              enzymexla::StructuredSparsityAttr::get(
                  mod.getContext(),
                  enzymexla::StructuredSparsityPatternAttr::get(
                      mod.getContext(),
                      enzymexla::StructuredSparsityKind::Unknown, -1, -1),
                  SmallVector<enzymexla::StructuredValueProperty>()));
          continue;
        }

        anyKnown = true;

        // TODO: get structured sparsity kind
        auto structuredSparsityKind =
            enzymexla::StructuredSparsityPatternAttr::get(
                mod.getContext(), enzymexla::StructuredSparsityKind::Unknown,
                state->getValue().getSparsityPattern().getLowerBandwidth(),
                state->getValue().getSparsityPattern().getUpperBandwidth());

        SmallVector<enzymexla::StructuredValueProperty>
            structuredValueProperties;
        auto valueProperties = state->getValue().getProperties();
        if (valueProperties.hasUnitDiagonal()) {
          structuredValueProperties.push_back(
              enzymexla::StructuredValueProperty::UnitDiagonal);
        }
        if (valueProperties.isSymmetric()) {
          structuredValueProperties.push_back(
              enzymexla::StructuredValueProperty::Symmetric);
        }
        if (valueProperties.isHermitian()) {
          structuredValueProperties.push_back(
              enzymexla::StructuredValueProperty::Hermitian);
        }
        if (valueProperties.isBroadcastedScalar()) {
          structuredValueProperties.push_back(
              enzymexla::StructuredValueProperty::BroadcastedScalar);
        }

        auto structuredSparsity = enzymexla::StructuredSparsityAttr::get(
            mod.getContext(), structuredSparsityKind,
            structuredValueProperties);

        structuredSparsityAttrs.push_back(structuredSparsity);
      }

      if (anyKnown) {
        op->setAttr("structured_sparsity",
                    ArrayAttr::get(mod.getContext(), structuredSparsityAttrs));
      }

      return WalkResult::advance();
    });

    // TODO: do things here
  }
};

} // namespace
