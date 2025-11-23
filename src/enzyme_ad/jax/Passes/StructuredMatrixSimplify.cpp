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

        if (state->getValue().getSparsityPattern().getKind() !=
            mlir::structure_analysis::StructuredSparsityKind::Unknown) {
          anyKnown = true;
        }

        enzymexla::StructuredSparsityKind ssKind;
        switch (state->getValue().getSparsityPattern().getKind()) {
        case mlir::structure_analysis::StructuredSparsityKind::Unknown:
          ssKind = enzymexla::StructuredSparsityKind::Unknown;
          break;
        case mlir::structure_analysis::StructuredSparsityKind::Dense:
          ssKind = enzymexla::StructuredSparsityKind::Dense;
          break;
        case mlir::structure_analysis::StructuredSparsityKind::Band:
          ssKind = enzymexla::StructuredSparsityKind::Band;
          break;
        case mlir::structure_analysis::StructuredSparsityKind::UpperTriangular:
          ssKind = enzymexla::StructuredSparsityKind::UpperTriangular;
          break;
        case mlir::structure_analysis::StructuredSparsityKind::UpperBidiagonal:
          ssKind = enzymexla::StructuredSparsityKind::UpperBidiagonal;
          break;
        case mlir::structure_analysis::StructuredSparsityKind::LowerTriangular:
          ssKind = enzymexla::StructuredSparsityKind::LowerTriangular;
          break;
        case mlir::structure_analysis::StructuredSparsityKind::LowerBidiagonal:
          ssKind = enzymexla::StructuredSparsityKind::LowerBidiagonal;
          break;
        case mlir::structure_analysis::StructuredSparsityKind::Tridiagonal:
          ssKind = enzymexla::StructuredSparsityKind::Tridiagonal;
          break;
        case mlir::structure_analysis::StructuredSparsityKind::Diagonal:
          ssKind = enzymexla::StructuredSparsityKind::Diagonal;
          break;
        case mlir::structure_analysis::StructuredSparsityKind::Empty:
          ssKind = enzymexla::StructuredSparsityKind::Empty;
          break;
        }

        auto structuredSparsityKind =
            enzymexla::StructuredSparsityPatternAttr::get(
                mod.getContext(), ssKind,
                state->getValue().getSparsityPattern().getLowerBandwidth(),
                state->getValue().getSparsityPattern().getUpperBandwidth());

        SmallVector<enzymexla::StructuredValueProperty>
            structuredValueProperties;
        auto valueProperties = state->getValue().getProperties();
        if (valueProperties.hasUnitDiagonal()) {
          anyKnown = true;
          structuredValueProperties.push_back(
              enzymexla::StructuredValueProperty::UnitDiagonal);
        }
        if (valueProperties.isSymmetric()) {
          anyKnown = true;
          structuredValueProperties.push_back(
              enzymexla::StructuredValueProperty::Symmetric);
        }
        if (valueProperties.isHermitian()) {
          anyKnown = true;
          structuredValueProperties.push_back(
              enzymexla::StructuredValueProperty::Hermitian);
        }
        if (valueProperties.isBroadcastedScalar()) {
          anyKnown = true;
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
