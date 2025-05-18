#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "resolve-custom-lowering"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_RESOLVECUSTOMLOWERINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

static bool isSubset(mlir::DictionaryAttr subset,
                     mlir::DictionaryAttr superset) {
  llvm::DenseMap<llvm::StringRef, mlir::Attribute> superMap;
  for (const auto &attr : superset) {
    superMap[attr.getName()] = attr.getValue();
  }

  for (const auto &attr : subset) {
    auto it = superMap.find(attr.getName());
    if (it == superMap.end() || it->second != attr.getValue())
      return false;
  }
  return true;
}

struct ResolveCustomLoweringPass
    : public enzyme::impl::ResolveCustomLoweringPassBase<
          ResolveCustomLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp modOp = getOperation();
    auto *ctx = modOp->getContext();
    OpBuilder builder(ctx);

    // Step 1. Lookup and register all custom lowering ops
    struct LoweringEntry {
      FlatSymbolRefAttr fn;
      DictionaryAttr config;
    };

    DenseMap<StringRef, SmallVector<LoweringEntry>> loweringMap;

    modOp.walk([&](enzymexla::LoweringRegisterOp op) {
      StringRef opName = op.getOpName();
      auto fn = op.getFnAttr();
      auto config = op.getConfig();

      loweringMap[opName].emplace_back(LoweringEntry{fn, config});
    });

    // ----
    // TODO: Remove
    llvm::errs() << "=== Lowering Map Starts ===\n\n";

    for (const auto &entry : loweringMap) {
      StringRef opName = entry.first;
      llvm::errs() << "Op: " << opName << "\n";

      for (const auto &entryVal : entry.second) {
        llvm::errs() << "  Function: " << entryVal.fn.getValue() << "\n";
        llvm::errs() << "  Config:\n";
        for (const auto &kv : entryVal.config.getValue()) {
          llvm::errs() << "    " << kv.getName() << ": ";
          kv.getValue().print(llvm::errs());
          llvm::errs() << "\n";
        }
      }
      llvm::errs() << "\n";
    }
    llvm::errs() << "=== Lowering Map Ends ===\n\n";
    // ----

    // Step 2. Go through all the ops and resolve custom lowering
    SmallVector<Operation *> opsToRemove;

    modOp.walk([&](Operation *op) {
      auto configAttr =
          op->getAttrOfType<DictionaryAttr>("enzymexla.lowering.config");
      if (!configAttr)
        return;

      auto dialectOpName = op->getName().getStringRef();
      llvm::errs() << "Checking op: " << dialectOpName << "\n";

      auto it = loweringMap.find(dialectOpName);
      if (it == loweringMap.end()) {
        op->emitError("No lowering registered for op.");
        signalPassFailure();
        return;
      }

      SmallVector<const LoweringEntry *> matching;
      for (const auto &entry : it->second) {
        if (isSubset(configAttr, entry.config)) {
          matching.push_back(&entry);
        }
      }

      if (matching.empty()) {
        op->emitError("No matching lowering found.");
        signalPassFailure();
      } else if (matching.size() > 1) {
        op->emitError("Ambiguous lowering match: multiple registered lowerings "
                      "match provided config.");
        for (const auto *entry : matching) {
          llvm::errs() << "  - Candidate fn: " << entry->fn.getValue()
                       << ", config: ";
          entry->config.print(llvm::errs());
          llvm::errs() << "\n";
        }
        signalPassFailure();
      } else {
        llvm::errs() << "  ✔ Matched lowering: "
                     << matching.front()->fn.getValue() << "\n";

        auto matchedFn = matching.front()->fn;
        auto configDict = matching.front()->config;

        auto fnSymbol = modOp.lookupSymbol(matchedFn.getAttr());
        auto fnOpInterface = dyn_cast<FunctionOpInterface>(fnSymbol);
        if (!fnOpInterface) {
          op->emitError() << "Matched symbol " << matchedFn.getValue()
                          << " does not implement FunctionOpInterface.";
          signalPassFailure();
          return;
        }

        auto fnType = dyn_cast<FunctionType>(fnOpInterface.getFunctionType());

        // Check number of operands
        if (fnType.getNumInputs() != op->getNumOperands()) {
          op->emitError() << "Operand count mismatch with lowering function "
                          << matchedFn.getValue();
          signalPassFailure();
          return;
        }

        // Check each operand type (ignoring dynamic sizes)
        for (auto [operand, inp] :
             llvm::zip(op->getOperands(), fnType.getInputs())) {
          auto operandType = dyn_cast<RankedTensorType>(operand.getType());
          auto expectedType = dyn_cast<RankedTensorType>(inp);

          if (!operandType || !expectedType) {
            op->emitError() << "Expected ranked tensor types for comparison.";
            signalPassFailure();
            return;
          }

          if (operandType.getRank() != expectedType.getRank()) {
            op->emitError()
                << "Rank mismatch for operand in lowering function.";
            signalPassFailure();
            return;
          }

          for (int i = 0; i < operandType.getRank(); ++i) {
            if (!operandType.isDynamicDim(i) && !expectedType.isDynamicDim(i) &&
                operandType.getDimSize(i) != expectedType.getDimSize(i)) {
              op->emitError()
                  << "Shape mismatch at dimension " << i << " in operand type.";
              signalPassFailure();
              return;
            }
          }

          if (operandType.getElementType() != expectedType.getElementType()) {
            op->emitError() << "Element type mismatch in operand.";
            signalPassFailure();
            return;
          }
        }

        // Check number of results
        if (fnType.getNumResults() != op->getNumResults()) {
          op->emitError() << "Result count mismatch with lowering function "
                          << matchedFn.getValue();
          signalPassFailure();
          return;
        }

        // Check each result type (ignoring dynamic sizes)
        for (auto [result, out] :
             llvm::zip(op->getResults(), fnType.getResults())) {
          auto resultType = dyn_cast<RankedTensorType>(result.getType());
          auto expectedType = dyn_cast<RankedTensorType>(out);

          if (!resultType || !expectedType) {
            op->emitError()
                << "Expected ranked tensor types for result comparison.";
            signalPassFailure();
            return;
          }

          if (resultType.getRank() != expectedType.getRank()) {
            op->emitError()
                << "Rank mismatch in result type for lowering function.";
            signalPassFailure();
            return;
          }

          for (int i = 0; i < resultType.getRank(); ++i) {
            if (!resultType.isDynamicDim(i) && !expectedType.isDynamicDim(i) &&
                resultType.getDimSize(i) != expectedType.getDimSize(i)) {
              op->emitError()
                  << "Shape mismatch at result dimension " << i << ".";
              signalPassFailure();
              return;
            }
          }

          if (resultType.getElementType() != expectedType.getElementType()) {
            op->emitError() << "Element type mismatch in result.";
            signalPassFailure();
            return;
          }
        }

        // Generate name for new function
        static int wrapperCounter = 0;
        std::string wrapperName =
            matchedFn.getValue().str() + std::to_string(wrapperCounter++);
        SymbolRefAttr wrapperSym =
            SymbolRefAttr::get(builder.getContext(), wrapperName);

        llvm::errs() << "Wrapper name: " << wrapperName << "\n";

        // Build the new function type from the op's operand and result types
        auto inputTypes = llvm::to_vector(op->getOperandTypes());
        auto resultTypes = llvm::to_vector(op->getResultTypes());
        auto newFnType = builder.getFunctionType(inputTypes, resultTypes);

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(modOp.getBody());

          // Clone the entire lowering function and rename it
          Operation *origFnOp = fnOpInterface.getOperation();
          Operation *clonedOp = builder.clone(*origFnOp);
          auto clonedFn = cast<FunctionOpInterface>(clonedOp);
          clonedFn.setName(wrapperName);
          clonedFn.setPrivate();
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(op);
        auto call = builder.create<func::CallOp>(
            op->getLoc(), wrapperSym, op->getResultTypes(), op->getOperands());
        opsToRemove.push_back(op);
        op->replaceAllUsesWith(call.getResults());
      }
    });

    for (auto op : opsToRemove)
      op->erase();

    llvm::errs() << "=== Lowered All Ops ===\n\n";

    // Step 3. Run shape refinement
  }
};
