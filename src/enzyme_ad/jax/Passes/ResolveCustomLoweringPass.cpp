#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
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

// https://github.com/openxla/stablehlo/blob/a85bcc1dd33d2dbc05670b914644971bf3e49671/stablehlo/transforms/StablehloRefineArguments.cpp#L52C1-L65C2
stablehlo::CustomCallOp
makeShapeRefinementOperandWrapper(OpBuilder &builder, Value operand,
                                  RankedTensorType refinedType) {
  auto constant = builder.create<stablehlo::ConstantOp>(
      operand.getLoc(), builder.getI64TensorAttr(refinedType.getShape()));
  return builder.create<stablehlo::CustomCallOp>(
      operand.getLoc(), operand.getType(), ValueRange{operand, constant},
      llvm::SmallVector<NamedAttribute>{
          builder.getNamedAttr(
              "call_target_name",
              builder.getStringAttr(
                  "stablehlo.shape_refinement_operand_wrapper")),
          builder.getNamedAttr("indices_of_shape_operands",
                               builder.getI64TensorAttr({1}))});
}

void wrapRefinedOperands(func::FuncOp func, TypeRange refinedTypes) {
  Region &body = func.getBody();
  OpBuilder builder(body);
  builder.setInsertionPointToStart(&body.front());
  for (int64_t i = 0; i < body.getNumArguments(); ++i) {
    BlockArgument arg = body.getArgument(i);
    Type argType = arg.getType();
    Type refinedType = refinedTypes[i];
    if (argType != refinedType) {
      auto rankedRefinedType = cast<RankedTensorType>(refinedType);
      auto customCall =
          makeShapeRefinementOperandWrapper(builder, arg, rankedRefinedType);
      auto callResult = customCall.getResult(0);
      arg.replaceAllUsesExcept(callResult, customCall);
    }
  }
}

void refineOperandsAndUpdateFunctionSignature(func::FuncOp func,
                                              TypeRange refinedInputTypes) {
  Region &body = func.getBody();
  OpBuilder builder(body);
  for (int64_t i = 0; i < body.getNumArguments(); ++i) {
    auto arg = body.getArgument(i);
    arg.setType(refinedInputTypes[i]);
  }
  func.setType(
      builder.getFunctionType(refinedInputTypes, func.getResultTypes()));
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
    SmallVector<Operation *> loweringOpsToRemove, loweredOpsToRemove;

    modOp.walk([&](enzymexla::LoweringRegisterOp op) {
      StringRef opName = op.getOpName();
      auto fn = op.getFnAttr();
      auto config = op.getConfig();

      if (removeRegisterOps)
        loweringOpsToRemove.push_back(op);
      loweringMap[opName].emplace_back(LoweringEntry{fn, config});
    });

    for (auto op : loweringOpsToRemove)
      op->erase();

    // Step 2. Go through all the ops and resolve custom lowering
    modOp.walk([&](Operation *op) {
      auto configAttr =
          op->getAttrOfType<DictionaryAttr>("enzymexla.lowering.config");
      if (!configAttr)
        return;

      auto dialectOpName = op->getName().getStringRef();

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
            return signalPassFailure();
          }

          if (operandType.getRank() != expectedType.getRank()) {
            op->emitError()
                << "Rank mismatch for operand in lowering function.";
            return signalPassFailure();
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
            return signalPassFailure();
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
            return signalPassFailure();
          }

          if (resultType.getRank() != expectedType.getRank()) {
            op->emitError()
                << "Rank mismatch in result type for lowering function.";
            return signalPassFailure();
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
            return signalPassFailure();
          }
        }

        // Generate name for new function
        static int wrapperCounter = 0;
        std::string wrapperName = matchedFn.getValue().str() + "__" +
                                  std::to_string(wrapperCounter++);
        SymbolRefAttr wrapperSym =
            SymbolRefAttr::get(builder.getContext(), wrapperName);

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(modOp.getBody());

          // Clone the entire lowering function and rename it
          Operation *origFnOp = fnOpInterface.getOperation();
          Operation *clonedOp = builder.clone(*origFnOp);
          auto clonedFn = cast<FunctionOpInterface>(clonedOp);
          clonedFn.setName(wrapperName);
          clonedFn.setPrivate();

          auto clonedFuncOp = cast<func::FuncOp>(clonedOp);
          if (!clonedFuncOp) {
            op->emitError() << "Currently we only support lowering func.func";
            return signalPassFailure();
          }

          auto inputTypes = op->getOperandTypes();
          wrapRefinedOperands(clonedFuncOp, inputTypes);
          refineOperandsAndUpdateFunctionSignature(clonedFuncOp, inputTypes);
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(op);
        auto call = builder.create<func::CallOp>(
            op->getLoc(), wrapperSym, op->getResultTypes(), op->getOperands());
        loweredOpsToRemove.push_back(op);
        op->replaceAllUsesWith(call.getResults());
      }
    });

    for (auto op : loweredOpsToRemove)
      op->erase();

    // Step 3. Run shape refinement
    RewritePatternSet patternsShapeRefinement(ctx);
    stablehlo::populateStablehloRefineShapesPatterns(&patternsShapeRefinement,
                                                     ctx);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(
            modOp, std::move(patternsShapeRefinement), config))) {
      modOp.emitError("Failed to apply stablehlo shape refinement patterns.");
      return signalPassFailure();
    }

    RewritePatternSet patternsCanonDynamism(ctx);
    stablehlo::populateStablehloCanonicalizeDynamismPatterns(
        &patternsCanonDynamism, ctx);

    if (failed(applyPatternsAndFoldGreedily(
            modOp, std::move(patternsCanonDynamism), config))) {
      modOp.emitError(
          "Failed to apply stablehlo canonicalize dynamism patterns.");
      return signalPassFailure();
    }
  }
};
