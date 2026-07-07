//===- LowerEnzymeJacobian.cpp  ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements patterns to convert JVP/VJPs originating from an enzyme.jacobian
// to enzyme fwddiff/autodiff calls
//===----------------------------------------------------------------------===//

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include <cstdint>
#include <optional>
#include <string>

#define DEBUG_TYPE "lower-enzyme-jacobian"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEJACOBIANSTABLEHLO
#define GEN_PASS_DEF_LOWERGRIDKITIDAJACOBIANACTIONSTABLEHLO
#define GEN_PASS_DEF_LOWERSUNDIALSIDAJACOBIANACTIONSTABLEHLO
#define GEN_PASS_DEF_SYNTHESIZESUNDIALSIDAJACOBIANACTIONS
#define GEN_PASS_DEF_SELECTSUNDIALSIDAMATRIXFREE
#define GEN_PASS_DEF_EMITSUNDIALSIDARUNTIMEGLUELLVM
#define GEN_PASS_DEF_RECOVERSUNDIALSIDALLVM
#define GEN_PASS_DEF_MARKGRIDKITSPARSEJACOBIANLLVM
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {
constexpr llvm::StringLiteral kGridKitSolverAttr = "gridkit.solver";
constexpr llvm::StringLiteral kGridKitIdaJacTimes = "ida_jac_times";
constexpr llvm::StringLiteral kGridKitJacobianMaterializationAttr =
    "gridkit.jacobian.materialization";
constexpr llvm::StringLiteral kGridKitJacobianSourceAttr =
    "gridkit.jacobian.source";
constexpr llvm::StringLiteral kGridKitJacobianActionAttr =
    "gridkit.jacobian.action";
constexpr llvm::StringLiteral kGridKitJacobianRoleAttr =
    "gridkit.jacobian.role";
constexpr llvm::StringLiteral kGridKitJacobianFwddiffCallsAttr =
    "gridkit.jacobian.fwddiff_calls";
constexpr llvm::StringLiteral kGridKitJacobianTodenseCallsAttr =
    "gridkit.jacobian.todense_calls";
constexpr llvm::StringLiteral kGridKitJacobianSparseStoreAddressesAttr =
    "gridkit.jacobian.sparse_store_addresses";
constexpr llvm::StringLiteral kGridKitJacobianMarkedHelpersAttr =
    "gridkit.jacobian.marked_sparse_helpers";
constexpr llvm::StringLiteral kGridKitSparseOneHotMaterialization =
    "enzyme_sparse_one_hot";
constexpr llvm::StringLiteral kGridKitResidualJvpCandidate =
    "residual_jvp_candidate";
constexpr llvm::StringLiteral kGridKitSparseTodenseRole = "sparse_todense";
constexpr llvm::StringLiteral kJacobianMaterializationOpName =
    "enzymexla.jacobian_materialization";
constexpr llvm::StringLiteral kJacobianActionOpName =
    "enzymexla.jacobian_action";
constexpr llvm::StringLiteral kSynthesizedJacobianActionsAttr =
    "enzymexla.jacobian_actions_synthesized";
constexpr llvm::StringLiteral kLinkedSundialsIdaJacobianActionsAttr =
    "enzymexla.sundials.ida_solves_linked_jacobian_actions";
constexpr llvm::StringLiteral kSelectedSundialsIdaMatrixFreeAttr =
    "enzymexla.sundials.ida_matrix_free_selected";
constexpr llvm::StringLiteral kSundialsAllowMatrixFreeAttr =
    "enzymexla.sundials.allow_matrix_free";
constexpr llvm::StringLiteral kSundialsMatrixFreeSelectedAttr =
    "enzymexla.sundials.matrix_free_selected";
constexpr llvm::StringLiteral kSundialsUserDataRegisteredAttr =
    "enzymexla.sundials.user_data_registered";
constexpr llvm::StringLiteral kSundialsIdaEffectiveJacobianActionAttr =
    "enzymexla.sundials.ida_effective_jacobian_action";
constexpr llvm::StringLiteral
    kSundialsIdaEffectiveJacobianActionsSynthesizedAttr =
        "enzymexla.sundials.ida_effective_jacobian_actions_synthesized";
constexpr llvm::StringLiteral kRecoveredSundialsIdaSolvesAttr =
    "enzymexla.sundials.ida_solves_recovered";
constexpr llvm::StringLiteral kSundialsIdaRuntimeGlueEmittedAttr =
    "enzymexla.sundials.ida_runtime_glue_emitted";
constexpr llvm::StringLiteral kSundialsRuntimeJacTimesCallbackAttr =
    "enzymexla.sundials.runtime_jactimes_callback";
constexpr llvm::StringLiteral kSundialsRuntimeRegistrationAttr =
    "enzymexla.sundials.runtime_registration";
constexpr llvm::StringLiteral kSundialsRuntimeJvpKernelAttr =
    "enzymexla.sundials.runtime_jvp_kernel";
constexpr llvm::StringLiteral kSundialsRuntimeRawJvpKernelAttr =
    "enzymexla.sundials.runtime_raw_jvp_kernel";
constexpr llvm::StringLiteral kSundialsLoweredRawJvpKernelAttr =
    "enzymexla.sundials.lowered_raw_jvp_kernel";
constexpr llvm::StringLiteral kSundialsRuntimeRoleAttr =
    "enzymexla.sundials.runtime_role";
constexpr llvm::StringLiteral kSundialsIdaJvpKernelAdaptersEmittedAttr =
    "enzymexla.sundials.ida_jvp_kernel_adapters_emitted";
constexpr llvm::StringLiteral kSundialsIdaRawJvpKernelsEmittedAttr =
    "enzymexla.sundials.ida_raw_jvp_kernels_emitted";
constexpr llvm::StringLiteral kSundialsIdaLoweredRawJvpKernelsLinkedAttr =
    "enzymexla.sundials.ida_lowered_raw_jvp_kernels_linked";
constexpr llvm::StringLiteral kSundialsIdaSolveOpName =
    "enzymexla.sundials.ida_solve";
constexpr llvm::StringLiteral kSundialsRoleAttr = "enzymexla.sundials.role";

struct JacobianVectorAction {
  enzyme::JacobianOp jacOp;
  unsigned diffOutputIndex;
  unsigned diffInputIndex;
  Value vector;
  bool isJVP;
  bool isVJP;
};

StringRef classifyGridKitSparseJacobianHelper(StringRef name) {
  if (!name.contains("GridKit6Enzyme6Sparse"))
    return {};
  if (name.contains("4DfDy"))
    return "DfDy";
  if (name.contains("5DfDyp"))
    return "DfDyp";
  if (name.contains("5DfDwb"))
    return "DfDwb";
  if (name.contains("4DhDy"))
    return "DhDy";
  return {};
}

bool isEnzymeFwddiffCallee(StringRef name) {
  return name.contains("__enzyme_fwddiff");
}

bool isEnzymeTodenseCallee(StringRef name) {
  return name.contains("__enzyme_todense");
}

bool isGridKitSparseStoreAddress(StringRef name) {
  return name.contains("GridKit6Enzyme6Sparse") &&
         name.contains("sparse_store");
}

struct GridKitSparseJacobianMatch {
  SmallVector<LLVM::CallOp> fwddiffCalls;
  SmallVector<LLVM::CallOp> todenseCalls;
  SmallVector<LLVM::AddressOfOp> sparseStoreAddresses;
  std::optional<std::string> residual;
  std::optional<SmallVector<std::string, 8>> enzymeActivity;
  std::optional<SmallVector<std::string, 8>> inputActivity;
  std::optional<SmallVector<std::string, 2>> outputActivity;
  std::optional<unsigned> inputCount;
  std::optional<unsigned> outputCount;
  std::optional<unsigned> activeInputIndex;
  std::optional<unsigned> activeOutputIndex;
  std::optional<unsigned> outputDimensionArg;
  std::optional<unsigned> activeInputDimensionArg;
  std::optional<unsigned> seedLoopDimensionArg;
  std::optional<unsigned> outputIndexMapArg;
  std::optional<unsigned> activeInputIndexMapArg;
  std::optional<unsigned> sparseRowsArg;
  std::optional<unsigned> sparseColsArg;
  std::optional<unsigned> sparseValuesArg;
  std::optional<unsigned> sparseNnzArg;
  std::optional<std::string> sparseAssembly;

  bool isComplete() const {
    return !fwddiffCalls.empty() && !todenseCalls.empty() &&
           !sparseStoreAddresses.empty();
  }
};

struct RecoveredJacobianMaterializationRecord {
  std::string materializer;
  std::optional<std::string> residual;
  std::optional<SmallVector<std::string, 8>> enzymeActivity;
  std::optional<SmallVector<std::string, 8>> inputActivity;
  std::optional<SmallVector<std::string, 2>> outputActivity;
  std::optional<unsigned> inputCount;
  std::optional<unsigned> outputCount;
  std::optional<unsigned> activeInputIndex;
  std::optional<unsigned> activeOutputIndex;
  std::optional<unsigned> outputDimensionArg;
  std::optional<unsigned> activeInputDimensionArg;
  std::optional<unsigned> seedLoopDimensionArg;
  std::optional<unsigned> outputIndexMapArg;
  std::optional<unsigned> activeInputIndexMapArg;
  std::optional<unsigned> sparseRowsArg;
  std::optional<unsigned> sparseColsArg;
  std::optional<unsigned> sparseValuesArg;
  std::optional<unsigned> sparseNnzArg;
  std::optional<std::string> sparseAssembly;
  std::string source;
  unsigned fwddiffCalls;
  unsigned todenseCalls;
  unsigned sparseStoreCallbacks;
};

std::optional<std::string> getFwddiffResidualSymbol(LLVM::CallOp call) {
  if (call->getNumOperands() == 0)
    return std::nullopt;

  auto addressOf = call->getOperand(0).getDefiningOp<LLVM::AddressOfOp>();
  if (!addressOf)
    return std::nullopt;

  return addressOf.getGlobalName().str();
}

std::optional<std::string> getLoadedGlobalSymbol(Value value) {
  auto load = value.getDefiningOp<LLVM::LoadOp>();
  if (!load)
    return std::nullopt;

  auto addressOf = load.getAddr().getDefiningOp<LLVM::AddressOfOp>();
  if (!addressOf)
    return std::nullopt;

  return addressOf.getGlobalName().str();
}

std::optional<std::string> getEnzymeActivitySymbol(Value value) {
  std::optional<std::string> global = getLoadedGlobalSymbol(value);
  if (!global)
    return std::nullopt;

  StringRef name(*global);
  if (!name.starts_with("enzyme_") || name.starts_with("__enzyme_"))
    return std::nullopt;

  return *global;
}

SmallVector<std::string, 8> getFwddiffActivitySymbols(LLVM::CallOp call) {
  SmallVector<std::string, 8> activities;
  auto args = call.getArgOperands();
  for (unsigned idx = 1, end = args.size(); idx < end; ++idx) {
    std::optional<std::string> activity = getEnzymeActivitySymbol(args[idx]);
    if (activity)
      activities.push_back(*activity);
  }
  return activities;
}

bool sameActivitySequence(ArrayRef<std::string> lhs,
                          ArrayRef<std::string> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip_equal(lhs, rhs)) {
    if (left != right)
      return false;
  }
  return true;
}

bool isOutputActivityMarker(StringRef activity) {
  return activity.contains("noneed");
}

bool isDifferentiatedInputActivityMarker(StringRef activity) {
  return activity.contains("active") || activity.contains("dup");
}

std::optional<unsigned>
getSingleActiveInputIndex(ArrayRef<std::string> inputActivity) {
  std::optional<unsigned> activeIndex;
  for (auto [idx, activity] : llvm::enumerate(inputActivity)) {
    if (!isDifferentiatedInputActivityMarker(activity))
      continue;
    if (activeIndex)
      return std::nullopt;
    activeIndex = static_cast<unsigned>(idx);
  }
  return activeIndex;
}

bool isPointerLike(Type type) { return isa<LLVM::LLVMPointerType>(type); }

bool hasSparseHelperDimensionSignature(LLVM::LLVMFuncOp func) {
  auto argTypes = func.getArgumentTypes();
  return argTypes.size() > 2 && argTypes[1].isIntOrIndex() &&
         argTypes[2].isIntOrIndex();
}

bool usesSparseStoreAddress(LLVM::CallOp call) {
  for (Value operand : call.getArgOperands()) {
    auto addressOf = operand.getDefiningOp<LLVM::AddressOfOp>();
    if (addressOf && isGridKitSparseStoreAddress(addressOf.getGlobalName()))
      return true;
  }
  return false;
}

std::optional<unsigned> getFunctionPointerArgumentIndex(LLVM::LLVMFuncOp func,
                                                        Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner()->getParentOp() != func.getOperation())
    return std::nullopt;
  if (!isPointerLike(blockArg.getType()))
    return std::nullopt;
  return blockArg.getArgNumber();
}

SmallVector<unsigned, 8>
getSparseTodenseFunctionPointerArgs(LLVM::LLVMFuncOp func,
                                    ArrayRef<LLVM::CallOp> todenseCalls) {
  for (LLVM::CallOp call : todenseCalls) {
    if (!usesSparseStoreAddress(call))
      continue;

    SmallVector<unsigned, 8> functionArgs;
    for (Value operand : call.getArgOperands()) {
      std::optional<unsigned> argIndex =
          getFunctionPointerArgumentIndex(func, operand);
      if (argIndex)
        functionArgs.push_back(*argIndex);
    }
    return functionArgs;
  }
  return {};
}

void deriveSparseLayoutMetadata(LLVM::LLVMFuncOp func,
                                GridKitSparseJacobianMatch &match) {
  if (!hasSparseHelperDimensionSignature(func))
    return;

  match.outputDimensionArg = 1;
  match.activeInputDimensionArg = 2;
  match.seedLoopDimensionArg = 2;
  match.sparseAssembly = "coo_column_seeded_callback";

  SmallVector<unsigned, 8> sparsePointerArgs =
      getSparseTodenseFunctionPointerArgs(func, match.todenseCalls);
  if (sparsePointerArgs.size() >= 2) {
    match.outputIndexMapArg = sparsePointerArgs[0];
    match.activeInputIndexMapArg = sparsePointerArgs[1];
  }
  if (sparsePointerArgs.size() >= 6) {
    ArrayRef<unsigned> outputArgs =
        ArrayRef<unsigned>(sparsePointerArgs).take_back(4);
    match.sparseRowsArg = outputArgs[0];
    match.sparseColsArg = outputArgs[1];
    match.sparseValuesArg = outputArgs[2];
    match.sparseNnzArg = outputArgs[3];
  }
}

ArrayAttr getStringArrayAttr(Builder &builder, ArrayRef<std::string> values) {
  SmallVector<Attribute> attrs;
  for (const std::string &value : values)
    attrs.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attrs);
}

void deriveInputOutputActivity(LLVM::LLVMFuncOp func,
                               GridKitSparseJacobianMatch &match) {
  if (!match.residual || !match.enzymeActivity || match.enzymeActivity->empty())
    return;

  auto module = func->getParentOfType<ModuleOp>();
  if (!module)
    return;

  auto residualFunc = dyn_cast_or_null<FunctionOpInterface>(
      module.lookupSymbol(*match.residual));
  if (!residualFunc)
    return;

  if (residualFunc.getNumResults() != 0 ||
      residualFunc.getNumArguments() != match.enzymeActivity->size())
    return;

  ArrayRef<std::string> activities = *match.enzymeActivity;
  if (!isOutputActivityMarker(activities.back()))
    return;

  SmallVector<std::string, 8> inputActivity;
  ArrayRef<std::string> inputs = activities.drop_back();
  inputActivity.append(inputs.begin(), inputs.end());
  SmallVector<std::string, 2> outputActivity;
  outputActivity.push_back(activities.back());
  std::optional<unsigned> activeInputIndex =
      getSingleActiveInputIndex(inputActivity);
  match.inputActivity = std::move(inputActivity);
  match.outputActivity = std::move(outputActivity);
  match.inputCount = match.inputActivity->size();
  match.outputCount = match.outputActivity->size();
  if (activeInputIndex) {
    match.activeInputIndex = *activeInputIndex;
    match.activeOutputIndex = 0;
  }
}

GridKitSparseJacobianMatch
collectGridKitSparseJacobianMatch(LLVM::LLVMFuncOp func) {
  GridKitSparseJacobianMatch match;
  std::optional<std::string> residual;
  std::optional<SmallVector<std::string, 8>> enzymeActivity;
  bool ambiguousResidual = false;
  bool ambiguousActivity = false;

  func.walk([&](LLVM::CallOp call) {
    auto callee = call.getCallee();
    if (!callee)
      return;
    if (isEnzymeFwddiffCallee(*callee)) {
      match.fwddiffCalls.push_back(call);
      std::optional<std::string> callResidual = getFwddiffResidualSymbol(call);
      if (!callResidual) {
        ambiguousResidual = true;
      } else if (!residual) {
        residual = *callResidual;
      } else if (*residual != *callResidual) {
        ambiguousResidual = true;
      }

      SmallVector<std::string, 8> callActivities =
          getFwddiffActivitySymbols(call);
      if (!callActivities.empty()) {
        if (!enzymeActivity) {
          enzymeActivity = callActivities;
        } else if (!sameActivitySequence(*enzymeActivity, callActivities)) {
          ambiguousActivity = true;
        }
      }
    }
    if (isEnzymeTodenseCallee(*callee))
      match.todenseCalls.push_back(call);
  });

  func.walk([&](LLVM::AddressOfOp addressOf) {
    if (isGridKitSparseStoreAddress(addressOf.getGlobalName()))
      match.sparseStoreAddresses.push_back(addressOf);
  });

  if (!ambiguousResidual)
    match.residual = residual;
  if (!ambiguousActivity)
    match.enzymeActivity = enzymeActivity;
  deriveInputOutputActivity(func, match);
  deriveSparseLayoutMetadata(func, match);

  return match;
}

std::optional<JacobianVectorAction>
matchJacobianVectorAction(stablehlo::DotGeneralOp op) {
  auto lhsOp = op.getLhs().getDefiningOp<enzyme::JacobianOp>();
  auto rhsOp = op.getRhs().getDefiningOp<enzyme::JacobianOp>();

  if (!lhsOp && !rhsOp)
    return std::nullopt;

  if (lhsOp && rhsOp)
    return std::nullopt;

  bool isJacobianLHS = static_cast<bool>(lhsOp);
  enzyme::JacobianOp jacOp = isJacobianLHS ? lhsOp : rhsOp;
  SymbolTableCollection symbolTable;
  auto fn = dyn_cast_or_null<FunctionOpInterface>(
      symbolTable.lookupNearestSymbolFrom(jacOp, jacOp.getFnAttr()));
  if (!fn)
    return std::nullopt;

  auto nargs = fn.getNumArguments();
  auto nouts = fn.getNumResults();
  auto J = cast<OpResult>(isJacobianLHS ? op.getLhs() : op.getRhs());
  auto jidx = J.getResultNumber();

  if (jidx >= nargs * nouts)
    return std::nullopt;

  auto diffoIdx = static_cast<unsigned>(jidx / nargs);
  auto diffinIdx = static_cast<unsigned>(jidx % nargs);
  Value dvec = isJacobianLHS ? op.getRhs() : op.getLhs();

  bool isJVP = true;
  bool isVJP = true;
  ArrayRef<int64_t> jrdims =
      isJacobianLHS ? op.getDotDimensionNumbers().getLhsContractingDimensions()
                    : op.getDotDimensionNumbers().getRhsContractingDimensions();

  auto jacobianType = dyn_cast<RankedTensorType>(J.getType());
  if (!jacobianType)
    return std::nullopt;

  auto inputType =
      dyn_cast<RankedTensorType>(fn.getArgument(diffinIdx).getType());
  if (!inputType)
    return std::nullopt;

  auto totaldims = jacobianType.getNumElements();
  auto nindims = inputType.getNumElements();
  auto noutdims = totaldims - nindims;

  for (auto dimid : jrdims) {
    isJVP = isJVP && (dimid < noutdims);
    isVJP = isVJP && (dimid >= noutdims);
  }

  if (!isJVP && !isVJP)
    return std::nullopt;

  if (isJVP && isVJP)
    return std::nullopt;

  return JacobianVectorAction{jacOp, diffoIdx, diffinIdx, dvec, isJVP, isVJP};
}

LogicalResult
replaceWithForwardDiff(Operation *op, enzyme::JacobianOp jacOp,
                       unsigned diffOutputIndex,
                       const llvm::DenseMap<unsigned, Value> &inputTangents,
                       PatternRewriter &rewriter) {
  SmallVector<Value> inArgs;
  SmallVector<ActivityAttr, 2> newInActivityArgs;
  SmallVector<ActivityAttr, 2> newRetActivityArgs;
  for (auto [idx, act] :
       llvm::enumerate(jacOp.getActivity().getAsRange<ActivityAttr>())) {
    Value in = jacOp.getInputs()[idx];
    auto tangent = inputTangents.find(idx);
    if (tangent == inputTangents.end()) {
      inArgs.push_back(in);
      newInActivityArgs.push_back(
          ActivityAttr::get(rewriter.getContext(), Activity::enzyme_const));
      continue;
    }

    inArgs.push_back(in);
    inArgs.push_back(tangent->second);
    newInActivityArgs.push_back(
        ActivityAttr::get(rewriter.getContext(), Activity::enzyme_dup));
  }

  for (auto [idx, retAct] :
       llvm::enumerate(jacOp.getRetActivity().getAsRange<ActivityAttr>())) {
    if (idx == diffOutputIndex) {
      newRetActivityArgs.push_back(
          ActivityAttr::get(rewriter.getContext(), Activity::enzyme_dupnoneed));
    } else {
      newRetActivityArgs.push_back(ActivityAttr::get(
          rewriter.getContext(), Activity::enzyme_constnoneed));
    }
  }

  ArrayAttr newInActivity =
      ArrayAttr::get(rewriter.getContext(),
                     llvm::ArrayRef<Attribute>(newInActivityArgs.begin(),
                                               newInActivityArgs.end()));
  ArrayAttr newRetActivity =
      ArrayAttr::get(rewriter.getContext(),
                     llvm::ArrayRef<Attribute>(newRetActivityArgs.begin(),
                                               newRetActivityArgs.end()));

  rewriter.replaceOpWithNewOp<ForwardDiffOp>(
      op, op->getResultTypes(), jacOp.getFnAttr(), inArgs, newInActivity,
      newRetActivity, nullptr, jacOp.getStrongZeroAttr());
  return success();
}

struct IdaEffectiveJacobianActionLowering
    : public OpRewritePattern<stablehlo::AddOp> {
  using OpRewritePattern<stablehlo::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsDot = op.getLhs().getDefiningOp<stablehlo::DotGeneralOp>();
    auto rhsDot = op.getRhs().getDefiningOp<stablehlo::DotGeneralOp>();
    if (!lhsDot || !rhsDot)
      return failure();

    auto lhsAction = matchJacobianVectorAction(lhsDot);
    auto rhsAction = matchJacobianVectorAction(rhsDot);
    if (!lhsAction || !rhsAction)
      return failure();

    if (!lhsAction->isJVP || !rhsAction->isJVP)
      return failure();

    if (lhsAction->jacOp.getOperation() != rhsAction->jacOp.getOperation())
      return failure();

    if (lhsAction->diffOutputIndex != rhsAction->diffOutputIndex)
      return failure();

    if (lhsAction->diffInputIndex == rhsAction->diffInputIndex)
      return failure();

    llvm::DenseMap<unsigned, Value> inputTangents;
    inputTangents[lhsAction->diffInputIndex] = lhsAction->vector;
    inputTangents[rhsAction->diffInputIndex] = rhsAction->vector;

    if (failed(replaceWithForwardDiff(op, lhsAction->jacOp,
                                      lhsAction->diffOutputIndex, inputTangents,
                                      rewriter)))
      return failure();

    if (lhsDot->use_empty())
      rewriter.eraseOp(lhsDot);
    if (rhsDot->use_empty())
      rewriter.eraseOp(rhsDot);

    return success();
  }
};

struct DotGeneralLowering : public OpRewritePattern<stablehlo::DotGeneralOp> {
  DotGeneralLowering(MLIRContext *context,
                     bool requireGridKitIdaJacTimes = false,
                     bool requireJVP = false)
      : OpRewritePattern<stablehlo::DotGeneralOp>(context),
        requireGridKitIdaJacTimes(requireGridKitIdaJacTimes),
        requireJVP(requireJVP) {}

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    if (requireGridKitIdaJacTimes) {
      auto solver = op->getAttrOfType<StringAttr>(kGridKitSolverAttr);
      if (!solver || solver.getValue() != kGridKitIdaJacTimes)
        return failure();
    }

    auto action = matchJacobianVectorAction(op);
    if (!action)
      return failure();
    if (requireJVP && !action->isJVP)
      return failure();

    enzyme::JacobianOp jacOp = action->jacOp;
    auto diffo_idx = action->diffOutputIndex;
    auto diffin_idx = action->diffInputIndex;
    Value dvec = action->vector;

    // TODO: add batching support
    if (action->isJVP) {
      // JVP -> enzyme.fwddiff transform
      // The resulting fwddiff op will only have in_idx -> enzyme_dup, out_idx
      // -> enzyme_dupnoneed

      llvm::DenseMap<unsigned, Value> inputTangents;
      inputTangents[diffin_idx] = dvec;
      return replaceWithForwardDiff(op, jacOp, diffo_idx, inputTangents,
                                    rewriter);
    } else {
      // VJP -> enzyme.autodiff transform
      // The resulting autodiff op will only have in_idx -> enzyme_dup, out_idx
      // -> enzyme_dupnoneed

      SmallVector<Value> in_args(jacOp.getInputs());
      SmallVector<ActivityAttr, 2> newInActivityArgs;
      SmallVector<ActivityAttr, 2> newRetActivityArgs;
      for (auto [idx, act] :
           llvm::enumerate(jacOp.getActivity().getAsRange<ActivityAttr>())) {

        if (idx != diffin_idx) {
          newInActivityArgs.push_back(
              ActivityAttr::get(rewriter.getContext(), Activity::enzyme_const));
        } else {
          newInActivityArgs.push_back(ActivityAttr::get(
              rewriter.getContext(), Activity::enzyme_active));
        }
      }

      // push dvec
      in_args.push_back(dvec);

      // construct ret_args
      for (auto [idx, ret_act] :
           llvm::enumerate(jacOp.getRetActivity().getAsRange<ActivityAttr>())) {
        if (idx == diffo_idx) {
          // accounts for dvec
          newRetActivityArgs.push_back(ActivityAttr::get(
              rewriter.getContext(), Activity::enzyme_activenoneed));
        } else {
          newRetActivityArgs.push_back(ActivityAttr::get(
              rewriter.getContext(), Activity::enzyme_constnoneed));
        }
      }

      ArrayAttr newInActivity =
          ArrayAttr::get(rewriter.getContext(),
                         llvm::ArrayRef<Attribute>(newInActivityArgs.begin(),
                                                   newInActivityArgs.end()));
      ArrayAttr newRetActivity =
          ArrayAttr::get(rewriter.getContext(),
                         llvm::ArrayRef<Attribute>(newRetActivityArgs.begin(),
                                                   newRetActivityArgs.end()));

      rewriter.replaceOpWithNewOp<AutoDiffOp>(
          op, op->getResultTypes(), jacOp.getFnAttr(), in_args, newInActivity,
          newRetActivity, nullptr, jacOp.getStrongZeroAttr());
    }

    return success();
  }

private:
  bool requireGridKitIdaJacTimes;
  bool requireJVP;
};

struct LowerEnzymeJacobianStableHLO
    : public mlir::enzyme::impl::LowerEnzymeJacobianStableHLOBase<
          LowerEnzymeJacobianStableHLO> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DotGeneralLowering>(context);

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }

    // // Verify that all illegal ops have been lowered
    // auto walkResult = getOperation()->walk([&](Operation *op) {
    //   if (isa<enzyme::ConcatOp, enzyme::ExtractOp>(op)) {
    //     op->emitError("Failed to lower enzyme batch operation");
    //     return WalkResult::interrupt();
    //   }
    //   return WalkResult::advance();
    // });
    //
    // if (walkResult.wasInterrupted()) {
    //   signalPassFailure();
    // }
  };
};

struct LowerGridKitIdaJacobianActionStableHLO
    : public mlir::enzyme::impl::LowerGridKitIdaJacobianActionStableHLOBase<
          LowerGridKitIdaJacobianActionStableHLO> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DotGeneralLowering>(context, true);

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  };
};

bool hasLiveResult(enzyme::JacobianOp jacOp) {
  return llvm::any_of(jacOp->getResults(),
                      [](Value result) { return !result.use_empty(); });
}

LogicalResult eraseDeadJacobiansAndRejectLiveUses(Operation *op) {
  SmallVector<enzyme::JacobianOp> deadJacobians;
  Operation *liveJacobian = nullptr;

  op->walk([&](enzyme::JacobianOp jacOp) {
    if (hasLiveResult(jacOp)) {
      if (!liveJacobian)
        liveJacobian = jacOp.getOperation();
      return;
    }
    deadJacobians.push_back(jacOp);
  });

  if (liveJacobian) {
    liveJacobian->emitError()
        << "unsupported live Jacobian materialization remains after "
           "SUNDIALS IDA Jacobian-action lowering";
    return failure();
  }

  for (enzyme::JacobianOp jacOp : deadJacobians)
    jacOp.erase();

  return success();
}

LogicalResult lowerJacobianActionsIn(Operation *op, bool requireOnlyJVP = false,
                                     bool rejectLiveJacobians = false) {
  MLIRContext *context = op->getContext();

  RewritePatternSet effectiveJacobianPatterns(context);
  effectiveJacobianPatterns.add<IdaEffectiveJacobianActionLowering>(context);
  GreedyRewriteConfig config;
  config.enableFolding();
  if (failed(applyPatternsGreedily(op, std::move(effectiveJacobianPatterns),
                                   config)))
    return failure();

  RewritePatternSet dotPatterns(context);
  dotPatterns.add<DotGeneralLowering>(context, false, requireOnlyJVP);
  if (failed(applyPatternsGreedily(op, std::move(dotPatterns), config)))
    return failure();

  if (rejectLiveJacobians && failed(eraseDeadJacobiansAndRejectLiveUses(op)))
    return failure();

  return success();
}

std::optional<std::string> getRootSymbolString(Attribute attr) {
  auto symbol = dyn_cast_or_null<SymbolRefAttr>(attr);
  if (!symbol)
    return std::nullopt;
  return symbol.getRootReference().str();
}

bool isJacobianMaterializationRecord(Operation *op) {
  return op->getName().getStringRef() == kJacobianMaterializationOpName;
}

bool isJacobianActionRecord(Operation *op) {
  return op->getName().getStringRef() == kJacobianActionOpName;
}

bool hasRequiredJacobianActionMetadata(Operation *op) {
  return op->getAttrOfType<SymbolRefAttr>("materializer") &&
         op->getAttrOfType<SymbolRefAttr>("residual") &&
         op->getAttrOfType<IntegerAttr>("active_input_index") &&
         op->getAttrOfType<IntegerAttr>("active_output_index");
}

std::string getUniqueJacobianActionSymbol(llvm::StringSet<> &usedSymbols) {
  unsigned index = 0;
  while (true) {
    std::string candidate =
        (llvm::Twine("__enzymexla_jacobian_action_") + llvm::Twine(index))
            .str();
    if (!usedSymbols.contains(candidate)) {
      usedSymbols.insert(candidate);
      return candidate;
    }
    ++index;
  }
}

std::string getUniqueSundialsIdaEffectiveJacobianActionSymbol(
    llvm::StringSet<> &usedSymbols) {
  unsigned index = 0;
  while (true) {
    std::string candidate =
        (llvm::Twine("__enzymexla_sundials_ida_effective_jacobian_action_") +
         llvm::Twine(index))
            .str();
    if (!usedSymbols.contains(candidate)) {
      usedSymbols.insert(candidate);
      return candidate;
    }
    ++index;
  }
}

int64_t getExistingI64ModuleAttr(ModuleOp module, StringRef name) {
  if (auto attr = module->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return 0;
}

void addUniqueResidualAction(llvm::StringMap<std::string> &actionByResidual,
                             llvm::StringSet<> &ambiguousResiduals,
                             StringRef residual, StringRef action) {
  if (ambiguousResiduals.contains(residual))
    return;

  auto iter = actionByResidual.find(residual);
  if (iter == actionByResidual.end()) {
    actionByResidual[residual] = action.str();
    return;
  }

  if (iter->second == action)
    return;

  actionByResidual.erase(iter);
  ambiguousResiduals.insert(residual);
}

struct JacobianMaterializationSummary {
  Operation *op = nullptr;
  std::string materializer;
  std::string residual;
  int64_t activeInputIndex = -1;
  int64_t activeOutputIndex = -1;
};

struct EffectiveIdaActionToCreate {
  JacobianMaterializationSummary yMaterialization;
  JacobianMaterializationSummary ypMaterialization;
  std::string actionName;
};

std::optional<JacobianMaterializationSummary>
getJacobianMaterializationSummary(Operation *op) {
  std::optional<std::string> materializer =
      getRootSymbolString(op->getAttr("materializer"));
  std::optional<std::string> residual =
      getRootSymbolString(op->getAttr("residual"));
  auto activeInput = op->getAttrOfType<IntegerAttr>("active_input_index");
  auto activeOutput = op->getAttrOfType<IntegerAttr>("active_output_index");
  if (!materializer || !residual || !activeInput || !activeOutput)
    return std::nullopt;

  return JacobianMaterializationSummary{op, *materializer, *residual,
                                        activeInput.getInt(),
                                        activeOutput.getInt()};
}

std::string getEffectiveIdaActionKey(StringRef yMaterializer,
                                     StringRef ypMaterializer) {
  return (llvm::Twine(yMaterializer) + "|" + ypMaterializer).str();
}

bool isMatrixFreeCandidateSolve(enzymexla::SundialsIdaSolveOp solve) {
  if (solve.getJacobianAction())
    return false;
  if (solve.getJacobianDemand() ==
          enzymexla::SundialsIdaJacobianDemand::jacobian_action &&
      solve.getLinearSolver() ==
          enzymexla::SundialsIdaLinearSolver::jacobian_action_iterative)
    return true;
  return solve->hasAttr(kSundialsAllowMatrixFreeAttr) &&
         solve.getJacobianDemand() ==
             enzymexla::SundialsIdaJacobianDemand::explicit_matrix &&
         (solve.getLinearSolver() ==
              enzymexla::SundialsIdaLinearSolver::explicit_sparse_direct ||
          solve.getLinearSolver() ==
              enzymexla::SundialsIdaLinearSolver::explicit_dense_direct);
}

std::optional<JacobianMaterializationSummary>
findUniqueMaterializationForResidual(
    ArrayRef<JacobianMaterializationSummary> materializations,
    StringRef residual, std::optional<int64_t> activeInputIndex,
    std::optional<int64_t> activeOutputIndex) {
  std::optional<JacobianMaterializationSummary> result;
  for (const JacobianMaterializationSummary &candidate : materializations) {
    if (candidate.residual != residual)
      continue;
    if (activeInputIndex && candidate.activeInputIndex != *activeInputIndex)
      continue;
    if (activeOutputIndex && candidate.activeOutputIndex != *activeOutputIndex)
      continue;
    if (result)
      return std::nullopt;
    result = candidate;
  }
  return result;
}

std::optional<JacobianMaterializationSummary>
findYMaterializationForSolve(
    enzymexla::SundialsIdaSolveOp solve,
    const llvm::StringMap<JacobianMaterializationSummary> &materializationBySymbol,
    ArrayRef<JacobianMaterializationSummary> materializations) {
  if (auto jacobian = solve.getJacobian()) {
    auto iter = materializationBySymbol.find(jacobian->getRootReference());
    if (iter != materializationBySymbol.end())
      return iter->second;
  }

  return findUniqueMaterializationForResidual(
      materializations, solve.getResidual().getRootReference(),
      /*activeInputIndex=*/std::nullopt,
      /*activeOutputIndex=*/0);
}

std::optional<JacobianMaterializationSummary> findYpMaterializationForY(
    const JacobianMaterializationSummary &yMaterialization,
    ArrayRef<JacobianMaterializationSummary> materializations) {
  return findUniqueMaterializationForResidual(
      materializations, yMaterialization.residual,
      yMaterialization.activeInputIndex + 1,
      yMaterialization.activeOutputIndex);
}

void copyJacobianActionMetadata(Operation *materialization,
                                OperationState &state) {
  constexpr llvm::StringLiteral kCopiedAttrs[] = {
      "enzyme_activity",
      "input_activity",
      "input_count",
      "output_activity",
      "output_count",
      "output_dimension_arg",
      "active_input_dimension_arg",
      "seed_loop_dimension_arg",
      "output_index_map_arg",
      "active_input_index_map_arg",
      "sparse_assembly",
      "sparse_rows_arg",
      "sparse_cols_arg",
      "sparse_values_arg",
      "sparse_nnz_arg",
  };

  for (llvm::StringLiteral name : kCopiedAttrs) {
    if (Attribute attr = materialization->getAttr(name))
      state.addAttribute(name, attr);
  }
}

struct SundialsIdaLLVMConfig {
  std::string sourceFunction;
  std::optional<std::string> residual;
  std::optional<std::string> jacobian;
  std::optional<std::string> jacobianAction;
  std::optional<std::string> preconditioner;
  bool hasLocalResidualRegistration = false;
  bool hasUserDataRegistration = false;
  bool hasKluLinearSolver = false;
  bool hasDenseLinearSolver = false;
  bool hasIterativeLinearSolver = false;
};

bool isSundialsIdaSolveRecord(Operation *op) {
  return op->getName().getStringRef() == kSundialsIdaSolveOpName;
}

bool isSundialsIdaInitCallee(StringRef name) { return name == "IDAInit"; }

bool isSundialsIdaSetLinearSolverCallee(StringRef name) {
  return name == "IDASetLinearSolver";
}

bool isSundialsIdaSetUserDataCallee(StringRef name) {
  return name == "IDASetUserData";
}

bool isSundialsIdaSetJacFnCallee(StringRef name) {
  return name == "IDASetJacFn";
}

bool isSundialsIdaSetJacTimesCallee(StringRef name) {
  return name == "IDASetJacTimes";
}

bool isSundialsIdaSetPreconditionerCallee(StringRef name) {
  return name == "IDASetPreconditioner";
}

bool isSundialsKluLinearSolverCallee(StringRef name) {
  return name == "SUNLinSol_KLU";
}

bool isSundialsDenseLinearSolverCallee(StringRef name) {
  return name == "SUNLinSol_Dense";
}

bool isSundialsIterativeLinearSolverCallee(StringRef name) {
  return name == "SUNLinSol_SPGMR" || name == "SUNLinSol_SPFGMR" ||
         name == "SUNLinSol_SPBCGS" || name == "SUNLinSol_SPTFQMR" ||
         name == "SUNLinSol_PCG";
}

std::optional<std::string> getAddressOfSymbol(Value value) {
  if (auto addressOf = value.getDefiningOp<LLVM::AddressOfOp>())
    return addressOf.getGlobalName().str();
  if (auto bitcast = value.getDefiningOp<LLVM::BitcastOp>())
    return getAddressOfSymbol(bitcast.getArg());
  return std::nullopt;
}

std::optional<std::string> getCallOperandSymbol(LLVM::CallOp call,
                                                unsigned operandIndex) {
  auto operands = call.getArgOperands();
  if (operandIndex >= operands.size())
    return std::nullopt;
  return getAddressOfSymbol(operands[operandIndex]);
}

std::optional<std::string> getLastCallOperandSymbol(LLVM::CallOp call) {
  auto operands = call.getArgOperands();
  for (Value operand : llvm::reverse(operands)) {
    std::optional<std::string> symbol = getAddressOfSymbol(operand);
    if (symbol)
      return symbol;
  }
  return std::nullopt;
}

void setSundialsCallRole(LLVM::CallOp call, Builder &builder, StringRef role) {
  call->setAttr(kSundialsRoleAttr, builder.getStringAttr(role));
}

bool mergeOptionalSymbol(std::optional<std::string> &dest,
                         const std::optional<std::string> &src) {
  if (dest || !src)
    return false;
  dest = src;
  return true;
}

bool mergeSundialsIdaLLVMConfig(SundialsIdaLLVMConfig &dest,
                                const SundialsIdaLLVMConfig &src) {
  bool changed = false;
  changed |= mergeOptionalSymbol(dest.residual, src.residual);
  changed |= mergeOptionalSymbol(dest.jacobian, src.jacobian);
  changed |= mergeOptionalSymbol(dest.jacobianAction, src.jacobianAction);
  changed |= mergeOptionalSymbol(dest.preconditioner, src.preconditioner);

  if (src.hasKluLinearSolver && !dest.hasKluLinearSolver) {
    dest.hasKluLinearSolver = true;
    changed = true;
  }
  if (src.hasDenseLinearSolver && !dest.hasDenseLinearSolver) {
    dest.hasDenseLinearSolver = true;
    changed = true;
  }
  if (src.hasIterativeLinearSolver && !dest.hasIterativeLinearSolver) {
    dest.hasIterativeLinearSolver = true;
    changed = true;
  }
  if (src.hasUserDataRegistration && !dest.hasUserDataRegistration) {
    dest.hasUserDataRegistration = true;
    changed = true;
  }
  return changed;
}

bool hasAnySundialsIdaLLVMConfig(const SundialsIdaLLVMConfig &config) {
  return config.residual || config.jacobian || config.jacobianAction ||
         config.preconditioner || config.hasUserDataRegistration ||
         config.hasKluLinearSolver || config.hasDenseLinearSolver ||
         config.hasIterativeLinearSolver;
}

bool isRecoverableSundialsIdaLLVMConfig(const SundialsIdaLLVMConfig &config) {
  if (!config.hasLocalResidualRegistration || !config.residual)
    return false;
  return config.hasKluLinearSolver || config.hasDenseLinearSolver ||
         config.hasIterativeLinearSolver || config.jacobian ||
         config.jacobianAction;
}

SundialsIdaLLVMConfig
collectLocalSundialsIdaLLVMConfig(LLVM::LLVMFuncOp func, Builder &builder) {
  SundialsIdaLLVMConfig config;
  config.sourceFunction = func.getName().str();

  func.walk([&](LLVM::CallOp call) {
    auto callee = call.getCallee();
    if (!callee)
      return;

    StringRef name = *callee;
    if (isSundialsIdaInitCallee(name)) {
      config.hasLocalResidualRegistration = true;
      if (!config.residual)
        config.residual = getCallOperandSymbol(call, 1);
      setSundialsCallRole(call, builder, "ida_residual_registration");
      return;
    }
    if (isSundialsIdaSetUserDataCallee(name)) {
      config.hasUserDataRegistration = true;
      setSundialsCallRole(call, builder, "ida_user_data_registration");
      return;
    }
    if (isSundialsIdaSetLinearSolverCallee(name)) {
      setSundialsCallRole(call, builder, "ida_linear_solver_registration");
      return;
    }
    if (isSundialsIdaSetJacFnCallee(name)) {
      if (!config.jacobian)
        config.jacobian = getCallOperandSymbol(call, 1);
      setSundialsCallRole(call, builder, "ida_jacobian_registration");
      return;
    }
    if (isSundialsIdaSetJacTimesCallee(name)) {
      if (!config.jacobianAction)
        config.jacobianAction = getLastCallOperandSymbol(call);
      setSundialsCallRole(call, builder, "ida_jacobian_action_registration");
      return;
    }
    if (isSundialsIdaSetPreconditionerCallee(name)) {
      if (!config.preconditioner)
        config.preconditioner = getLastCallOperandSymbol(call);
      setSundialsCallRole(call, builder, "ida_preconditioner_registration");
      return;
    }
    if (isSundialsKluLinearSolverCallee(name)) {
      config.hasKluLinearSolver = true;
      setSundialsCallRole(call, builder, "ida_sparse_direct_linear_solver");
      return;
    }
    if (isSundialsDenseLinearSolverCallee(name)) {
      config.hasDenseLinearSolver = true;
      setSundialsCallRole(call, builder, "ida_dense_direct_linear_solver");
      return;
    }
    if (isSundialsIterativeLinearSolverCallee(name)) {
      config.hasIterativeLinearSolver = true;
      setSundialsCallRole(call, builder, "ida_iterative_linear_solver");
      return;
    }
  });

  return config;
}

enzymexla::SundialsIdaLinearSolver
getRecoveredIdaLinearSolver(const SundialsIdaLLVMConfig &config) {
  if (config.jacobianAction || config.hasIterativeLinearSolver)
    return enzymexla::SundialsIdaLinearSolver::jacobian_action_iterative;
  if (config.hasKluLinearSolver)
    return enzymexla::SundialsIdaLinearSolver::explicit_sparse_direct;
  return enzymexla::SundialsIdaLinearSolver::explicit_dense_direct;
}

enzymexla::SundialsIdaJacobianDemand
getRecoveredIdaJacobianDemand(const SundialsIdaLLVMConfig &config) {
  if (config.jacobianAction)
    return enzymexla::SundialsIdaJacobianDemand::jacobian_action;
  if (config.jacobian)
    return enzymexla::SundialsIdaJacobianDemand::explicit_matrix;
  return enzymexla::SundialsIdaJacobianDemand::none;
}

std::string getRecoveredIdaSolveKey(const SundialsIdaLLVMConfig &config) {
  return (llvm::Twine(config.sourceFunction) + "|" +
          (config.residual ? *config.residual : "") + "|" +
          (config.jacobian ? *config.jacobian : "") + "|" +
          (config.jacobianAction ? *config.jacobianAction : ""))
      .str();
}

std::optional<std::string> getExistingIdaSolveSourceFunction(Operation *op) {
  auto sourceFunction = op->getAttrOfType<StringAttr>("source_function");
  if (!sourceFunction)
    return std::nullopt;
  return sourceFunction.getValue().str();
}

struct RecoverSundialsIdaLLVM
    : public mlir::enzyme::impl::RecoverSundialsIdaLLVMBase<
          RecoverSundialsIdaLLVM> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    Builder builder(context);
    llvm::StringSet<> existingSolveKeys;
    SmallVector<SundialsIdaLLVMConfig, 4> recoveredSolves;
    SmallVector<LLVM::LLVMFuncOp, 32> llvmFuncs;
    llvm::StringMap<SundialsIdaLLVMConfig> functionSummaries;

    module.walk([&](Operation *op) {
      if (!isSundialsIdaSolveRecord(op))
        return;
      std::optional<std::string> sourceFunction =
          getExistingIdaSolveSourceFunction(op);
      if (sourceFunction)
        existingSolveKeys.insert(*sourceFunction);
    });

    module.walk([&](LLVM::LLVMFuncOp func) {
      llvmFuncs.push_back(func);
      functionSummaries[func.getName()] =
          collectLocalSundialsIdaLLVMConfig(func, builder);
    });

    bool changed = true;
    while (changed) {
      changed = false;
      for (LLVM::LLVMFuncOp func : llvmFuncs) {
        auto summary = functionSummaries.find(func.getName());
        if (summary == functionSummaries.end())
          continue;

        SundialsIdaLLVMConfig &config = summary->second;
        func.walk([&](LLVM::CallOp call) {
          auto callee = call.getCallee();
          if (!callee || *callee == func.getName())
            return;

          auto calleeSummary = functionSummaries.find(*callee);
          if (calleeSummary == functionSummaries.end() ||
              !hasAnySundialsIdaLLVMConfig(calleeSummary->second))
            return;
          changed |=
              mergeSundialsIdaLLVMConfig(config, calleeSummary->second);
        });
      }
    }

    for (LLVM::LLVMFuncOp func : llvmFuncs) {
      auto summary = functionSummaries.find(func.getName());
      if (summary == functionSummaries.end())
        continue;

      SundialsIdaLLVMConfig &config = summary->second;
      if (!isRecoverableSundialsIdaLLVMConfig(config))
        continue;

      std::string key = getRecoveredIdaSolveKey(config);
      if (existingSolveKeys.contains(key) ||
          existingSolveKeys.contains(config.sourceFunction))
        continue;
      existingSolveKeys.insert(key);
      recoveredSolves.push_back(config);
    }

    OpBuilder opBuilder(context);
    opBuilder.setInsertionPointToStart(module.getBody());
    for (const SundialsIdaLLVMConfig &config : recoveredSolves) {
      OperationState state(module.getLoc(), kSundialsIdaSolveOpName);
      state.addAttribute("residual",
                         SymbolRefAttr::get(context, *config.residual));
      if (config.jacobian)
        state.addAttribute("jacobian",
                           SymbolRefAttr::get(context, *config.jacobian));
      if (config.jacobianAction)
        state.addAttribute(
            "jacobian_action",
            SymbolRefAttr::get(context, *config.jacobianAction));
      if (config.preconditioner)
        state.addAttribute(
            "preconditioner",
            SymbolRefAttr::get(context, *config.preconditioner));
      state.addAttribute(
          "linear_solver",
          enzymexla::SundialsIdaLinearSolverAttr::get(
              context, getRecoveredIdaLinearSolver(config)));
      state.addAttribute(
          "jacobian_demand",
          enzymexla::SundialsIdaJacobianDemandAttr::get(
              context, getRecoveredIdaJacobianDemand(config)));
      if (config.hasUserDataRegistration)
        state.addAttribute(kSundialsUserDataRegisteredAttr,
                           builder.getUnitAttr());
      state.addAttribute("source", builder.getStringAttr("llvm_sundials_ida"));
      state.addAttribute("source_function",
                         builder.getStringAttr(config.sourceFunction));
      opBuilder.create(state);
    }

    if (!recoveredSolves.empty())
      module->setAttr(kRecoveredSundialsIdaSolvesAttr,
                      builder.getI64IntegerAttr(recoveredSolves.size()));
  }
};

struct SynthesizeSundialsIdaJacobianActions
    : public mlir::enzyme::impl::SynthesizeSundialsIdaJacobianActionsBase<
          SynthesizeSundialsIdaJacobianActions> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    Builder builder(context);
    OpBuilder opBuilder(context);
    llvm::StringSet<> usedSymbols;
    llvm::StringMap<std::string> actionByMaterializer;
    llvm::StringMap<std::string> actionByResidual;
    llvm::StringSet<> ambiguousResiduals;
    llvm::StringMap<JacobianMaterializationSummary> materializationBySymbol;
    SmallVector<JacobianMaterializationSummary, 8> materializations;
    llvm::StringMap<std::string> effectiveActionByPair;
    SmallVector<std::pair<Operation *, std::string>, 8> actionsToCreate;
    SmallVector<EffectiveIdaActionToCreate, 4> effectiveActionsToCreate;

    module.walk([&](Operation *op) {
      if (auto symName = op->getAttrOfType<StringAttr>(
              SymbolTable::getSymbolAttrName()))
        usedSymbols.insert(symName.getValue());

      if (!isJacobianActionRecord(op))
        return;

      std::optional<std::string> materialization =
          getRootSymbolString(op->getAttr("materialization"));
      std::optional<std::string> ypMaterialization =
          getRootSymbolString(op->getAttr("yp_materialization"));
      auto symName =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
      if (materialization && symName) {
        if (op->hasAttr(kSundialsIdaEffectiveJacobianActionAttr)) {
          actionByMaterializer[*materialization] = symName.getValue().str();
          if (ypMaterialization) {
            effectiveActionByPair[getEffectiveIdaActionKey(
                *materialization, *ypMaterialization)] =
                symName.getValue().str();
          }
        } else if (!actionByMaterializer.contains(*materialization)) {
          actionByMaterializer[*materialization] = symName.getValue().str();
        }
      }

      std::optional<std::string> residual =
          getRootSymbolString(op->getAttr("residual"));
      if (residual && symName)
        addUniqueResidualAction(actionByResidual, ambiguousResiduals, *residual,
                                symName.getValue());
    });

    module.walk([&](Operation *op) {
      if (!isJacobianMaterializationRecord(op) ||
          !hasRequiredJacobianActionMetadata(op))
        return;

      std::optional<JacobianMaterializationSummary> summary =
          getJacobianMaterializationSummary(op);
      if (summary) {
        materializationBySymbol[summary->materializer] = *summary;
        materializations.push_back(*summary);
      }

      std::optional<std::string> materializer =
          getRootSymbolString(op->getAttr("materializer"));
      if (!materializer || actionByMaterializer.contains(*materializer))
        return;

      std::string actionName = getUniqueJacobianActionSymbol(usedSymbols);
      actionByMaterializer[*materializer] = actionName;
      if (std::optional<std::string> residual =
              getRootSymbolString(op->getAttr("residual")))
        addUniqueResidualAction(actionByResidual, ambiguousResiduals, *residual,
                                actionName);
      actionsToCreate.push_back({op, actionName});
    });

    module.walk([&](enzymexla::SundialsIdaSolveOp solve) {
      if (!isMatrixFreeCandidateSolve(solve))
        return;

      std::optional<JacobianMaterializationSummary> yMaterialization =
          findYMaterializationForSolve(solve, materializationBySymbol,
                                       materializations);
      if (!yMaterialization)
        return;
      std::optional<JacobianMaterializationSummary> ypMaterialization =
          findYpMaterializationForY(*yMaterialization, materializations);
      if (!ypMaterialization)
        return;

      std::string key = getEffectiveIdaActionKey(
          yMaterialization->materializer, ypMaterialization->materializer);
      if (effectiveActionByPair.contains(key)) {
        actionByMaterializer[yMaterialization->materializer] =
            effectiveActionByPair[key];
        return;
      }

      std::string actionName =
          getUniqueSundialsIdaEffectiveJacobianActionSymbol(usedSymbols);
      effectiveActionByPair[key] = actionName;
      actionByMaterializer[yMaterialization->materializer] = actionName;
      effectiveActionsToCreate.push_back(
          {*yMaterialization, *ypMaterialization, actionName});
    });

    opBuilder.setInsertionPointToStart(module.getBody());
    for (auto [materialization, actionName] : actionsToCreate) {
      OperationState state(module.getLoc(), kJacobianActionOpName);
      state.addAttribute(SymbolTable::getSymbolAttrName(),
                         builder.getStringAttr(actionName));
      state.addAttribute("materialization",
                         materialization->getAttr("materializer"));
      state.addAttribute("residual", materialization->getAttr("residual"));
      state.addAttribute("active_input_index",
                         materialization->getAttr("active_input_index"));
      state.addAttribute("active_output_index",
                         materialization->getAttr("active_output_index"));
      if (Attribute source = materialization->getAttr("source"))
        state.addAttribute("source", source);
      copyJacobianActionMetadata(materialization, state);
      opBuilder.create(state);
    }
    for (const EffectiveIdaActionToCreate &effectiveAction :
         effectiveActionsToCreate) {
      OperationState state(module.getLoc(), kJacobianActionOpName);
      state.addAttribute(SymbolTable::getSymbolAttrName(),
                         builder.getStringAttr(effectiveAction.actionName));
      state.addAttribute("materialization",
                         SymbolRefAttr::get(
                             context,
                             effectiveAction.yMaterialization.materializer));
      state.addAttribute("residual",
                         SymbolRefAttr::get(
                             context,
                             effectiveAction.yMaterialization.residual));
      state.addAttribute(
          "active_input_index",
          builder.getI64IntegerAttr(
              effectiveAction.yMaterialization.activeInputIndex));
      state.addAttribute(
          "active_output_index",
          builder.getI64IntegerAttr(
              effectiveAction.yMaterialization.activeOutputIndex));
      state.addAttribute(kSundialsIdaEffectiveJacobianActionAttr,
                         builder.getUnitAttr());
      state.addAttribute(
          "source", builder.getStringAttr("sundials_ida_effective_jacobian"));
      state.addAttribute("y_materialization",
                         SymbolRefAttr::get(
                             context,
                             effectiveAction.yMaterialization.materializer));
      state.addAttribute("yp_materialization",
                         SymbolRefAttr::get(
                             context,
                             effectiveAction.ypMaterialization.materializer));
      state.addAttribute(
          "yp_active_input_index",
          builder.getI64IntegerAttr(
              effectiveAction.ypMaterialization.activeInputIndex));
      copyJacobianActionMetadata(effectiveAction.yMaterialization.op, state);
      opBuilder.create(state);
    }

    unsigned linkedSolves = 0;
    module.walk([&](enzymexla::SundialsIdaSolveOp solve) {
      if (solve.getJacobianAction())
        return;
      if (solve.getJacobianDemand() !=
              enzymexla::SundialsIdaJacobianDemand::jacobian_action ||
          solve.getLinearSolver() !=
              enzymexla::SundialsIdaLinearSolver::jacobian_action_iterative)
        return;

      std::optional<std::string> actionName;
      if (auto jacobian = solve.getJacobian()) {
        std::string materializer = jacobian->getRootReference().str();
        auto iter = actionByMaterializer.find(materializer);
        if (iter != actionByMaterializer.end())
          actionName = iter->second;
      }

      if (!actionName) {
        std::string residual = solve.getResidual().getRootReference().str();
        auto iter = actionByResidual.find(residual);
        if (iter != actionByResidual.end())
          actionName = iter->second;
      }

      if (!actionName)
        return;

      solve->setAttr("jacobian_action",
                     SymbolRefAttr::get(context, *actionName));
      ++linkedSolves;
    });

    unsigned totalActionsCreated =
        actionsToCreate.size() + effectiveActionsToCreate.size();
    if (totalActionsCreated != 0)
      module->setAttr(kSynthesizedJacobianActionsAttr,
                      builder.getI64IntegerAttr(
                          getExistingI64ModuleAttr(
                              module, kSynthesizedJacobianActionsAttr) +
                          totalActionsCreated));
    if (!effectiveActionsToCreate.empty())
      module->setAttr(
          kSundialsIdaEffectiveJacobianActionsSynthesizedAttr,
          builder.getI64IntegerAttr(
              getExistingI64ModuleAttr(
                  module, kSundialsIdaEffectiveJacobianActionsSynthesizedAttr) +
              effectiveActionsToCreate.size()));
    if (linkedSolves != 0)
      module->setAttr(kLinkedSundialsIdaJacobianActionsAttr,
                      builder.getI64IntegerAttr(linkedSolves));
  }
};

struct SelectSundialsIdaMatrixFree
    : public mlir::enzyme::impl::SelectSundialsIdaMatrixFreeBase<
          SelectSundialsIdaMatrixFree> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    Builder builder(context);
    llvm::StringMap<std::string> actionByMaterializer;
    llvm::StringMap<std::string> actionByResidual;
    llvm::StringSet<> ambiguousResiduals;

    module.walk([&](Operation *op) {
      if (!isJacobianActionRecord(op))
        return;

      auto symName =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
      if (!symName)
        return;

      std::optional<std::string> materialization =
          getRootSymbolString(op->getAttr("materialization"));
      if (materialization) {
        auto existing = actionByMaterializer.find(*materialization);
        if (existing == actionByMaterializer.end() ||
            op->hasAttr(kSundialsIdaEffectiveJacobianActionAttr))
          actionByMaterializer[*materialization] = symName.getValue().str();
      }

      std::optional<std::string> residual =
          getRootSymbolString(op->getAttr("residual"));
      if (residual)
        addUniqueResidualAction(actionByResidual, ambiguousResiduals, *residual,
                                symName.getValue());
    });

    unsigned selectedSolves = 0;
    module.walk([&](enzymexla::SundialsIdaSolveOp solve) {
      if (!solve->hasAttr(kSundialsAllowMatrixFreeAttr))
        return;
      if (solve.getJacobianDemand() !=
          enzymexla::SundialsIdaJacobianDemand::explicit_matrix)
        return;
      if (solve.getLinearSolver() !=
              enzymexla::SundialsIdaLinearSolver::explicit_sparse_direct &&
          solve.getLinearSolver() !=
              enzymexla::SundialsIdaLinearSolver::explicit_dense_direct)
        return;

      std::optional<std::string> actionName;
      if (auto jacobianAction = solve.getJacobianAction())
        actionName = jacobianAction->getRootReference().str();

      if (!actionName) {
        if (auto jacobian = solve.getJacobian()) {
          std::string materializer = jacobian->getRootReference().str();
          auto iter = actionByMaterializer.find(materializer);
          if (iter != actionByMaterializer.end())
            actionName = iter->second;
        }
      }

      if (!actionName) {
        std::string residual = solve.getResidual().getRootReference().str();
        if (!ambiguousResiduals.contains(residual)) {
          auto iter = actionByResidual.find(residual);
          if (iter != actionByResidual.end())
            actionName = iter->second;
        }
      }

      if (!actionName)
        return;

      solve->setAttr("jacobian_action",
                     SymbolRefAttr::get(context, *actionName));
      solve->setAttr(
          "linear_solver",
          enzymexla::SundialsIdaLinearSolverAttr::get(
              context,
              enzymexla::SundialsIdaLinearSolver::jacobian_action_iterative));
      solve->setAttr(
          "jacobian_demand",
          enzymexla::SundialsIdaJacobianDemandAttr::get(
              context,
              enzymexla::SundialsIdaJacobianDemand::jacobian_action));
      solve->setAttr(kSundialsMatrixFreeSelectedAttr, builder.getUnitAttr());
      ++selectedSolves;
    });

    if (selectedSolves != 0)
      module->setAttr(kSelectedSundialsIdaMatrixFreeAttr,
                      builder.getI64IntegerAttr(selectedSolves));
  }
};

std::string getUniqueSundialsRuntimeSymbol(llvm::StringSet<> &usedSymbols,
                                           StringRef prefix) {
  unsigned index = 0;
  while (true) {
    std::string candidate = (llvm::Twine(prefix) + llvm::Twine(index)).str();
    if (!usedSymbols.contains(candidate)) {
      usedSymbols.insert(candidate);
      return candidate;
    }
    ++index;
  }
}

LLVM::LLVMFuncOp getOrCreateLLVMDeclaration(ModuleOp module,
                                            OpBuilder &builder, Location loc,
                                            StringRef name,
                                            LLVM::LLVMFunctionType type) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  return LLVM::LLVMFuncOp::create(builder, loc, name, type,
                                  LLVM::Linkage::External);
}

void ensureSundialsIdaRuntimeDeclarations(ModuleOp module, OpBuilder &builder,
                                          Location loc) {
  MLIRContext *context = module.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i32Type = IntegerType::get(context, 32);
  Type f64Type = Float64Type::get(context);

  getOrCreateLLVMDeclaration(
      module, builder, loc, "SUNLinSol_SPGMR",
      LLVM::LLVMFunctionType::get(ptrType, {ptrType, i32Type, i32Type, ptrType},
                                  /*isVarArg=*/false));
  getOrCreateLLVMDeclaration(
      module, builder, loc, "IDASetLinearSolver",
      LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType, ptrType},
                                  /*isVarArg=*/false));
  getOrCreateLLVMDeclaration(
      module, builder, loc, "IDASetUserData",
      LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType},
                                  /*isVarArg=*/false));
  getOrCreateLLVMDeclaration(
      module, builder, loc, "IDASetJacTimes",
      LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType, ptrType},
                                  /*isVarArg=*/false));
  getOrCreateLLVMDeclaration(
      module, builder, loc, "N_VGetArrayPointer",
      LLVM::LLVMFunctionType::get(ptrType, {ptrType}, /*isVarArg=*/false));
  getOrCreateLLVMDeclaration(
      module, builder, loc, "N_VScale",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                  {f64Type, ptrType, ptrType},
                                  /*isVarArg=*/false));
  getOrCreateLLVMDeclaration(
      module, builder, loc, "__enzymexla_sundials_ida_register_jvp_context",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), {ptrType},
                                  /*isVarArg=*/false));
}

void ensureSundialsIdaRawJvpContextDeclarations(ModuleOp module,
                                                OpBuilder &builder,
                                                Location loc) {
  MLIRContext *context = module.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);

  getOrCreateLLVMDeclaration(
      module, builder, loc, "__enzymexla_sundials_ida_context_input",
      LLVM::LLVMFunctionType::get(ptrType, {ptrType, i64Type},
                                  /*isVarArg=*/false));
  getOrCreateLLVMDeclaration(
      module, builder, loc, "__enzymexla_sundials_ida_accumulate_raw_jvp",
      LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType, ptrType},
                                  /*isVarArg=*/false));
}

bool hasSundialsIdaJacTimesCallbackABI(LLVM::LLVMFuncOp func, Type ptrType,
                                       Type i32Type, Type f64Type) {
  LLVM::LLVMFunctionType type = func.getFunctionType();
  if (type.getReturnType() != i32Type)
    return false;

  ArrayRef<Type> params = type.getParams();
  if (params.size() != 10)
    return false;

  return params[0] == f64Type && params[1] == ptrType &&
         params[2] == ptrType && params[3] == ptrType &&
         params[4] == ptrType && params[5] == ptrType &&
         params[6] == f64Type && params[7] == ptrType &&
         params[8] == ptrType && params[9] == ptrType;
}

bool hasSundialsIdaRawJvpKernelABI(LLVM::LLVMFuncOp func, Type ptrType,
                                   Type i32Type, Type f64Type) {
  return hasSundialsIdaJacTimesCallbackABI(func, ptrType, i32Type, f64Type);
}

LLVM::LLVMFunctionType getSundialsIdaJacTimesCallbackType(MLIRContext *context) {
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i32Type = IntegerType::get(context, 32);
  Type f64Type = Float64Type::get(context);
  return LLVM::LLVMFunctionType::get(
      i32Type,
      {f64Type, ptrType, ptrType, ptrType, ptrType, ptrType, f64Type, ptrType,
       ptrType, ptrType},
      /*isVarArg=*/false);
}

LLVM::LLVMFunctionType getSundialsIdaRawJvpKernelType(MLIRContext *context) {
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i32Type = IntegerType::get(context, 32);
  Type f64Type = Float64Type::get(context);
  return LLVM::LLVMFunctionType::get(
      i32Type,
      {f64Type, ptrType, ptrType, ptrType, ptrType, ptrType, f64Type, ptrType,
       ptrType, ptrType},
      /*isVarArg=*/false);
}

std::optional<std::string> getRuntimeJvpKernelSymbol(Operation *solve) {
  if (std::optional<std::string> runtimeKernel =
          getRootSymbolString(solve->getAttr(kSundialsRuntimeJvpKernelAttr)))
    return runtimeKernel;
  return getRootSymbolString(solve->getAttr("jacobian_action"));
}

LLVM::LLVMFuncOp getSundialsIdaJacTimesActionCallee(ModuleOp module,
                                                    Operation *solve,
                                                    Type ptrType, Type i32Type,
                                                    Type f64Type) {
  std::optional<std::string> actionName = getRuntimeJvpKernelSymbol(solve);
  if (!actionName)
    return nullptr;

  auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>(*actionName);
  if (!callee ||
      !hasSundialsIdaJacTimesCallbackABI(callee, ptrType, i32Type, f64Type))
    return nullptr;

  return callee;
}

Operation *getSemanticJacobianActionRecord(ModuleOp module, Operation *solve) {
  std::optional<std::string> actionName =
      getRootSymbolString(solve->getAttr("jacobian_action"));
  if (!actionName)
    return nullptr;

  Operation *record = nullptr;
  module.walk([&](Operation *op) {
    if (record || !isJacobianActionRecord(op))
      return;
    auto symName =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (symName && symName.getValue() == *actionName)
      record = op;
  });
  return record;
}

bool symbolRefRootEquals(Attribute attr, StringRef expected) {
  auto symbol = dyn_cast_or_null<SymbolRefAttr>(attr);
  return symbol && symbol.getRootReference() == expected;
}

FailureOr<std::optional<std::string>>
getLoweredRawJvpKernelSymbol(ModuleOp module, Operation *solve,
                             Operation *actionRecord) {
  if (std::optional<std::string> solveKernel =
          getRootSymbolString(solve->getAttr(kSundialsLoweredRawJvpKernelAttr)))
    return solveKernel;
  if (actionRecord)
    if (std::optional<std::string> actionKernel = getRootSymbolString(
            actionRecord->getAttr(kSundialsLoweredRawJvpKernelAttr)))
      return actionKernel;

  if (!actionRecord)
    return std::optional<std::string>();

  auto actionName =
      actionRecord->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
  if (!actionName)
    return std::optional<std::string>();

  SmallVector<std::string, 2> matches;
  module.walk([&](LLVM::LLVMFuncOp func) {
    auto role = func->getAttrOfType<StringAttr>(kSundialsRuntimeRoleAttr);
    if (!role || role.getValue() != "lowered_raw_jvp_kernel")
      return;
    if (!symbolRefRootEquals(
            func->getAttr("enzymexla.sundials.jacobian_action"),
            actionName.getValue()))
      return;
    matches.push_back(func.getName().str());
  });

  if (matches.size() > 1) {
    InFlightDiagnostic diag = solve->emitError()
                              << "multiple lowered raw IDA JVP kernels "
                                 "advertise jacobian action @"
                              << actionName.getValue() << ":";
    for (StringRef match : matches)
      diag << " @" << match;
    return failure();
  }

  if (matches.empty())
    return std::optional<std::string>();

  return std::optional<std::string>(matches.front());
}

void copyAttrIfPresent(Operation *source, Operation *dest, StringRef sourceName,
                       StringRef destName) {
  if (Attribute attr = source->getAttr(sourceName))
    dest->setAttr(destName, attr);
}

void copyJacobianActionProvenance(Operation *actionRecord, Operation *dest) {
  copyAttrIfPresent(actionRecord, dest, "materialization",
                    "enzymexla.sundials.materialization");
  copyAttrIfPresent(actionRecord, dest, "y_materialization",
                    "enzymexla.sundials.y_materialization");
  copyAttrIfPresent(actionRecord, dest, "yp_materialization",
                    "enzymexla.sundials.yp_materialization");
  copyAttrIfPresent(actionRecord, dest, "active_input_index",
                    "enzymexla.sundials.y_active_input_index");
  copyAttrIfPresent(actionRecord, dest, "yp_active_input_index",
                    "enzymexla.sundials.yp_active_input_index");
  copyAttrIfPresent(actionRecord, dest, "active_output_index",
                    "enzymexla.sundials.active_output_index");
}

std::optional<int64_t> getI64AttrValue(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}

struct FwddiffMaterializerCall {
  std::string callee;
  int64_t activeInputIndex = -1;
};

bool isSupportedFwddiffActivityLayout(ArrayRef<std::string> activities,
                                      int64_t inputCount,
                                      int64_t outputCount,
                                      int64_t activeInputIndex) {
  if (inputCount < 0 || outputCount != 1 || activeInputIndex < 0 ||
      activeInputIndex >= inputCount)
    return false;
  if (static_cast<int64_t>(activities.size()) != inputCount + outputCount)
    return false;

  for (int64_t index = 0; index < inputCount; ++index) {
    StringRef activity = activities[index];
    if (index == activeInputIndex) {
      if (activity != "enzyme_dup")
        return false;
      continue;
    }
    if (activity != "enzyme_const")
      return false;
  }

  return activities[inputCount] == "enzyme_dupnoneed";
}

std::optional<FwddiffMaterializerCall>
getSingleFwddiffMaterializerCall(ModuleOp module, StringRef materializerName,
                                 int64_t inputCount, int64_t outputCount,
                                 int64_t activeInputIndex) {
  auto materializer = module.lookupSymbol<LLVM::LLVMFuncOp>(materializerName);
  if (!materializer)
    return std::nullopt;

  SmallVector<LLVM::CallOp, 2> fwddiffCalls;
  materializer.walk([&](LLVM::CallOp call) {
    auto callee = call.getCallee();
    if (callee && isEnzymeFwddiffCallee(*callee))
      fwddiffCalls.push_back(call);
  });
  if (fwddiffCalls.size() != 1)
    return std::nullopt;

  LLVM::CallOp call = fwddiffCalls.front();
  SmallVector<std::string, 8> activities = getFwddiffActivitySymbols(call);
  if (!isSupportedFwddiffActivityLayout(activities, inputCount, outputCount,
                                        activeInputIndex))
    return std::nullopt;

  auto callee = call.getCallee();
  if (!callee)
    return std::nullopt;
  return FwddiffMaterializerCall{callee->str(), activeInputIndex};
}

struct RawJvpFwddiffPlan {
  FwddiffMaterializerCall yCall;
  std::optional<FwddiffMaterializerCall> ypCall;
  int64_t inputCount = -1;
  int64_t outputCount = -1;
  std::string residual;
};

std::optional<RawJvpFwddiffPlan>
getRawJvpFwddiffPlan(ModuleOp module, Operation *actionRecord) {
  std::optional<int64_t> inputCount =
      getI64AttrValue(actionRecord, "input_count");
  std::optional<int64_t> outputCount =
      getI64AttrValue(actionRecord, "output_count");
  std::optional<int64_t> yActiveInput =
      getI64AttrValue(actionRecord, "active_input_index");
  std::optional<std::string> residual =
      getRootSymbolString(actionRecord->getAttr("residual"));
  if (!inputCount || !outputCount || !yActiveInput || !residual)
    return std::nullopt;

  std::optional<std::string> yMaterializer =
      getRootSymbolString(actionRecord->getAttr("y_materialization"));
  if (!yMaterializer)
    yMaterializer =
        getRootSymbolString(actionRecord->getAttr("materialization"));
  if (!yMaterializer)
    return std::nullopt;

  std::optional<FwddiffMaterializerCall> yCall =
      getSingleFwddiffMaterializerCall(module, *yMaterializer, *inputCount,
                                       *outputCount, *yActiveInput);
  if (!yCall)
    return std::nullopt;

  RawJvpFwddiffPlan plan{*yCall, std::nullopt, *inputCount, *outputCount,
                         *residual};

  std::optional<std::string> ypMaterializer =
      getRootSymbolString(actionRecord->getAttr("yp_materialization"));
  std::optional<int64_t> ypActiveInput =
      getI64AttrValue(actionRecord, "yp_active_input_index");
  if (ypMaterializer || ypActiveInput) {
    if (!ypMaterializer || !ypActiveInput)
      return std::nullopt;
    std::optional<FwddiffMaterializerCall> ypCall =
        getSingleFwddiffMaterializerCall(module, *ypMaterializer, *inputCount,
                                         *outputCount, *ypActiveInput);
    if (!ypCall)
      return std::nullopt;
    plan.ypCall = *ypCall;
  }

  return plan;
}

Value createI64Constant(OpBuilder &builder, Location loc, int64_t value) {
  Type i64Type = IntegerType::get(builder.getContext(), 64);
  return LLVM::ConstantOp::create(builder, loc, i64Type,
                                  builder.getI64IntegerAttr(value));
}

Value createI32Zero(OpBuilder &builder, Location loc) {
  Type i32Type = IntegerType::get(builder.getContext(), 32);
  return LLVM::ZeroOp::create(builder, loc, i32Type);
}

Value loadEnzymeActivityMarker(OpBuilder &builder, Location loc,
                               StringRef markerName) {
  MLIRContext *context = builder.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i32Type = IntegerType::get(context, 32);
  Value markerAddress =
      LLVM::AddressOfOp::create(builder, loc, ptrType, markerName);
  return LLVM::LoadOp::create(builder, loc, i32Type, markerAddress);
}

Value emitContextInputLoad(ModuleOp module, OpBuilder &builder, Location loc,
                           Value userData, int64_t inputIndex) {
  MLIRContext *context = module.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Value index = createI64Constant(builder, loc, inputIndex);
  auto call = LLVM::CallOp::create(
      builder, loc, TypeRange{ptrType},
      SymbolRefAttr::get(context, "__enzymexla_sundials_ida_context_input"),
      ValueRange{userData, index});
  call->setAttr(kSundialsRoleAttr,
                builder.getStringAttr("ida_raw_jvp_context_input"));
  call->setAttr("enzymexla.sundials.input_index",
                builder.getI64IntegerAttr(inputIndex));
  return call.getResult();
}

std::optional<Value> getRawPrimalInput(ModuleOp module, OpBuilder &builder,
                                       Location loc, int64_t inputIndex,
                                       Value yyData, Value ypData,
                                       Value userData) {
  if (inputIndex == 1)
    return yyData;
  if (inputIndex == 2)
    return ypData;
  return emitContextInputLoad(module, builder, loc, userData, inputIndex);
}

std::optional<Value> getRawTangentInput(int64_t inputIndex, Value vData,
                                        Value ypTangentData) {
  if (inputIndex == 1)
    return vData;
  if (inputIndex == 2)
    return ypTangentData;
  return std::nullopt;
}

LogicalResult emitFwddiffRawBufferCall(ModuleOp module, OpBuilder &builder,
                                       Location loc,
                                       const FwddiffMaterializerCall &callSpec,
                                       const RawJvpFwddiffPlan &plan,
                                       Value yyData, Value ypData, Value rrData,
                                       Value vData, Value outputTangentData,
                                       Value userData, Value ypTangentData) {
  MLIRContext *context = module.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  SmallVector<Value, 16> operands;
  Value residual = LLVM::AddressOfOp::create(builder, loc, ptrType,
                                             plan.residual);
  operands.push_back(residual);

  for (int64_t inputIndex = 0; inputIndex < plan.inputCount; ++inputIndex) {
    std::optional<Value> primal = getRawPrimalInput(
        module, builder, loc, inputIndex, yyData, ypData, userData);
    if (!primal)
      return failure();

    if (inputIndex == callSpec.activeInputIndex) {
      std::optional<Value> tangent =
          getRawTangentInput(inputIndex, vData, ypTangentData);
      if (!tangent)
        return failure();
      operands.push_back(loadEnzymeActivityMarker(builder, loc, "enzyme_dup"));
      operands.push_back(*primal);
      operands.push_back(*tangent);
    } else {
      operands.push_back(
          loadEnzymeActivityMarker(builder, loc, "enzyme_const"));
      operands.push_back(*primal);
    }
  }

  operands.push_back(
      loadEnzymeActivityMarker(builder, loc, "enzyme_dupnoneed"));
  operands.push_back(rrData);
  operands.push_back(outputTangentData);

  auto fwddiffCall = LLVM::CallOp::create(
      builder, loc, TypeRange{}, SymbolRefAttr::get(context, callSpec.callee),
      operands);
  fwddiffCall->setAttr(kSundialsRoleAttr,
                       builder.getStringAttr("ida_raw_jvp_fwddiff"));
  fwddiffCall->setAttr("enzymexla.sundials.active_input_index",
                       builder.getI64IntegerAttr(callSpec.activeInputIndex));
  return success();
}

LLVM::LLVMFuncOp emitSundialsIdaFwddiffRawJvpKernel(
    ModuleOp module, OpBuilder &builder, Location loc, StringRef rawKernelName,
    Operation *solve, Operation *actionRecord,
    const RawJvpFwddiffPlan &plan) {
  MLIRContext *context = module.getContext();
  Type i32Type = IntegerType::get(context, 32);

  ensureSundialsIdaRawJvpContextDeclarations(module, builder, loc);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  LLVM::LLVMFuncOp rawKernel = LLVM::LLVMFuncOp::create(
      builder, loc, rawKernelName, getSundialsIdaRawJvpKernelType(context));
  rawKernel->setAttr(kSundialsRuntimeRoleAttr,
                     builder.getStringAttr("ida_raw_jvp_kernel"));
  rawKernel->setAttr("enzymexla.sundials.jvp_kernel_body",
                     builder.getStringAttr("enzyme_fwddiff_raw_buffer_calls"));
  rawKernel->setAttr("enzymexla.sundials.callback_context",
                     builder.getStringAttr("context_input_accessor"));
  if (plan.ypCall)
    rawKernel->setAttr("enzymexla.sundials.raw_jvp_accumulation",
                       builder.getStringAttr("context_accumulate"));
  if (Attribute jacobianAction = solve->getAttr("jacobian_action"))
    rawKernel->setAttr("enzymexla.sundials.jacobian_action", jacobianAction);
  if (Attribute residual = solve->getAttr("residual"))
    rawKernel->setAttr("enzymexla.sundials.residual", residual);
  copyJacobianActionProvenance(actionRecord, rawKernel);

  Block *entry = rawKernel.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  Value yyData = entry->getArgument(1);
  Value ypData = entry->getArgument(2);
  Value rrData = entry->getArgument(3);
  Value vData = entry->getArgument(4);
  Value jvData = entry->getArgument(5);
  Value userData = entry->getArgument(7);
  Value ypTangentData = entry->getArgument(8);
  Value tmp2Data = entry->getArgument(9);

  if (failed(emitFwddiffRawBufferCall(module, builder, loc, plan.yCall, plan,
                                      yyData, ypData, rrData, vData, jvData,
                                      userData, ypTangentData))) {
    rawKernel.erase();
    return nullptr;
  }

  if (plan.ypCall) {
    if (failed(emitFwddiffRawBufferCall(
            module, builder, loc, *plan.ypCall, plan, yyData, ypData, rrData,
            vData, tmp2Data, userData, ypTangentData))) {
      rawKernel.erase();
      return nullptr;
    }

    auto accumulate = LLVM::CallOp::create(
        builder, loc, TypeRange{i32Type},
        SymbolRefAttr::get(context,
                           "__enzymexla_sundials_ida_accumulate_raw_jvp"),
        ValueRange{userData, jvData, tmp2Data});
    accumulate->setAttr(kSundialsRoleAttr,
                        builder.getStringAttr("ida_raw_jvp_accumulate"));
    LLVM::ReturnOp::create(builder, loc, ValueRange{accumulate.getResult()});
    return rawKernel;
  }

  LLVM::ReturnOp::create(builder, loc, ValueRange{createI32Zero(builder, loc)});
  return rawKernel;
}

LLVM::LLVMFuncOp emitSundialsIdaPlaceholderRawJvpKernel(
    ModuleOp module, OpBuilder &builder, Location loc, StringRef rawKernelName,
    Operation *solve, Operation *actionRecord) {
  MLIRContext *context = module.getContext();
  Type i32Type = IntegerType::get(context, 32);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  LLVM::LLVMFuncOp rawKernel = LLVM::LLVMFuncOp::create(
      builder, loc, rawKernelName, getSundialsIdaRawJvpKernelType(context));
  rawKernel->setAttr(kSundialsRuntimeRoleAttr,
                     builder.getStringAttr("ida_raw_jvp_kernel"));
  rawKernel->setAttr("enzymexla.sundials.jvp_kernel_body",
                     builder.getStringAttr("semantic_raw_kernel_requires_lowering"));
  if (Attribute jacobianAction = solve->getAttr("jacobian_action"))
    rawKernel->setAttr("enzymexla.sundials.jacobian_action", jacobianAction);
  if (Attribute residual = solve->getAttr("residual"))
    rawKernel->setAttr("enzymexla.sundials.residual", residual);
  copyJacobianActionProvenance(actionRecord, rawKernel);

  Block *entry = rawKernel.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  Value notExecutable = LLVM::ConstantOp::create(
      builder, loc, i32Type, builder.getIntegerAttr(i32Type, 1));
  LLVM::ReturnOp::create(builder, loc, ValueRange{notExecutable});
  return rawKernel;
}

LLVM::LLVMFuncOp emitSundialsIdaRawJvpKernel(ModuleOp module,
                                             OpBuilder &builder, Location loc,
                                             StringRef rawKernelName,
                                             Operation *solve,
                                             Operation *actionRecord) {
  if (actionRecord) {
    if (std::optional<RawJvpFwddiffPlan> plan =
            getRawJvpFwddiffPlan(module, actionRecord)) {
      if (LLVM::LLVMFuncOp rawKernel = emitSundialsIdaFwddiffRawJvpKernel(
              module, builder, loc, rawKernelName, solve, actionRecord, *plan))
        return rawKernel;
    }
  }

  return emitSundialsIdaPlaceholderRawJvpKernel(
      module, builder, loc, rawKernelName, solve, actionRecord);
}

LLVM::LLVMFuncOp emitSundialsIdaJvpKernelAdapter(ModuleOp module,
                                                 OpBuilder &builder,
                                                 Location loc,
                                                 StringRef adapterName,
                                                 StringRef rawKernelName,
                                                 Operation *solve,
                                                 Operation *actionRecord) {
  MLIRContext *context = module.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i32Type = IntegerType::get(context, 32);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  LLVM::LLVMFuncOp adapter = LLVM::LLVMFuncOp::create(
      builder, loc, adapterName, getSundialsIdaJacTimesCallbackType(context));
  adapter->setAttr(kSundialsRuntimeRoleAttr,
                   builder.getStringAttr("ida_jvp_kernel_adapter"));
  adapter->setAttr("enzymexla.sundials.jvp_kernel_body",
                   builder.getStringAttr("nvector_unpack_and_raw_jvp_call"));
  adapter->setAttr("enzymexla.sundials.yp_tangent",
                   builder.getStringAttr("tmp1 = cj * v"));
  adapter->setAttr("enzymexla.sundials.raw_jvp_kernel",
                   SymbolRefAttr::get(context, rawKernelName));
  if (Attribute jacobianAction = solve->getAttr("jacobian_action"))
    adapter->setAttr("enzymexla.sundials.jacobian_action", jacobianAction);
  if (Attribute residual = solve->getAttr("residual"))
    adapter->setAttr("enzymexla.sundials.residual", residual);
  copyJacobianActionProvenance(actionRecord, adapter);

  Block *entry = adapter.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  Value tt = entry->getArgument(0);
  Value yy = entry->getArgument(1);
  Value yp = entry->getArgument(2);
  Value rr = entry->getArgument(3);
  Value v = entry->getArgument(4);
  Value jv = entry->getArgument(5);
  Value cj = entry->getArgument(6);
  Value userData = entry->getArgument(7);
  Value tmp1 = entry->getArgument(8);
  Value tmp2 = entry->getArgument(9);

  auto ypTangentScale = LLVM::CallOp::create(
      builder, loc, TypeRange{}, SymbolRefAttr::get(context, "N_VScale"),
      ValueRange{cj, v, tmp1});
  ypTangentScale->setAttr(kSundialsRoleAttr,
                          builder.getStringAttr("ida_yp_tangent_scale"));

  auto getArrayPointer = [&](Value nvector) -> Value {
    return LLVM::CallOp::create(
               builder, loc, TypeRange{ptrType},
               SymbolRefAttr::get(context, "N_VGetArrayPointer"),
               ValueRange{nvector})
        .getResult();
  };

  Value yyData = getArrayPointer(yy);
  Value ypData = getArrayPointer(yp);
  Value rrData = getArrayPointer(rr);
  Value vData = getArrayPointer(v);
  Value jvData = getArrayPointer(jv);
  Value tmp1Data = getArrayPointer(tmp1);
  Value tmp2Data = getArrayPointer(tmp2);

  auto status = LLVM::CallOp::create(
      builder, loc, TypeRange{i32Type}, SymbolRefAttr::get(context, rawKernelName),
      ValueRange{tt, yyData, ypData, rrData, vData, jvData, cj, userData,
                 tmp1Data, tmp2Data});
  status->setAttr(kSundialsRoleAttr,
                  builder.getStringAttr("ida_raw_jvp_kernel"));
  LLVM::ReturnOp::create(builder, loc, ValueRange{status.getResult()});
  return adapter;
}

LLVM::LLVMFuncOp emitSundialsIdaJacTimesCallback(ModuleOp module,
                                                 OpBuilder &builder,
                                                 Location loc,
                                                 StringRef callbackName,
                                                 Operation *solve) {
  MLIRContext *context = module.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i32Type = IntegerType::get(context, 32);
  Type f64Type = Float64Type::get(context);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  LLVM::LLVMFuncOp callback =
      LLVM::LLVMFuncOp::create(builder, loc, callbackName,
                               getSundialsIdaJacTimesCallbackType(context));
  callback->setAttr(kSundialsRuntimeRoleAttr,
                    builder.getStringAttr("ida_jactimes_callback"));
  if (Attribute jacobianAction = solve->getAttr("jacobian_action"))
    callback->setAttr("enzymexla.sundials.jacobian_action", jacobianAction);
  if (Attribute residual = solve->getAttr("residual"))
    callback->setAttr("enzymexla.sundials.residual", residual);
  if (Attribute sourceFunction = solve->getAttr("source_function")) {
    callback->setAttr("enzymexla.sundials.source_function", sourceFunction);
  }

  LLVM::LLVMFuncOp actionCallee =
      getSundialsIdaJacTimesActionCallee(module, solve, ptrType, i32Type,
                                         f64Type);
  if (actionCallee) {
    callback->setAttr("enzymexla.sundials.callback_body",
                      builder.getStringAttr("delegates_jvp_kernel"));
    callback->setAttr("enzymexla.sundials.jvp_kernel",
                      SymbolRefAttr::get(context, actionCallee.getName()));
  } else {
    callback->setAttr("enzymexla.sundials.callback_body",
                      builder.getStringAttr("placeholder"));
  }

  Block *entry = callback.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  if (actionCallee) {
    auto status = LLVM::CallOp::create(
        builder, loc, TypeRange{i32Type},
        SymbolRefAttr::get(context, actionCallee.getName()),
        ValueRange(entry->getArguments()));
    status->setAttr(kSundialsRoleAttr,
                    builder.getStringAttr("ida_jacobian_action_jvp_kernel"));
    LLVM::ReturnOp::create(builder, loc, ValueRange{status.getResult()});
  } else {
    Value ok = LLVM::ZeroOp::create(builder, loc, i32Type);
    LLVM::ReturnOp::create(builder, loc, ValueRange{ok});
  }
  return callback;
}

LLVM::LLVMFuncOp emitSundialsIdaJacTimesRegistration(ModuleOp module,
                                                     OpBuilder &builder,
                                                     Location loc,
                                                     StringRef registrationName,
                                                     StringRef callbackName,
                                                     Operation *solve) {
  MLIRContext *context = module.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i32Type = IntegerType::get(context, 32);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto registrationType = LLVM::LLVMFunctionType::get(
      i32Type, {ptrType, ptrType, ptrType, ptrType}, /*isVarArg=*/false);
  LLVM::LLVMFuncOp registration =
      LLVM::LLVMFuncOp::create(builder, loc, registrationName, registrationType);
  registration->setAttr(kSundialsRuntimeRoleAttr,
                        builder.getStringAttr("ida_jactimes_registration"));
  registration->setAttr("enzymexla.sundials.callback_context",
                        builder.getStringAttr("ida_jvp_user_data_context"));
  registration->setAttr("enzymexla.sundials.jactimes_callback",
                        SymbolRefAttr::get(context, callbackName));
  registration->setAttr("enzymexla.sundials.linear_solver",
                        builder.getStringAttr("SUNLinSol_SPGMR"));
  if (Attribute jacobianAction = solve->getAttr("jacobian_action"))
    registration->setAttr("enzymexla.sundials.jacobian_action",
                          jacobianAction);
  if (Attribute sourceFunction = solve->getAttr("source_function")) {
    registration->setAttr("enzymexla.sundials.source_function",
                          sourceFunction);
  }

  Block *entry = registration.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  Value idaMem = entry->getArgument(0);
  Value yyTemplate = entry->getArgument(1);
  Value sunctx = entry->getArgument(2);
  Value userData = entry->getArgument(3);

  Value pretypeNone = LLVM::ConstantOp::create(
      builder, loc, i32Type, builder.getIntegerAttr(i32Type, 0));
  Value defaultKrylovDim = LLVM::ConstantOp::create(
      builder, loc, i32Type, builder.getIntegerAttr(i32Type, 0));
  Value nullPtr = LLVM::ZeroOp::create(builder, loc, ptrType);

  auto registerContext = LLVM::CallOp::create(
      builder, loc, TypeRange{},
      SymbolRefAttr::get(context,
                         "__enzymexla_sundials_ida_register_jvp_context"),
      ValueRange{userData});
  setSundialsCallRole(registerContext, builder,
                      "ida_jvp_context_registration");

  auto setUserData = LLVM::CallOp::create(
      builder, loc, TypeRange{i32Type},
      SymbolRefAttr::get(context, "IDASetUserData"),
      ValueRange{idaMem, userData});
  setSundialsCallRole(setUserData, builder, "ida_user_data_registration");

  auto linearSolver = LLVM::CallOp::create(
      builder, loc, TypeRange{ptrType}, SymbolRefAttr::get(context,
                                                           "SUNLinSol_SPGMR"),
      ValueRange{yyTemplate, pretypeNone, defaultKrylovDim, sunctx});
  setSundialsCallRole(linearSolver, builder, "ida_iterative_linear_solver");

  auto setLinearSolver = LLVM::CallOp::create(
      builder, loc, TypeRange{i32Type},
      SymbolRefAttr::get(context, "IDASetLinearSolver"),
      ValueRange{idaMem, linearSolver.getResult(), nullPtr});
  setSundialsCallRole(setLinearSolver, builder,
                      "ida_linear_solver_registration");

  Value callback = LLVM::AddressOfOp::create(builder, loc, ptrType,
                                             callbackName);
  auto setJacTimes = LLVM::CallOp::create(
      builder, loc, TypeRange{i32Type},
      SymbolRefAttr::get(context, "IDASetJacTimes"),
      ValueRange{idaMem, nullPtr, callback});
  setSundialsCallRole(setJacTimes, builder,
                      "ida_jacobian_action_registration");

  LLVM::ReturnOp::create(builder, loc, ValueRange{setJacTimes.getResult()});
  return registration;
}

struct EmitSundialsIdaRuntimeGlueLLVM
    : public mlir::enzyme::impl::EmitSundialsIdaRuntimeGlueLLVMBase<
          EmitSundialsIdaRuntimeGlueLLVM> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    Builder attrBuilder(context);
    OpBuilder builder(context);
    llvm::StringSet<> usedSymbols;
    SmallVector<enzymexla::SundialsIdaSolveOp, 8> targetSolves;

    module.walk([&](Operation *op) {
      if (auto symName = op->getAttrOfType<StringAttr>(
              SymbolTable::getSymbolAttrName()))
        usedSymbols.insert(symName.getValue());
    });

    module.walk([&](enzymexla::SundialsIdaSolveOp solve) {
      if (solve.getJacobianDemand() !=
          enzymexla::SundialsIdaJacobianDemand::jacobian_action)
        return;
      if (solve.getLinearSolver() !=
          enzymexla::SundialsIdaLinearSolver::jacobian_action_iterative)
        return;
      if (!solve.getJacobianAction())
        return;
      if (solve->hasAttr(kSundialsRuntimeJacTimesCallbackAttr) &&
          solve->hasAttr(kSundialsRuntimeRegistrationAttr))
        return;
      targetSolves.push_back(solve);
    });

    if (targetSolves.empty())
      return;

    ensureSundialsIdaRuntimeDeclarations(module, builder, module.getLoc());

    unsigned emitted = 0;
    unsigned emittedJvpKernelAdapters = 0;
    unsigned emittedRawJvpKernels = 0;
    unsigned linkedLoweredRawJvpKernels = 0;
    for (enzymexla::SundialsIdaSolveOp solve : targetSolves) {
      std::string callbackName = getUniqueSundialsRuntimeSymbol(
          usedSymbols, "__enzymexla_sundials_ida_jactimes_");
      std::string registrationName = getUniqueSundialsRuntimeSymbol(
          usedSymbols, "__enzymexla_sundials_ida_register_jactimes_");

      if (!getSundialsIdaJacTimesActionCallee(
              module, solve.getOperation(), LLVM::LLVMPointerType::get(context),
              IntegerType::get(context, 32), Float64Type::get(context))) {
        if (Operation *actionRecord =
                getSemanticJacobianActionRecord(module, solve.getOperation())) {
          std::string adapterName = getUniqueSundialsRuntimeSymbol(
              usedSymbols, "__enzymexla_sundials_ida_jvp_kernel_");
          std::string rawKernelName;
          FailureOr<std::optional<std::string>> loweredRawKernelName =
              getLoweredRawJvpKernelSymbol(module, solve.getOperation(),
                                           actionRecord);
          if (failed(loweredRawKernelName)) {
            signalPassFailure();
            return;
          }

          if (*loweredRawKernelName) {
            auto rawKernel =
                module.lookupSymbol<LLVM::LLVMFuncOp>(**loweredRawKernelName);
            if (!rawKernel ||
                !hasSundialsIdaRawJvpKernelABI(
                    rawKernel, LLVM::LLVMPointerType::get(context),
                    IntegerType::get(context, 32), Float64Type::get(context))) {
              solve.emitError()
                  << "lowered raw IDA JVP kernel @" << **loweredRawKernelName
                  << " is missing or does not match the raw callback ABI";
              signalPassFailure();
              return;
            }
            rawKernelName = **loweredRawKernelName;
            ++linkedLoweredRawJvpKernels;
          } else {
            rawKernelName = getUniqueSundialsRuntimeSymbol(
                usedSymbols, "__enzymexla_sundials_ida_raw_jvp_kernel_");
            emitSundialsIdaRawJvpKernel(module, builder, solve.getLoc(),
                                        rawKernelName, solve.getOperation(),
                                        actionRecord);
            ++emittedRawJvpKernels;
          }
          emitSundialsIdaJvpKernelAdapter(
              module, builder, solve.getLoc(), adapterName, rawKernelName,
              solve.getOperation(), actionRecord);
          solve->setAttr(kSundialsRuntimeJvpKernelAttr,
                         SymbolRefAttr::get(context, adapterName));
          solve->setAttr(kSundialsRuntimeRawJvpKernelAttr,
                         SymbolRefAttr::get(context, rawKernelName));
          ++emittedJvpKernelAdapters;
        }
      }

      emitSundialsIdaJacTimesCallback(module, builder, solve.getLoc(),
                                      callbackName, solve.getOperation());
      emitSundialsIdaJacTimesRegistration(module, builder, solve.getLoc(),
                                          registrationName, callbackName,
                                          solve.getOperation());

      solve->setAttr(kSundialsRuntimeJacTimesCallbackAttr,
                     SymbolRefAttr::get(context, callbackName));
      solve->setAttr(kSundialsRuntimeRegistrationAttr,
                     SymbolRefAttr::get(context, registrationName));
      ++emitted;
    }

    module->setAttr(kSundialsIdaRuntimeGlueEmittedAttr,
                    attrBuilder.getI64IntegerAttr(emitted));
    if (emittedJvpKernelAdapters != 0)
      module->setAttr(kSundialsIdaJvpKernelAdaptersEmittedAttr,
                      attrBuilder.getI64IntegerAttr(emittedJvpKernelAdapters));
    if (emittedRawJvpKernels != 0)
      module->setAttr(kSundialsIdaRawJvpKernelsEmittedAttr,
                      attrBuilder.getI64IntegerAttr(emittedRawJvpKernels));
    if (linkedLoweredRawJvpKernels != 0)
      module->setAttr(
          kSundialsIdaLoweredRawJvpKernelsLinkedAttr,
          attrBuilder.getI64IntegerAttr(linkedLoweredRawJvpKernels));
  }
};

struct LowerSundialsIdaJacobianActionStableHLO
    : public mlir::enzyme::impl::LowerSundialsIdaJacobianActionStableHLOBase<
          LowerSundialsIdaJacobianActionStableHLO> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTableCollection symbolTable;
    llvm::SmallPtrSet<Operation *, 8> targetActions;

    module.walk([&](enzymexla::SundialsIdaSolveOp solve) {
      if (solve.getJacobianDemand() !=
          enzymexla::SundialsIdaJacobianDemand::jacobian_action)
        return;

      if (solve.getLinearSolver() !=
          enzymexla::SundialsIdaLinearSolver::jacobian_action_iterative)
        return;

      auto jacobianAction = solve.getJacobianAction();
      if (!jacobianAction)
        return;

      auto action = dyn_cast_or_null<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(solve, *jacobianAction));
      if (!action)
        return;

      targetActions.insert(action.getOperation());
    });

    for (Operation *action : targetActions) {
      if (failed(lowerJacobianActionsIn(action, /*requireOnlyJVP=*/true,
                                        /*rejectLiveJacobians=*/true))) {
        signalPassFailure();
        return;
      }
    }
  };
};

struct MarkGridKitSparseJacobianLLVM
    : public mlir::enzyme::impl::MarkGridKitSparseJacobianLLVMBase<
          MarkGridKitSparseJacobianLLVM> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    Builder builder(context);
    unsigned markedHelpers = 0;
    llvm::StringSet<> existingSemanticRecords;
    SmallVector<RecoveredJacobianMaterializationRecord, 8> recoveredJacobians;

    module.walk([&](Operation *op) {
      if (!isJacobianMaterializationRecord(op))
        return;
      auto materializer = op->getAttrOfType<SymbolRefAttr>("materializer");
      if (!materializer)
        return;
      existingSemanticRecords.insert(materializer.getRootReference());
    });

    module.walk([&](LLVM::LLVMFuncOp func) {
      StringRef source = classifyGridKitSparseJacobianHelper(func.getName());
      if (source.empty())
        return;

      GridKitSparseJacobianMatch match =
          collectGridKitSparseJacobianMatch(func);
      if (!match.isComplete())
        return;

      func->setAttr(kGridKitJacobianMaterializationAttr,
                    builder.getStringAttr(kGridKitSparseOneHotMaterialization));
      func->setAttr(kGridKitJacobianSourceAttr, builder.getStringAttr(source));
      func->setAttr(kGridKitSolverAttr,
                    builder.getStringAttr(kGridKitIdaJacTimes));
      func->setAttr(kGridKitJacobianFwddiffCallsAttr,
                    builder.getI64IntegerAttr(match.fwddiffCalls.size()));
      func->setAttr(kGridKitJacobianTodenseCallsAttr,
                    builder.getI64IntegerAttr(match.todenseCalls.size()));
      func->setAttr(
          kGridKitJacobianSparseStoreAddressesAttr,
          builder.getI64IntegerAttr(match.sparseStoreAddresses.size()));

      for (LLVM::CallOp call : match.fwddiffCalls) {
        call->setAttr(kGridKitJacobianActionAttr,
                      builder.getStringAttr(kGridKitResidualJvpCandidate));
        call->setAttr(kGridKitJacobianSourceAttr,
                      builder.getStringAttr(source));
        call->setAttr(kGridKitSolverAttr,
                      builder.getStringAttr(kGridKitIdaJacTimes));
      }

      for (LLVM::CallOp call : match.todenseCalls)
        call->setAttr(kGridKitJacobianRoleAttr,
                      builder.getStringAttr(kGridKitSparseTodenseRole));

      recoveredJacobians.push_back(
          {func.getName().str(), match.residual, match.enzymeActivity,
           match.inputActivity, match.outputActivity, match.inputCount,
           match.outputCount, match.activeInputIndex, match.activeOutputIndex,
           match.outputDimensionArg, match.activeInputDimensionArg,
           match.seedLoopDimensionArg, match.outputIndexMapArg,
           match.activeInputIndexMapArg, match.sparseRowsArg,
           match.sparseColsArg, match.sparseValuesArg, match.sparseNnzArg,
           match.sparseAssembly, source.str(),
           static_cast<unsigned>(match.fwddiffCalls.size()),
           static_cast<unsigned>(match.todenseCalls.size()),
           static_cast<unsigned>(match.sparseStoreAddresses.size())});
      ++markedHelpers;
    });

    if (markedHelpers != 0)
      module->setAttr(kGridKitJacobianMarkedHelpersAttr,
                      builder.getI64IntegerAttr(markedHelpers));

    OpBuilder opBuilder(context);
    opBuilder.setInsertionPointToStart(module.getBody());
    for (const RecoveredJacobianMaterializationRecord &record :
         recoveredJacobians) {
      if (existingSemanticRecords.contains(record.materializer))
        continue;

      OperationState state(module.getLoc(), kJacobianMaterializationOpName);
      state.addAttribute("materializer",
                         SymbolRefAttr::get(context, record.materializer));
      if (record.residual)
        state.addAttribute("residual",
                           SymbolRefAttr::get(context, *record.residual));
      if (record.enzymeActivity)
        state.addAttribute("enzyme_activity",
                           getStringArrayAttr(builder, *record.enzymeActivity));
      if (record.inputActivity)
        state.addAttribute("input_activity",
                           getStringArrayAttr(builder, *record.inputActivity));
      if (record.outputActivity)
        state.addAttribute("output_activity",
                           getStringArrayAttr(builder, *record.outputActivity));
      if (record.inputCount)
        state.addAttribute("input_count",
                           builder.getI64IntegerAttr(*record.inputCount));
      if (record.outputCount)
        state.addAttribute("output_count",
                           builder.getI64IntegerAttr(*record.outputCount));
      if (record.activeInputIndex)
        state.addAttribute("active_input_index",
                           builder.getI64IntegerAttr(*record.activeInputIndex));
      if (record.activeOutputIndex)
        state.addAttribute(
            "active_output_index",
            builder.getI64IntegerAttr(*record.activeOutputIndex));
      if (record.outputDimensionArg)
        state.addAttribute(
            "output_dimension_arg",
            builder.getI64IntegerAttr(*record.outputDimensionArg));
      if (record.activeInputDimensionArg)
        state.addAttribute(
            "active_input_dimension_arg",
            builder.getI64IntegerAttr(*record.activeInputDimensionArg));
      if (record.seedLoopDimensionArg)
        state.addAttribute(
            "seed_loop_dimension_arg",
            builder.getI64IntegerAttr(*record.seedLoopDimensionArg));
      if (record.outputIndexMapArg)
        state.addAttribute(
            "output_index_map_arg",
            builder.getI64IntegerAttr(*record.outputIndexMapArg));
      if (record.activeInputIndexMapArg)
        state.addAttribute(
            "active_input_index_map_arg",
            builder.getI64IntegerAttr(*record.activeInputIndexMapArg));
      if (record.sparseRowsArg)
        state.addAttribute("sparse_rows_arg",
                           builder.getI64IntegerAttr(*record.sparseRowsArg));
      if (record.sparseColsArg)
        state.addAttribute("sparse_cols_arg",
                           builder.getI64IntegerAttr(*record.sparseColsArg));
      if (record.sparseValuesArg)
        state.addAttribute(
            "sparse_values_arg",
            builder.getI64IntegerAttr(*record.sparseValuesArg));
      if (record.sparseNnzArg)
        state.addAttribute("sparse_nnz_arg",
                           builder.getI64IntegerAttr(*record.sparseNnzArg));
      if (record.sparseAssembly)
        state.addAttribute("sparse_assembly",
                           builder.getStringAttr(*record.sparseAssembly));
      state.addAttribute(
          "method",
          enzymexla::JacobianMaterializationMethodAttr::get(
              context,
              enzymexla::JacobianMaterializationMethod::one_hot_forward));
      state.addAttribute(
          "storage", enzymexla::JacobianStorageAttr::get(
                         context, enzymexla::JacobianStorage::sparse_callback));
      state.addAttribute("fwddiff_calls",
                         builder.getI64IntegerAttr(record.fwddiffCalls));
      state.addAttribute("todense_calls",
                         builder.getI64IntegerAttr(record.todenseCalls));
      state.addAttribute(
          "sparse_store_callbacks",
          builder.getI64IntegerAttr(record.sparseStoreCallbacks));
      state.addAttribute("source", builder.getStringAttr(record.source));
      opBuilder.create(state);
      existingSemanticRecords.insert(record.materializer);
    }
  }
};
} // namespace
