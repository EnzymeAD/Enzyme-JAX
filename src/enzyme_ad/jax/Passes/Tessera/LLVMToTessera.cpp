//===----------------------------------------------------------------------===//
//
// Converts marked llvm.funcs to tessera.defines, and calls to them to
// tessera.calls.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_LLVMTOTESSERAPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

namespace {

// On the stub that stands in for a lifted function's address, naming the
// original function. See redirectAddressUses.
constexpr llvm::StringLiteral kAddressStubAttr = "tessera.address_stub_for";

// llvm.mlir.addressof can only name an LLVM symbol, so taking the address of
// a function (`&DenseMatrix::Invert`, or putting it in llvm.used) stops the
// module verifying once it becomes a tessera.define. Point those uses at a
// stub llvm.func declaration instead; tessera-to-llvm points them back. Calls
// through the pointer are indirect, so no rule could match them anyway.
void redirectAddressUses(LLVM::LLVMFuncOp funcOp, PatternRewriter &rewriter,
                         ModuleOp module) {
  auto uses = SymbolTable::getSymbolUses(funcOp.getSymNameAttr(), module);
  if (!uses)
    return;

  SmallVector<LLVM::AddressOfOp> addressUses;
  for (const SymbolTable::SymbolUse &use : *uses) {
    auto addressOf = dyn_cast<LLVM::AddressOfOp>(use.getUser());
    if (!addressOf)
      continue;
    // This pass erases the annotation table at the end.
    auto global = addressOf->getParentOfType<LLVM::GlobalOp>();
    if (global && global.getSymName() == "llvm.global.annotations")
      continue;
    addressUses.push_back(addressOf);
  }
  if (addressUses.empty())
    return;

  auto *ctx = funcOp->getContext();
  std::string stubName = (funcOp.getName() + ".tessera_address").str();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(funcOp);
  auto stub = LLVM::LLVMFuncOp::create(rewriter, funcOp.getLoc(), stubName,
                                       funcOp.getFunctionType());
  // A string, not a symbol reference, so the renaming below leaves it alone.
  stub->setAttr(kAddressStubAttr, funcOp.getSymNameAttr());

  auto stubRef = FlatSymbolRefAttr::get(ctx, stubName);
  for (LLVM::AddressOfOp addressOf : addressUses)
    rewriter.modifyOpInPlace(addressOf,
                             [&] { addressOf.setGlobalNameAttr(stubRef); });
}

// The tessera_op or pure_tessera_op marking on `funcOp`, if it has one.
StringAttr getMarking(LLVM::LLVMFuncOp funcOp, bool &isPure) {
  isPure = false;
  if (auto attr = funcOp->getAttrOfType<StringAttr>("tessera_op"))
    return attr;
  if (auto attr = funcOp->getAttrOfType<StringAttr>("pure_tessera_op")) {
    isPure = true;
    return attr;
  }
  return {};
}

// The tessera op name a marking gives its function: everything before '('.
StringRef getTesseraName(StringRef marking) {
  return marking.take_while([](char c) { return c != '('; });
}

// The argModes for the tessera.define `funcOp` becomes, from a marking like
// "name(arg1:val=in, arg2, ...)". Lifted types come from tessera.arg_types.
// Fails, with the reason in `why`, if the marking doesn't fit the function.
FailureOr<ArrayAttr> buildArgModes(LLVM::LLVMFuncOp funcOp, StringRef raw,
                                   std::string &why) {
  MLIRContext *ctx = funcOp->getContext();
  if (getTesseraName(raw).trim().empty()) {
    why = "it does not name a tessera op";
    return failure();
  }

  // Parse args in parentheses. A marking without them lists no arguments.
  StringRef argList = raw.contains('(')
                          ? raw.slice(raw.find('(') + 1, raw.find(')'))
                          : StringRef();
  SmallVector<StringRef> argParts;
  if (!argList.trim().empty())
    argList.split(argParts, ',');

  // One entry per argument, not counting an sret pointer.
  unsigned sretOffset =
      funcOp.getNumArguments() > 0 &&
              funcOp.getArgAttr(0, LLVM::LLVMDialect::getStructRetAttrName())
          ? 1
          : 0;
  unsigned numArgs = funcOp.getNumArguments() - sretOffset;
  if (argParts.size() != numArgs) {
    why = "its argument list names " + std::to_string(argParts.size()) +
          " argument(s), but the function takes " + std::to_string(numArgs);
    return failure();
  }

  auto argTypes = funcOp->getAttrOfType<ArrayAttr>(kArgTypesAttr);
  SmallVector<Attribute> argModes;
  for (unsigned i = 0, e = argParts.size(); i != e; ++i) {
    StringRef arg = argParts[i].trim();
    StringRef marker = arg.split(':').second.trim();
    if (marker.empty()) {
      // Unmarked: not lifted. A unit attribute keeps one entry per arg.
      argModes.push_back(UnitAttr::get(ctx));
      continue;
    }

    // Expect "arg:val=<dir>", ignoring spaces.
    StringRef dir = marker;
    bool wellFormed = dir.consume_front("val");
    if (wellFormed) {
      dir = dir.ltrim();
      wellFormed = dir.consume_front("=");
      dir = dir.trim();
    }
    std::optional<ArgDirection> argDirection =
        wellFormed ? symbolizeArgDirection(dir) : std::nullopt;
    if (!argDirection) {
      why = "argument '" + arg.str() + "' has invalid marker '" + marker.str() +
            "', expected :val=in, :val=out, or :val=inout";
      return failure();
    }

    // Only an argument passed through a pointer has a pointee to lift.
    if (!isa<LLVM::LLVMPointerType>(
            funcOp.getArgumentTypes()[i + sretOffset])) {
      why = "argument '" + arg.str() +
            "' is marked, but is not passed through a pointer";
      return failure();
    }
    if (sretOffset && *argDirection != ArgDirection::in) {
      why = "argument '" + arg.str() +
            "' is marked as written, but the function already returns "
            "through an sret pointer";
      return failure();
    }

    // The frontend records the pointee type of every pointer to a complete
    // type.
    auto type = argTypes && i < argTypes.size()
                    ? dyn_cast<TypeAttr>(argTypes[i])
                    : TypeAttr();
    if (!type) {
      why = "argument '" + arg.str() +
            "' is marked, but no type is known for the value behind it";
      return failure();
    }
    argModes.push_back(DictionaryAttr::get(
        ctx, {NamedAttribute(StringAttr::get(ctx, "type"), type),
              NamedAttribute(StringAttr::get(ctx, "dir"),
                             ArgDirectionAttr::get(ctx, *argDirection))}));
  }
  return ArrayAttr::get(ctx, argModes);
}

// Leave `funcOp` as an ordinary function, saying why its marking is not used.
void dropMarking(LLVM::LLVMFuncOp funcOp, StringAttr marking,
                 const Twine &why) {
  funcOp.emitWarning() << "ignoring tessera op marking '" << marking.getValue()
                       << "': " << why;
  funcOp->removeAttr("tessera_op");
  funcOp->removeAttr("pure_tessera_op");
  funcOp->removeAttr(kArgTypesAttr);
}

// Rewrite 'llvm.func' -> 'tessera.define'
class FuncOpRewrite final : public OpRewritePattern<LLVM::LLVMFuncOp> {
public:
  using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp funcOp,
                                PatternRewriter &rewriter) const override {

    auto module = funcOp->getParentOfType<ModuleOp>();
    auto *ctx = funcOp->getContext();

    bool isPure = false;
    StringAttr tesseraOpAttr = getMarking(funcOp, isPure);
    if (!tesseraOpAttr)
      return failure();

    // Bad markings were already dropped, with a warning, before rewriting.
    std::string why;
    FailureOr<ArrayAttr> argModes =
        buildArgModes(funcOp, tesseraOpAttr.getValue(), why);
    if (failed(argModes))
      return failure();
    StringRef tesseraName = getTesseraName(tesseraOpAttr.getValue());

    auto funcName = funcOp.getName();
    auto llvmFuncType = funcOp.getFunctionType();
    auto params = llvmFuncType.getParams();
    auto retType = llvmFuncType.getReturnType();

    auto fnType = FunctionType::get(
        ctx, params,
        isa<LLVM::LLVMVoidType>(retType) ? TypeRange{} : TypeRange{retType});

    // Redirect taken addresses first, so only calls follow the rename.
    redirectAddressUses(funcOp, rewriter, module);

    if (failed(SymbolTable::replaceAllSymbolUses(
            funcOp.getSymNameAttr(), StringAttr::get(ctx, tesseraName),
            module)))
      return failure();

    // A declaration must be private: FunctionOpInterface rejects a public
    // symbol with no body.
    auto tesseraDefineOp = tessera::DefineOp::create(
        rewriter, funcOp.getLoc(), tesseraName.str(), fnType, *argModes, isPure,
        funcOp.isExternal() ? rewriter.getStringAttr("private") : StringAttr());

    // Copy the other attributes, minus the name, type and marking.
    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != SymbolTable::getSymbolAttrName() &&
          namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != "tessera_op" &&
          namedAttr.getName() != "pure_tessera_op" &&
          namedAttr.getName() != kArgTypesAttr)
        tesseraDefineOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // For tessera-to-llvm to restore.
    tesseraDefineOp->setAttr("tessera.original_name",
                             rewriter.getStringAttr(funcName));

    if (!funcOp.isExternal()) {
      rewriter.inlineRegionBefore(funcOp.getBody(), tesseraDefineOp.getBody(),
                                  tesseraDefineOp.end());
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

// Rewrite 'llvm.call' -> 'tessera.call'
class CallOpRewrite final : public OpRewritePattern<LLVM::CallOp> {
public:
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp,
                                PatternRewriter &rewriter) const override {

    auto module = callOp->getParentOfType<ModuleOp>();

    auto calleeAttr = callOp.getCalleeAttr();
    if (!calleeAttr)
      return failure();

    auto callee = SymbolTable::lookupSymbolIn(module, calleeAttr);
    auto defineOp = dyn_cast_or_null<tessera::DefineOp>(callee);
    if (!defineOp)
      return failure();

    Value sretPtr;
    Type sretType;
    auto operands = callOp.getOperands();
    auto argAttrs = callOp.getArgAttrsAttr();
    SmallVector<Attribute> newArgAttrs;
    SmallVector<NamedAttribute> newAttrs;

    // tessera.call returns an sret value directly, as its pointee type.
    if (!operands.empty() && argAttrs) {
      if (auto sretAttr = defineOp.getSretAttr()) {
        sretPtr = callOp.getOperand(0);
        sretType = cast<TypeAttr>(sretAttr).getValue();
      }
    }

    // Drop the arg_attrs of the sret and "out" arguments, whose operands are
    // dropped below, so arg_attrs stays aligned with the operands.
    if (argAttrs) {
      for (unsigned i = 0, e = argAttrs.size(); i != e; ++i) {
        bool skip = sretPtr
                        ? (i == 0)
                        : (defineOp.argIsWritten(i) && !defineOp.argIsRead(i));
        if (!skip)
          newArgAttrs.push_back(argAttrs[i]);
      }
      for (auto attr : callOp->getAttrs()) {
        if (attr.getName() != callOp.getArgAttrsAttrName())
          newAttrs.push_back(attr);
      }
      newAttrs.push_back(rewriter.getNamedAttr(
          callOp.getArgAttrsAttrName(), rewriter.getArrayAttr(newArgAttrs)));
    } else {
      newAttrs.append(callOp->getAttrs().begin(), callOp->getAttrs().end());
    }

    // Operands, minus the sret pointer and "out" arguments, with byval and
    // lifted pointers replaced by the value loaded from them.
    SmallVector<Value> newOperands;
    SmallVector<Value> resultArgPtrs;
    SmallVector<Type> resultArgTypes;
    int argOffset = sretPtr ? 1 : 0;

    for (unsigned i = 0; i < operands.size() - argOffset; i++) {
      auto operand = callOp.getOperand(i + argOffset);

      if (!isa<LLVM::LLVMPointerType>(operand.getType())) {
        newOperands.push_back(operand);
        continue;
      }

      // Each written argument becomes a result of the call. An "inout" one
      // also stays an operand, carrying its incoming value.
      if (defineOp.argIsWritten(i)) {
        Type resultArgType = defineOp.getArgLiftedType(i);
        resultArgPtrs.push_back(operand);
        resultArgTypes.push_back(resultArgType);
        if (!defineOp.argIsRead(i))
          continue;
      }

      // Load lifted pointers. getCallOperandPointeeType takes the index in
      // the new operand list, not `i`.
      if (Type pointeeType =
              defineOp.getCallOperandPointeeType(newOperands.size())) {
        auto loadedVal = LLVM::LoadOp::create(rewriter, callOp.getLoc(),
                                              pointeeType, operand);
        newOperands.push_back(loadedVal);
      } else {
        newOperands.push_back(operand);
      }
    }

    // Results are stored back through their pointers after the call, so
    // existing loads from them still see the right value.
    if (sretPtr) {
      auto newCall =
          tessera::CallOp::create(rewriter, callOp.getLoc(),
                                  TypeRange{sretType}, newOperands, newAttrs);
      LLVM::StoreOp::create(rewriter, callOp.getLoc(), newCall.getResult(0),
                            sretPtr);
      rewriter.eraseOp(callOp);
    } else {
      // Written arguments come first, then the original results, so a rule's
      // result 0 is the output argument rather than, say, a status code.
      SmallVector<Type> newResultTypes(resultArgTypes.begin(),
                                       resultArgTypes.end());
      newResultTypes.append(callOp.getResultTypes().begin(),
                            callOp.getResultTypes().end());
      auto newCall = tessera::CallOp::create(
          rewriter, callOp.getLoc(), newResultTypes, newOperands, newAttrs);
      for (auto [idx, ptr] : llvm::enumerate(resultArgPtrs)) {
        LLVM::StoreOp::create(rewriter, callOp.getLoc(), newCall.getResult(idx),
                              ptr);
      }
      unsigned originalNumResults = callOp.getNumResults();
      rewriter.replaceOp(callOp,
                         newCall.getResults().take_back(originalNumResults));
    }
    return success();
  }
};

// Rewrite 'llvm.return' -> 'tessera.return'
class ReturnOpRewrite final : public OpRewritePattern<LLVM::ReturnOp> {
public:
  using OpRewritePattern<LLVM::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<tessera::DefineOp>(returnOp->getParentOp()))
      return failure();

    rewriter.replaceOpWithNewOp<tessera::ReturnOp>(returnOp,
                                                   returnOp.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass to convert Func operations into Tessera operations
//===----------------------------------------------------------------------===//

struct LLVMToTesseraPass
    : public enzyme::tessera::impl::LLVMToTesseraPassBase<LLVMToTesseraPass> {
  using LLVMToTesseraPassBase::LLVMToTesseraPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    auto module = cast<ModuleOp>(getOperation());

    // Check markings before rewriting. A bad one warns once and its function
    // stays an ordinary function, so a mistake never fails the compile.
    llvm::MapVector<StringRef, SmallVector<LLVM::LLVMFuncOp>> byTesseraName;
    for (auto funcOp : module.getOps<LLVM::LLVMFuncOp>()) {
      bool isPure;
      StringAttr marking = getMarking(funcOp, isPure);
      if (!marking)
        continue;
      std::string why;
      if (failed(buildArgModes(funcOp, marking.getValue(), why))) {
        dropMarking(funcOp, marking, why);
        continue;
      }
      byTesseraName[getTesseraName(marking.getValue())].push_back(funcOp);
    }
    // If two functions want the same tessera name, or it's already taken,
    // which one a rule meant is unknown, so none is lifted.
    for (auto &[tesseraName, funcs] : byTesseraName) {
      Operation *existing = module.lookupSymbol(tesseraName);
      bool clashes = funcs.size() > 1 ||
                     (existing && existing != funcs.front().getOperation());
      if (!clashes)
        continue;
      for (LLVM::LLVMFuncOp funcOp : funcs) {
        bool isPure;
        dropMarking(funcOp, getMarking(funcOp, isPure),
                    "another symbol already has the name '" + tesseraName +
                        "'");
      }
    }

    // Functions first, so every callee is a tessera.define before calls are
    // converted. In one sweep, a call reached before its callee (clang emits
    // declarations last) could be missed, since renaming the callee doesn't
    // revisit the call.
    patterns.add<FuncOpRewrite>(ctx);
    RewritePatternSet callPatterns(ctx);
    callPatterns.add<CallOpRewrite, ReturnOpRewrite>(ctx);

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config)) ||
        failed(applyPatternsGreedily(getOperation(), std::move(callPatterns),
                                     config))) {
      llvm::errs() << "Failed to convert LLVM dialect operations to tessera "
                      "dialect operations\n";
      return signalPassFailure();
    }

    if (auto annotations = module.lookupSymbol("llvm.global.annotations")) {
      annotations->erase();
    }
  }
};
} // namespace
