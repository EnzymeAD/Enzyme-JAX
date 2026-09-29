//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert operations in the Tessera dialect to
// operations in the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Properties.h"

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_TESSERATOLLVMPASS
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

/// For each tessera.call, the address to pass for each operand that can be
/// passed as the pointer it was loaded through, or null (see findAddresses).
using CallAddresses = DenseMap<Operation *, SmallVector<Value>>;

/// Whether `op`, and everything nested in it, leaves memory as it is, except
/// perhaps through calls that write none of their arguments: the checks a
/// guard makes before its branch, such as mpfi_is_empty.
bool writesNothing(Operation *op) {
  WalkResult result = op->walk([](Operation *nested) {
    if (auto call = dyn_cast<tessera::CallOp>(nested)) {
      auto define = SymbolTable::lookupNearestSymbolFrom<tessera::DefineOp>(
          call, call.getCalleeAttr());
      return define && define.getNumWrittenArgs() == 0
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    }
    // Its nested ops are walked on their own.
    if (nested->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return WalkResult::advance();
    auto effects = dyn_cast<MemoryEffectOpInterface>(nested);
    if (effects && !effects.hasEffect<MemoryEffects::Write>() &&
        !effects.hasEffect<MemoryEffects::Free>())
      return WalkResult::advance();
    return WalkResult::interrupt();
  });
  return !result.wasInterrupted();
}

/// The address `value` was loaded from, if memory there still holds it when
/// `call` runs, so the call can be given that address instead of a copy: the
/// value is a load, and nothing that runs between the two may write memory.
///
/// Between them are the ops after the load in its block, any blocks control
/// can pass through on the way to the call (a guard at function level branches
/// between blocks), and, where the call is nested in an scf.if (a guard in a
/// loop body becomes one), the ops ahead of it at each level. A loop between
/// them would run the call again after whatever follows it, so none may be.
Value addressAt(Value value, Operation *call) {
  auto load = value.getDefiningOp<LLVM::LoadOp>();
  if (!load || load.getVolatile_())
    return Value();
  Block *loadBlock = load->getBlock();
  Region *region = loadBlock->getParent();

  // The ops ahead of the call at each level it is nested below the load's.
  Operation *outer = region->findAncestorOpInRegion(*call);
  if (!outer)
    return Value();
  for (Operation *op = call; op != outer; op = op->getParentOp()) {
    if (!isa<scf::IfOp>(op->getParentOp()) || !op->getBlock()->isEntryBlock())
      return Value();
    for (Operation *prev = op->getPrevNode(); prev; prev = prev->getPrevNode())
      if (!writesNothing(prev))
        return Value();
  }

  // The blocks between the load and the call.
  Block *callBlock = outer->getBlock();
  auto after = [&](Operation *from, Operation *until) {
    for (Operation *op = from; op && op != until; op = op->getNextNode())
      if (!writesNothing(op))
        return false;
    return true;
  };
  if (callBlock == loadBlock)
    return load->isBeforeInBlock(outer) && after(load->getNextNode(), outer)
               ? load.getAddr()
               : Value();
  if (!after(load->getNextNode(), nullptr) ||
      !after(&callBlock->front(), outer))
    return Value();
  SmallVector<Block *> worklist(callBlock->getPredecessors());
  DenseSet<Block *> seen;
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();
    if (block == loadBlock || !seen.insert(block).second)
      continue;
    if (block == callBlock || !after(&block->front(), nullptr))
      return Value();
    llvm::append_range(worklist, block->getPredecessors());
  }
  return load.getAddr();
}

/// Decide, before any call is converted, which operands each tessera.call can
/// be passed by their original address. That is what the call this came from
/// passed, so the callee sees the same pointers: MPFR and MPFI test whether
/// operands alias by comparing them, and two copies of values that point to
/// the same limbs look like different operands while sharing those limbs.
/// Decided up front because a guard's checks, which may run in between, are
/// recognized as writing nothing while they are still tessera.calls.
CallAddresses findAddresses(ModuleOp module) {
  CallAddresses addresses;
  module.walk([&](tessera::CallOp call) {
    auto define = SymbolTable::lookupNearestSymbolFrom<tessera::DefineOp>(
        call, call.getCalleeAttr());
    if (!define)
      return;
    SmallVector<Value> perOperand;
    bool any = false;
    for (auto [i, operand] : llvm::enumerate(call.getOperands())) {
      Value address;
      if (!isa<LLVM::LLVMPointerType>(operand.getType()) &&
          define.getCallOperandPointeeType(i))
        address = addressAt(operand, call);
      any |= static_cast<bool>(address);
      perOperand.push_back(address);
    }
    if (any)
      addresses[call] = std::move(perOperand);
  });
  return addresses;
}

// Rewrite 'tessera.define' -> 'llvm.func'
class DefineOpRewrite final : public OpRewritePattern<tessera::DefineOp> {
public:
  DefineOpRewrite(LLVMTypeConverter &typeConverter, MLIRContext *ctx)
      : OpRewritePattern(ctx), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(tessera::DefineOp defineOp,
                                PatternRewriter &rewriter) const override {
    auto funcNameAttr =
        defineOp->getAttrOfType<StringAttr>("tessera.original_name");
    if (!funcNameAttr)
      return failure();
    auto funcName = funcNameAttr.getValue();
    auto module = defineOp->getParentOfType<ModuleOp>();
    auto *ctx = defineOp->getContext();
    auto fnType = defineOp.getFunctionType();

    // Convert argument types
    SmallVector<Type> argTypes;
    for (auto type : fnType.getInputs())
      argTypes.push_back(typeConverter.convertType(type));

    // Handle return type - void if no results
    Type returnType = fnType.getNumResults() == 0
                          ? LLVM::LLVMVoidType::get(ctx)
                          : typeConverter.convertType(fnType.getResult(0));
    auto llvmFuncType = LLVM::LLVMFunctionType::get(returnType, argTypes);
    if (!llvmFuncType)
      return failure();

    // Replace tessera name with original function name
    if (failed(SymbolTable::replaceAllSymbolUses(
            defineOp.getSymNameAttr(), StringAttr::get(ctx, funcName), module)))
      return failure();

    // Create the `llvm.func` op
    auto funcOp = LLVM::LLVMFuncOp::create(rewriter, defineOp.getLoc(),
                                           funcName, llvmFuncType);

    // Copy over attributes other than the function name and type, arg modes,
    // pure, and other attributes used only for tessera conversion
    for (const auto &namedAttr : defineOp->getAttrs()) {
      if (namedAttr.getName() != SymbolTable::getSymbolAttrName() &&
          namedAttr.getName() != defineOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != defineOp.getArgModesAttrName() &&
          namedAttr.getName() != defineOp.getPureAttrName() &&
          namedAttr.getName() != "tessera.original_name")
        funcOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Clone body of function
    if (!defineOp.isExternal()) {
      rewriter.inlineRegionBefore(defineOp.getBody(), funcOp.getBody(),
                                  funcOp.end());
    }

    rewriter.eraseOp(defineOp);

    return success();
  }

private:
  LLVMTypeConverter &typeConverter;
};

// Rewrite 'tessera.call' -> 'llvm.call'
class CallOpRewrite final : public OpRewritePattern<tessera::CallOp> {
public:
  CallOpRewrite(const CallAddresses &addresses, MLIRContext *ctx)
      : OpRewritePattern(ctx), addresses(addresses) {}

  LogicalResult matchAndRewrite(tessera::CallOp callOp,
                                PatternRewriter &rewriter) const override {

    auto calleeAttr = callOp.getCalleeAttr();
    if (!calleeAttr)
      return failure();

    auto callee = SymbolTable::lookupSymbolIn(
        callOp->getParentOfType<ModuleOp>(), calleeAttr);

    auto defineOp = dyn_cast_or_null<tessera::DefineOp>(callee);
    if (!defineOp)
      return failure();

    // Any operand the callee takes through a pointer was loaded from
    // memory when converting LLVM to tessera, so put it back. Where memory
    // at the address it was loaded from still holds it, pass that address, as
    // the original call did (see findAddresses). Otherwise allocate fresh
    // stack storage, store the operand's value into it, and substitute that
    // pointer in its place. All other operands pass through unchanged.
    //
    // A value passed at several positions gets one allocation passed at each
    // of them, as in mpfi_mul(r, r, y): two copies would share whatever the
    // value points to without sharing the storage, and the callee could no
    // longer see that the operands alias.
    auto known = addresses.find(callOp);
    Value one;
    SmallVector<Value> reconstructedOperands;
    DenseMap<Value, Value> storageFor;
    for (auto [i, operand] : llvm::enumerate(callOp.getOperands())) {
      if (known != addresses.end() && known->second[i]) {
        reconstructedOperands.push_back(known->second[i]);
        continue;
      }
      if (Value storage = storageFor.lookup(operand)) {
        reconstructedOperands.push_back(storage);
        continue;
      }
      if (!isa<LLVM::LLVMPointerType>(operand.getType()) &&
          defineOp.getCallOperandPointeeType(i)) {
        if (!one)
          one = LLVM::ConstantOp::create(rewriter, callOp.getLoc(),
                                         rewriter.getI32Type(),
                                         rewriter.getI32IntegerAttr(1));
        int64_t alignment = 0;
        if (auto alignAttr =
                defineOp.getArgAttr(i, LLVM::LLVMDialect::getAlignAttrName()))
          alignment = cast<IntegerAttr>(alignAttr).getInt();
        auto AIOp = LLVM::AllocaOp::create(
            rewriter, callOp.getLoc(),
            LLVM::LLVMPointerType::get(callOp->getContext()), operand.getType(),
            one, alignment);
        Value AI = AIOp;
        LLVM::StoreOp::create(rewriter, callOp.getLoc(), operand, AI);
        reconstructedOperands.push_back(AI);
        storageFor[operand] = AI;
      } else {
        reconstructedOperands.push_back(operand);
      }
    }

    auto buildNewAttrs = [&](ArrayRef<NamedAttribute> baseAttrs,
                             int32_t numOperands,
                             std::optional<ArrayAttr> argAttrsOverride) {
      // A tail call promises that the callee does not access the caller's
      // allocas. The call this came from may have kept that promise, but one
      // given the stack storage allocated here (`one` is set exactly when
      // there is some) breaks it, and LLVM takes it at its word: it assumes
      // the callee leaves that storage alone and folds a result read back from
      // it to the value stored before the call, losing what the callee wrote.
      bool passesAllocas = static_cast<bool>(one);
      StringAttr tailCallKind =
          LLVM::CallOp::getTailCallKindAttrName(OperationName(
              LLVM::CallOp::getOperationName(), rewriter.getContext()));
      SmallVector<NamedAttribute> newAttrs;
      for (auto attr : baseAttrs) {
        if (passesAllocas && attr.getName() == tailCallKind)
          continue;
        if (attr.getName() != callOp.getArgAttrsAttrName() &&
            attr.getName() != "tessera.applied_rules" &&
            attr.getName() != "operandSegmentSizes" &&
            attr.getName() != "op_bundle_sizes")
          newAttrs.push_back(attr);
      }
      if (argAttrsOverride)
        newAttrs.push_back(rewriter.getNamedAttr(callOp.getArgAttrsAttrName(),
                                                 *argAttrsOverride));
      else if (auto argAttrs = callOp.getArgAttrsAttr())
        newAttrs.push_back(
            rewriter.getNamedAttr(callOp.getArgAttrsAttrName(), argAttrs));
      newAttrs.push_back(rewriter.getNamedAttr(
          "operandSegmentSizes",
          rewriter.getDenseI32ArrayAttr({numOperands, 0})));
      newAttrs.push_back(rewriter.getNamedAttr(
          "op_bundle_sizes", rewriter.getDenseI32ArrayAttr({})));
      return newAttrs;
    };

    // Check if callee's first argument has sret attribute. If so, allocate new
    // pointer to contain result of tessera.call and insert as first argument in
    // llvm.call.
    if (defineOp.getNumArguments() > 0 && defineOp.getSretAttr()) {
      if (callOp.getNumResults() == 0)
        return callOp.emitOpError(
            "tessera.call to sret function must have a result");
      auto sretArgAttrs = defineOp.getArgAttrDict(0);
      int64_t sret_alignment = 0;
      if (auto sretAlignAttr =
              sretArgAttrs.get(LLVM::LLVMDialect::getAlignAttrName()))
        sret_alignment = cast<IntegerAttr>(sretAlignAttr).getInt();
      if (!one)
        one = LLVM::ConstantOp::create(rewriter, callOp.getLoc(),
                                       rewriter.getI32Type(),
                                       rewriter.getI32IntegerAttr(1));

      // Allocate stack storage for the sret return value
      auto sretType = callOp.getResult(0).getType();
      Value sretPtr = LLVM::AllocaOp::create(
          rewriter, callOp.getLoc(),
          LLVM::LLVMPointerType::get(callOp->getContext()), sretType, one,
          sret_alignment);

      // Build new operands with sretPtr as first arg, followed by the
      // reconstructed operands
      SmallVector<Value> newOperands;
      newOperands.push_back(sretPtr);
      newOperands.append(reconstructedOperands.begin(),
                         reconstructedOperands.end());

      // Reconstruct arg attributes with sret attr first
      SmallVector<Attribute> newArgAttrs;
      newArgAttrs.push_back(sretArgAttrs);
      if (auto argAttrs = callOp.getArgAttrsAttr()) {
        for (auto argAttr : argAttrs)
          newArgAttrs.push_back(argAttr);
      }

      auto newAttrs = buildNewAttrs(callOp->getAttrs(), newOperands.size(),
                                    rewriter.getArrayAttr(newArgAttrs));

      LLVM::CallOp::create(rewriter, callOp.getLoc(), TypeRange{}, newOperands,
                           newAttrs);

      // Load result from sret pointer and replace uses
      auto loadedResult =
          LLVM::LoadOp::create(rewriter, callOp.getLoc(), sretType, sretPtr);
      rewriter.replaceOp(callOp, loadedResult.getResult());

    } else if (defineOp.getNumWrittenArgs() > 0) {
      // Allocate stack storage for each written argument, and splice those
      // pointers into the operand list in the right positions, so that the
      // LLVM::CallOp can be issued with the callee's real function type
      // and the write-only arguments passed by pointer.
      SmallVector<Value> newOperands;
      SmallVector<Attribute> newArgAttrs;
      SmallVector<Value> resultArgPtrs;
      SmallVector<Type> resultArgTypes;
      auto callArgAttrs = callOp.getArgAttrsAttr();
      unsigned tesseraCallIdx = 0;

      for (unsigned i = 0, e = defineOp.getNumArguments(); i != e; ++i) {
        // Check if argument is write-only
        if (defineOp.argIsWritten(i) && !defineOp.argIsRead(i)) {
          Type resultArgType = defineOp.getArgLiftedType(i);
          if (!resultArgType)
            return callOp.emitOpError(
                "tessera.call to function with write-only argument must "
                "have a result type for that argument");

          // Get alignment from callee's arg attrs
          int64_t alignment = 0;
          auto resultArgAttrs = defineOp.getArgAttrDict(i);
          if (resultArgAttrs) {
            if (auto alignAttr =
                    resultArgAttrs.get(LLVM::LLVMDialect::getAlignAttrName()))
              alignment = cast<IntegerAttr>(alignAttr).getInt();
          }
          if (!one)
            one = LLVM::ConstantOp::create(rewriter, callOp.getLoc(),
                                           rewriter.getI32Type(),
                                           rewriter.getI32IntegerAttr(1));

          // Allocate stack storage for the result/output argument
          auto resultArgPtrOp = LLVM::AllocaOp::create(
              rewriter, callOp.getLoc(),
              LLVM::LLVMPointerType::get(callOp->getContext()), resultArgType,
              one, alignment);

          Value resultArgPtr = resultArgPtrOp;
          newOperands.push_back(resultArgPtr);
          resultArgPtrs.push_back(resultArgPtr);
          resultArgTypes.push_back(resultArgType);
          newArgAttrs.push_back(
              resultArgAttrs ? resultArgAttrs : rewriter.getDictionaryAttr({}));

          // Argument is not written, or is "inout"
        } else {
          if (tesseraCallIdx >= reconstructedOperands.size())
            return callOp.emitOpError(
                "tessera.call has fewer operands than expected by callee");
          Value operand = reconstructedOperands[tesseraCallIdx];
          newOperands.push_back(operand);
          // If arg is inout, it was already reconstructed into an alloca.
          // We want to read that same alloca back after the call so the
          // callee's writes through the pointer become the arg's trailing
          // tessera.call result. Pushed in argument order, matching the
          // leading-result order LLVMToTessera built and the verifier checks.
          if (defineOp.argIsWritten(i)) {
            resultArgPtrs.push_back(operand);
            resultArgTypes.push_back(defineOp.getArgLiftedType(i));
          }
          newArgAttrs.push_back(callArgAttrs ? callArgAttrs[tesseraCallIdx]
                                             : rewriter.getDictionaryAttr({}));
          ++tesseraCallIdx;
        }
      }

      auto newAttrs = buildNewAttrs(callOp->getAttrs(), newOperands.size(),
                                    rewriter.getArrayAttr(newArgAttrs));

      auto fnType = defineOp.getFunctionType();
      Type resultType =
          fnType.getNumResults() > 0 ? fnType.getResult(0) : Type();
      TypeRange returnType = resultType ? TypeRange(resultType) : TypeRange();
      auto newCall = LLVM::CallOp::create(rewriter, callOp.getLoc(), returnType,
                                          newOperands, newAttrs);

      // Load each result arg's value back from its alloca, then replace
      // callOp's results: one loaded value per result argument (the
      // leading results on the tessera.call), in argument order, followed
      // by the natural result (if any), matching the new LLVM call's
      // direct result.
      SmallVector<Value> replacementValues;
      for (auto [ptr, type] : llvm::zip(resultArgPtrs, resultArgTypes)) {
        auto loaded =
            LLVM::LoadOp::create(rewriter, callOp.getLoc(), type, ptr);
        replacementValues.push_back(loaded.getResult());
      }
      if (fnType.getNumResults() > 0)
        replacementValues.push_back(newCall.getResult());
      rewriter.replaceOp(callOp, replacementValues);

      // Callee has no result args
    } else {
      auto newAttrs = buildNewAttrs(callOp->getAttrs(),
                                    reconstructedOperands.size(), std::nullopt);
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          callOp, callOp.getResultTypes(), reconstructedOperands, newAttrs);
    }

    return success();
  }

private:
  const CallAddresses &addresses;
};

// Rewrite 'tessera.return' -> 'llvm.return'
class ReturnOpRewrite final : public OpRewritePattern<tessera::ReturnOp> {
public:
  using OpRewritePattern<tessera::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tessera::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(returnOp,
                                                returnOp.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass to convert Tessera operations into Func operations
//===----------------------------------------------------------------------===//

struct TesseraToLLVMPass
    : public enzyme::tessera::impl::TesseraToLLVMPassBase<TesseraToLLVMPass> {
  using TesseraToLLVMPassBase::TesseraToLLVMPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LLVMTypeConverter typeConverter(ctx);
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);

    // Calls first, while their callees are still tessera.define: CallOpRewrite
    // reads the define, and converting a define renames the symbol every call
    // to it names. In one sweep the result would depend on the order the
    // driver visits ops in, and a define that comes after its callers -- as
    // any op that is only declared in the file does -- would convert first
    // and strand them.
    CallAddresses addresses = findAddresses(getOperation());
    RewritePatternSet callPatterns(ctx);
    callPatterns.add<CallOpRewrite>(addresses, ctx);
    RewritePatternSet definePatterns(ctx);
    definePatterns.add<DefineOpRewrite>(typeConverter, ctx);
    definePatterns.add<ReturnOpRewrite>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(callPatterns),
                                     config)) ||
        failed(applyPatternsGreedily(getOperation(), std::move(definePatterns),
                                     config))) {
      llvm::errs() << "Failed to convert tessera dialect operations to LLVM "
                      "dialect operations\n";
      return signalPassFailure();
    }

    // llvm-to-tessera pointed every taken address of a tessera op at a stub,
    // since llvm.mlir.addressof cannot name a tessera.define. The op is an
    // llvm.func of its original name again now, so point them back and drop
    // the stubs.
    ModuleOp module = getOperation();
    SmallVector<LLVM::LLVMFuncOp> stubs;
    for (auto func : module.getOps<LLVM::LLVMFuncOp>())
      if (func->hasAttr("tessera.address_stub_for"))
        stubs.push_back(func);
    for (LLVM::LLVMFuncOp stub : stubs) {
      auto target = stub->getAttrOfType<StringAttr>("tessera.address_stub_for");
      if (!target || !module.lookupSymbol<LLVM::LLVMFuncOp>(target) ||
          failed(SymbolTable::replaceAllSymbolUses(stub.getSymNameAttr(),
                                                   target, module))) {
        stub.emitError("could not restore uses of the tessera address stub");
        return signalPassFailure();
      }
      stub.erase();
    }

    // A fact stated on a statement arrived as a call to a marker that does
    // nothing (see Properties.h). Everything that reads facts has run, so the
    // calls go, and the markers with them.
    SmallVector<LLVM::LLVMFuncOp> markers;
    for (auto func : module.getOps<LLVM::LLVMFuncOp>())
      if (func->hasAttr(kFactMarkerAttr))
        markers.push_back(func);
    for (LLVM::LLVMFuncOp marker : markers) {
      SmallVector<LLVM::CallOp> calls;
      module.walk([&](LLVM::CallOp call) {
        if (call.getCallee() == marker.getSymName())
          calls.push_back(call);
      });
      for (LLVM::CallOp call : calls)
        call.erase();
      // Only calls are expected; anything else that names the marker keeps it.
      if (SymbolTable::symbolKnownUseEmpty(marker, module))
        marker.erase();
    }
  }
};
} // namespace
