//===----------------------------------------------------------------------===//
//
// Converts tessera.defines and tessera.calls back to llvm.funcs and
// llvm.calls.
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
#include "src/enzyme_ad/jax/Passes/Tessera/PropertyPropagation.h"

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

// For each tessera.call, the original address to pass for each operand, or
// null. See findAddresses.
using CallAddresses = DenseMap<Operation *, SmallVector<Value>>;

// Whether `op` and its nested ops write no memory. Calls that write none of
// their arguments, like a guard's mpfi_is_empty check, count as not writing.
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

// The address `value` was loaded from, if nothing between the load and `call`
// writes memory, so the call can be given that address instead of a copy.
// "Between" covers the rest of the load's block, the blocks on the way to the
// call (a guard branches between blocks), and the ops ahead of the call in
// each enclosing scf.if (a guard in a loop body). No loop may be in between.
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

// Which operands of each tessera.call can be passed by their original address,
// so the callee sees the same pointers as before. MPFR and MPFI compare
// pointers to detect aliasing, and copies sharing the same limbs would fool
// them. Done before converting, while a guard's checks are still tessera.calls
// that writesNothing recognizes.
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

    SmallVector<Type> argTypes;
    for (auto type : fnType.getInputs())
      argTypes.push_back(typeConverter.convertType(type));

    Type returnType = fnType.getNumResults() == 0
                          ? LLVM::LLVMVoidType::get(ctx)
                          : typeConverter.convertType(fnType.getResult(0));
    auto llvmFuncType = LLVM::LLVMFunctionType::get(returnType, argTypes);
    if (!llvmFuncType)
      return failure();

    if (failed(SymbolTable::replaceAllSymbolUses(
            defineOp.getSymNameAttr(), StringAttr::get(ctx, funcName), module)))
      return failure();

    auto funcOp = LLVM::LLVMFuncOp::create(rewriter, defineOp.getLoc(),
                                           funcName, llvmFuncType);

    // Copy the attributes that aren't tessera-only.
    for (const auto &namedAttr : defineOp->getAttrs()) {
      if (namedAttr.getName() != SymbolTable::getSymbolAttrName() &&
          namedAttr.getName() != defineOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != defineOp.getArgModesAttrName() &&
          namedAttr.getName() != defineOp.getPureAttrName() &&
          namedAttr.getName() != "tessera.original_name")
        funcOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

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

    // Pass each lifted operand by pointer again: its original address if
    // still valid (see findAddresses), else a fresh alloca holding it. A value
    // passed twice, as in mpfi_mul(r, r, y), gets one alloca, so the callee
    // still sees the operands alias.
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
      // A tail call promises the callee doesn't touch the caller's allocas.
      // Passing the allocas made here (`one` is set exactly when there are
      // some) breaks that, and LLVM would then fold away what the callee
      // writes to them, so drop the tail marker.
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

    // An sret callee gets an alloca for the result as its first argument.
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

      auto sretType = callOp.getResult(0).getType();
      Value sretPtr = LLVM::AllocaOp::create(
          rewriter, callOp.getLoc(),
          LLVM::LLVMPointerType::get(callOp->getContext()), sretType, one,
          sret_alignment);

      SmallVector<Value> newOperands;
      newOperands.push_back(sretPtr);
      newOperands.append(reconstructedOperands.begin(),
                         reconstructedOperands.end());

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

      auto loadedResult =
          LLVM::LoadOp::create(rewriter, callOp.getLoc(), sretType, sretPtr);
      rewriter.replaceOp(callOp, loadedResult.getResult());

    } else if (defineOp.getNumWrittenArgs() > 0) {
      // Give each "out" argument an alloca, in its position, and read every
      // written argument back after the call.
      SmallVector<Value> newOperands;
      SmallVector<Attribute> newArgAttrs;
      SmallVector<Value> resultArgPtrs;
      SmallVector<Type> resultArgTypes;
      auto callArgAttrs = callOp.getArgAttrsAttr();
      unsigned tesseraCallIdx = 0;

      for (unsigned i = 0, e = defineOp.getNumArguments(); i != e; ++i) {
        if (defineOp.argIsWritten(i) && !defineOp.argIsRead(i)) {
          Type resultArgType = defineOp.getArgLiftedType(i);
          if (!resultArgType)
            return callOp.emitOpError(
                "tessera.call to function with write-only argument must "
                "have a result type for that argument");

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
        } else {
          if (tesseraCallIdx >= reconstructedOperands.size())
            return callOp.emitOpError(
                "tessera.call has fewer operands than expected by callee");
          Value operand = reconstructedOperands[tesseraCallIdx];
          newOperands.push_back(operand);
          // An "inout" argument is read back from the pointer it was passed
          // through.
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

      // Written arguments first, in order, then the original result.
      SmallVector<Value> replacementValues;
      for (auto [ptr, type] : llvm::zip(resultArgPtrs, resultArgTypes)) {
        auto loaded =
            LLVM::LoadOp::create(rewriter, callOp.getLoc(), type, ptr);
        replacementValues.push_back(loaded.getResult());
      }
      if (fnType.getNumResults() > 0)
        replacementValues.push_back(newCall.getResult());
      rewriter.replaceOp(callOp, replacementValues);
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

    // Calls first, while their callees are still tessera.defines, which
    // CallOpRewrite reads. Converting a define first would strand its calls.
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

    // Point taken addresses back from llvm-to-tessera's stubs to the original
    // functions, and drop the stubs.
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

    // Facts are no longer needed, so erase the statement-fact marker calls
    // (see PropertyPropagation.h) and the markers.
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
