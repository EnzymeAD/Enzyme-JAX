//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert operations in the LLVM dialect to
// operations in the Tessera dialect.
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

// Rewrite 'llvm.func' -> 'tessera.define'
class FuncOpRewrite final : public OpRewritePattern<LLVM::LLVMFuncOp> {
public:
  FuncOpRewrite(MLIRContext *ctx,
                const llvm::DenseMap<unsigned, mlir::Type> &argTypesByIndex)
      : OpRewritePattern<LLVM::LLVMFuncOp>(ctx),
        argTypesByIndex(argTypesByIndex) {}

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp funcOp,
                                PatternRewriter &rewriter) const override {

    auto module = funcOp->getParentOfType<ModuleOp>();
    auto *ctx = funcOp->getContext();

    // Only rewrite if op has tessera_op or pure_tessera_op attribute
    StringAttr tesseraOpAttr;
    bool isPure = false;

    if (auto attr = funcOp->getAttrOfType<StringAttr>("tessera_op")) {
      tesseraOpAttr = attr;
    } else if (auto attr =
                   funcOp->getAttrOfType<StringAttr>("pure_tessera_op")) {
      tesseraOpAttr = attr;
      isPure = true;
    }

    if (!tesseraOpAttr)
      return failure();

    // Parse the tessera op attribute, which is expected to be in the format:
    // "tessera_op(arg1:byref, arg2, ...):globals=index1,..." or
    // "pure_tessera_op(arg1:byref, arg2, ...):globals=index1,..." where
    // index1,... corresponds positionally, in left-to-right order, to the byref
    // args in the arg list above, and are used to look up the types of the
    // byref args. The attribute could also contain output arguments marked
    // with ":result" (e.g. "tessera_op(arg1:result, arg2:byref):globals=0,1"),
    // whose types are also stored in the map.
    StringRef raw = tesseraOpAttr.getValue();

    // Parse op name (everything before the '(')
    StringRef tesseraName = raw.take_while([](char c) { return c != '('; });

    // Parse args in parentheses
    StringRef argList = raw.slice(raw.find('(') + 1, raw.find(')'));

    // Parse indices after ":globals=" (order corresponds to order byref
    // or result args appear in argList)
    SmallVector<StringRef> indexList;
    StringRef indicesStr = raw.substr(raw.find(')') + 1);
    if (!indicesStr.empty() && indicesStr.consume_front(":globals=")) {
      indicesStr.split(indexList, ',');
    }

    // Identify which args are marked byref or result and look up their types
    // in the argTypesByIndex map. :byref and :result args share a single
    // :globals= index list, consumed in left-to-right arg order regardless
    // of which marker each pointer-typed arg uses, so both loops below
    // advance the same shared counter.
    SmallVector<Attribute> byRefTypes;
    SmallVector<Attribute> resultArgTypes;
    unsigned numIndicesFound = 0;

    // Consumes the next shared global index for a byref/result-marked arg,
    // looks up its pointee type by that index in argTypesByIndex, and
    // appends the resolved TypeAttr to `out`.
    auto consumeMarkedArg =
        [&](StringRef kindName,
            SmallVectorImpl<Attribute> &out) -> LogicalResult {
      if (numIndicesFound >= indexList.size()) {
        funcOp->emitError(
            "tessera: not enough global indices for byref/result args");
        return failure();
      }
      // Find the types of the byref/result arguments by looking for global
      // variables with names that match the pattern "__tessera_arg_type_<idx>"
      // where <idx> is a number parsed in the tessera_op attribute after
      // "globals=".
      StringRef indexStr = indexList[numIndicesFound++];
      unsigned idx;
      if (indexStr.trim().getAsInteger(10, idx)) {
        funcOp->emitError("tessera: invalid ")
            << kindName << " type index: " << indexStr;
        return failure();
      }
      auto it = argTypesByIndex.find(idx);
      if (it == argTypesByIndex.end()) {
        funcOp->emitError("tessera: no ")
            << kindName << " type found for index: " << idx;
        return failure();
      }
      out.push_back(TypeAttr::get(it->second));
      return success();
    };

    if (!argList.trim().empty()) {
      SmallVector<StringRef> argParts;
      argList.split(argParts, ',');
      for (unsigned i = 0, e = argParts.size(); i != e; ++i) {
        StringRef arg = argParts[i].trim();

        // Check if arg is marked as byref and add its type to byRefTypes.
        if (arg.contains(":byref") || arg.contains(": byref")) {
          if (failed(consumeMarkedArg("byref", byRefTypes)))
            return failure();
        } else {
          // Push a unit attribute so that the byRefTypes array has the same
          // size as the number of args.
          byRefTypes.push_back(UnitAttr::get(ctx));
        }

        // Check if arg is marked as an output/result and add its type to
        // resultArgTypes.
        if (arg.contains(":result") || arg.contains(": result")) {
          if (failed(consumeMarkedArg("result", resultArgTypes)))
            return failure();
        } else {
          // Push a unit attribute so that the resultArgTypes array has the
          // same size as the number of args.
          resultArgTypes.push_back(UnitAttr::get(ctx));
        }
      }
    }

    // The format guarantees one global index per byref/result arg.
    // A mismatch means the annotation string is malformed or the Clang
    // plugin's emission order has drifted from this parser's assumption.
    if (numIndicesFound != indexList.size()) {
      funcOp->emitError("tessera: mismatch between byref/result arg count and "
                        "global index count");
      return failure();
    }

    auto funcName = funcOp.getName();
    auto llvmFuncType = funcOp.getFunctionType();
    auto params = llvmFuncType.getParams();
    auto retType = llvmFuncType.getReturnType();

    auto fnType = FunctionType::get(
        ctx, params,
        isa<LLVM::LLVMVoidType>(retType) ? TypeRange{} : TypeRange{retType});

    // Replace current function name with tessera name defined in
    // tessera_op / pure_tessera_op attribute
    if (failed(SymbolTable::replaceAllSymbolUses(
            funcOp.getSymNameAttr(), StringAttr::get(ctx, tesseraName),
            module)))
      return failure();

    bool hasResultArg = llvm::any_of(
        resultArgTypes, [](Attribute attr) { return isa<TypeAttr>(attr); });

    // Create the tessera.define op with the new name, function type, byRef
    // args, sizes, and purity (side effect free) attribute
    auto tesseraDefineOp = tessera::DefineOp::create(
        rewriter, funcOp.getLoc(), tesseraName.str(), fnType,
        ArrayAttr::get(ctx, byRefTypes), isPure,
        hasResultArg ? ArrayAttr::get(ctx, resultArgTypes) : ArrayAttr());

    // Copy over all attributes other than the function name and type
    // and tessera_op / pure_tessera_op attribute.
    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != SymbolTable::getSymbolAttrName() &&
          namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != "tessera_op" &&
          namedAttr.getName() != "pure_tessera_op")
        tesseraDefineOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Store the original function name so we can convert back to it later
    tesseraDefineOp->setAttr("tessera.original_name",
                             rewriter.getStringAttr(funcName));

    // Clone body of function
    if (!funcOp.isExternal()) {
      rewriter.inlineRegionBefore(funcOp.getBody(), tesseraDefineOp.getBody(),
                                  tesseraDefineOp.end());
    }

    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  const llvm::DenseMap<unsigned, mlir::Type> &argTypesByIndex;
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

    // Only rewrite if callee is a tessera.define op
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

    // If the first operand has an sret attribute, use its pointed-to type as
    // the SSA return type, since tessera.call returns values directly rather
    // than writing through a pointer.
    if (!operands.empty() && argAttrs) {
      if (auto sretAttr = defineOp.getSretAttr()) {
        sretPtr = callOp.getOperand(0);
        sretType = cast<TypeAttr>(sretAttr).getValue();
      }
    }

    // Build newAttrs from the original call's attributes. If arg_attrs is
    // present, filter out the entry for the sret arg (always index 0) or
    // for any pure result/output arguments, since those operands are
    // excluded from tessera.call's operand list below and arg_attrs must
    // stay aligned with operand position.
    if (argAttrs) {
      for (unsigned i = 0, e = argAttrs.size(); i != e; ++i) {
        bool skip = sretPtr ? (i == 0) : defineOp.isResultOnlyArg(i);
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

    // Build operands without first element if sretPtr is present.
    // If a pointer operand has a LLVM byVal attribute or was marked
    // as byRef by the user, load the value from the pointer and store
    // that as the new operand. Don't include pure result/output
    // arguments in the new operand list.
    SmallVector<Value> newOperands;
    SmallVector<int32_t> loadedOperands;
    SmallVector<Value> resultArgPtrs;
    SmallVector<Type> resultArgTypes;
    int argOffset = sretPtr ? 1 : 0;

    for (unsigned i = 0; i < operands.size() - argOffset; i++) {
      auto operand = callOp.getOperand(i + argOffset);

      if (!isa<LLVM::LLVMPointerType>(operand.getType())) {
        newOperands.push_back(operand);
        continue;
      }

      // Exclude pure result/output arguments from the new operands
      // list and remember their pointer and pointee types for later
      // use after the tessera.call is built
      if (defineOp.isResultOnlyArg(i)) {
        resultArgPtrs.push_back(operand);
        resultArgTypes.push_back(defineOp.getResultArgType(i));
        continue;
      }

      // Determine whether to load pointer and what type to load based on LLVM
      // byVal attribute or user-marked byRef attribute on the tessera.define
      // op. If neither is present, just pass the pointer through.
      // newOperands.size() is the number of tessera.call operands emitted
      // so far, i.e. the call-operand index this entry will land at once
      // pushed below -- getArgAttr expects that stripped index, not the
      // pre-strip loop index i.
      Type pointeeType;
      if (auto byValAttr = defineOp.getArgAttr(
              newOperands.size(), LLVM::LLVMDialect::getByValAttrName())) {
        pointeeType = cast<TypeAttr>(byValAttr).getValue();
      } else if (auto byRefType = defineOp.getByRefType(i)) {
        pointeeType = byRefType;
      }

      if (pointeeType) {
        auto loadedVal = LLVM::LoadOp::create(rewriter, callOp.getLoc(),
                                              pointeeType, operand);
        newOperands.push_back(loadedVal);
        loadedOperands.push_back(i);
      } else {
        newOperands.push_back(operand);
      }
    }

    newAttrs.push_back(
        rewriter.getNamedAttr("tessera.loaded_operands",
                              rewriter.getDenseI32ArrayAttr(loadedOperands)));

    // Create tessera.call op with results conveyed as direct SSA values rather
    // than written through pointers -- either the sret-derived value, or the
    // natural return (if any) plus one trailing value per result/output
    // argument.
    if (sretPtr) {
      // Set the result type of the new tessera.call op to be the sret pointee
      // type and store the result back through the sret pointer.
      auto newCall =
          tessera::CallOp::create(rewriter, callOp.getLoc(),
                                  TypeRange{sretType}, newOperands, newAttrs);
      LLVM::StoreOp::create(rewriter, callOp.getLoc(), newCall.getResult(0),
                            sretPtr);
      rewriter.eraseOp(callOp);
    } else {
      // Add result argument types to the front of the result type list of
      // the new tessera.call op -- one leading result per pure-output
      // argument, in argument order -- followed by the natural (original)
      // result types. Result-arg-derived results come first so that a PDL
      // rule referencing result 0 always sees the semantically meaningful
      // output-param value rather than the natural/ternary return.
      SmallVector<Type> newResultTypes(resultArgTypes.begin(),
                                       resultArgTypes.end());
      newResultTypes.append(callOp.getResultTypes().begin(),
                            callOp.getResultTypes().end());
      auto newCall = tessera::CallOp::create(
          rewriter, callOp.getLoc(), newResultTypes, newOperands, newAttrs);
      // Store each leading result back through its original pointer so any
      // pre-existing downstream load from that pointer still sees the right
      // value.
      for (auto [idx, ptr] : llvm::enumerate(resultArgPtrs)) {
        LLVM::StoreOp::create(rewriter, callOp.getLoc(), newCall.getResult(idx),
                              ptr);
      }
      // If the original call had any (natural) results, replace them with
      // the new call's trailing results.
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

    // Only rewrite if parent op is a tessera.define op
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

    // Build a lookup of argument index -> resolved type by scanning the
    // module once for globals named "__tessera_arg_type_<index>" (names
    // may be mangled with compiler-added prefixes, e.g.
    // "_ZL20__tessera_arg_type_0", so we search for the marker rather
    // than requiring it at the start). Each tessera.define op's own index
    // list (from its "globals=" attribute) is later resolved against this
    // map, avoiding a full module rescan per op.
    llvm::DenseMap<unsigned, mlir::Type> argTypesByIndex;
    StringRef prefix = "__tessera_arg_type_";
    for (auto global : module.getOps<mlir::LLVM::GlobalOp>()) {
      StringRef name = global.getSymName();
      size_t pos = name.find(prefix);
      if (pos == StringRef::npos)
        continue;
      StringRef idxStr = name.drop_front(pos + prefix.size());
      unsigned idx;
      if (idxStr.getAsInteger(10, idx))
        continue; // not a well-formed index suffix, skip

      auto [it, inserted] = argTypesByIndex.try_emplace(idx, global.getType());
      if (!inserted) {
        llvm::errs()
            << "Tessera: found multiple globals matching argument type index "
            << idx << ":\n";
        return signalPassFailure();
      }
    }

    // Convert annotated functions to tessera.define first, in their own
    // pass. CallOpRewrite only matches once its callee is already a
    // tessera.define, so running both patterns in one top-down sweep 
    // therefore silently drops calls whose callee declaration appears later 
    // in the module than the call site itself -- e.g. an extern function only 
    // referenced from the first function defined in a translation unit, 
    // Running two phases eliminates dependency on module order.
    RewritePatternSet funcPatterns(ctx);
    funcPatterns.add<FuncOpRewrite>(ctx, argTypesByGlobalIndices);
    if (failed(applyPatternsGreedily(getOperation(), std::move(funcPatterns)))) {
      llvm::errs() << "Failed to convert LLVM dialect operations to tessera "
                      "dialect operations\n";
      return signalPassFailure();
    }

    RewritePatternSet callPatterns(ctx);
    callPatterns.add<CallOpRewrite, ReturnOpRewrite>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(callPatterns)))) {
      llvm::errs() << "Failed to convert LLVM dialect operations to tessera "
                      "dialect operations\n";
      return signalPassFailure();
    }

    // Clean up llvm.global.annotations after conversion if it still exists
    if (auto annotations = module.lookupSymbol("llvm.global.annotations")) {
      annotations->erase();
    }
  }
};
} // namespace
