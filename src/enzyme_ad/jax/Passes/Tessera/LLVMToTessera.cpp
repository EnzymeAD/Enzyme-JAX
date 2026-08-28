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
  FuncOpRewrite(
      MLIRContext *ctx,
      const llvm::DenseMap<unsigned, mlir::Type> &argTypesByGlobalIndices)
      : OpRewritePattern<LLVM::LLVMFuncOp>(ctx),
        argTypesByGlobalIndices(argTypesByGlobalIndices) {}

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
    // "tessera_op(arg1:val=in, arg2, ...):globals=index1,..." or
    // "pure_tessera_op(arg1:val=in, arg2, ...):globals=index1,...".
    //
    // A marker lifts an argument into the value domain: the call site
    // dereferences the pointer so that tessera.call carries the pointee as an
    // SSA value. That is what lets a PDL pattern match over calls, since
    // PDL matches SSA def-use edges and cannot see dependencies through
    // memory. An unmarked argument is passed through as-is.
    //
    // The marker's direction says what the callee does to the pointee:
    //   :val=in    reads it   -- takes an operand, yields no result
    //   :val=out   writes it  -- takes no operand, yields a trailing result
    //   :val=inout both       -- takes an operand and yields a result
    //
    // index1,... corresponds positionally, in left-to-right order, to the
    // marked args, one global counter index per marked arg, and is used to
    // look up each arg's type in argTypesByGlobalIndices.
    StringRef raw = tesseraOpAttr.getValue();

    // Parse op name (everything before the '(')
    StringRef tesseraName = raw.take_while([](char c) { return c != '('; });

    // Parse args in parentheses
    StringRef argList = raw.slice(raw.find('(') + 1, raw.find(')'));

    // Parse indices after ":globals=" (order corresponds to the order marked
    // args appear in argList)
    SmallVector<StringRef> indexList;
    StringRef indicesStr = raw.substr(raw.find(')') + 1);
    if (!indicesStr.empty() && indicesStr.consume_front(":globals=")) {
      indicesStr.split(indexList, ',');
    }

    // Identify which args carry a lifting marker and look up their pointee
    // types in the look up each map. Marked args consume the :globals=
    // index list in left-to-right arg order, one index each.
    SmallVector<Attribute> argModes;
    unsigned numIndicesFound = 0;

    // Consumes the next global index for a marked arg, looks up its type
    // by that index in argTypesByGlobalIndices, and returns the mode dictionary
    // pairing that type with `dir`.
    auto consumeMarkedArg = [&](StringRef dir) -> Attribute {
      if (numIndicesFound >= indexList.size()) {
        funcOp->emitError("tessera: not enough global indices for marked args");
        return nullptr;
      }
      // Find the pointee types of the marked arguments by looking for global
      // variables with names that match the pattern "__tessera_arg_type_<idx>"
      // where <idx> is a number parsed in the tessera_op attribute after
      // "globals=".
      StringRef indexStr = indexList[numIndicesFound++];
      unsigned idx;
      if (indexStr.trim().getAsInteger(10, idx)) {
        funcOp->emitError("tessera: invalid type index for :val=")
            << dir << " arg: " << indexStr;
        return nullptr;
      }
      auto it = argTypesByGlobalIndices.find(idx);
      if (it == argTypesByGlobalIndices.end()) {
        funcOp->emitError("tessera: no lifting entry found for :val=")
            << dir << " arg at index: " << idx;
        return nullptr;
      }
      return DictionaryAttr::get(ctx,
                                 {NamedAttribute(StringAttr::get(ctx, "type"),
                                                 TypeAttr::get(it->second)),
                                  NamedAttribute(StringAttr::get(ctx, "dir"),
                                                 StringAttr::get(ctx, dir))});
    };

    if (!argList.trim().empty()) {
      SmallVector<StringRef> argParts;
      argList.split(argParts, ',');
      for (unsigned i = 0, e = argParts.size(); i != e; ++i) {
        StringRef arg = argParts[i].trim();
        StringRef marker = arg.split(':').second.trim();
        if (marker.empty()) {
          // Unmarked: not lifted. Push a unit attribute so that argModes has
          // the same size as the number of args.
          argModes.push_back(UnitAttr::get(ctx));
          continue;
        }

        // Expect the form "arg:val=<dir>", disregarding additional spaces.
        StringRef dir = marker;
        bool wellFormed = dir.consume_front("val");
        if (wellFormed) {
          dir = dir.ltrim();
          wellFormed = dir.consume_front("=");
          dir = dir.trim();
        }
        if (!wellFormed || (dir != "in" && dir != "out" && dir != "inout")) {
          funcOp->emitError("tessera: argument '")
              << arg << "' has invalid marker '" << marker
              << "', expected :val=in, :val=out, or :val=inout";
          return failure();
        }

        Attribute mode = consumeMarkedArg(dir);
        if (!mode)
          return failure();
        argModes.push_back(mode);
      }
    }

    // The format guarantees one global index per marked arg.
    // A mismatch means the annotation string is malformed or the Clang
    // plugin's emission order has drifted from this parser's assumption.
    if (numIndicesFound != indexList.size()) {
      funcOp->emitError("tessera: mismatch between marked arg count and "
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

    // Create the tessera.define op with the new name, function type, per-arg
    // lifting modes, and purity (side effect free) attribute. External
    // declarations must be marked private: FunctionOpInterface's verifier
    // rejects a body-less symbol with public visibility, unlike
    // llvm.func where a public extern declaration is normal.
    auto tesseraDefineOp = tessera::DefineOp::create(
        rewriter, funcOp.getLoc(), tesseraName.str(), fnType,
        ArrayAttr::get(ctx, argModes), isPure,
        funcOp.isExternal() ? rewriter.getStringAttr("private") : StringAttr());

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
  const llvm::DenseMap<unsigned, mlir::Type> &argTypesByGlobalIndices;
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
    // for any write-only ("out") arguments, since those operands are
    // excluded from tessera.call's operand list below and arg_attrs must
    // stay aligned with operand position.
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

    // Build operands without first element if sretPtr is present.
    // If a pointer operand has a LLVM byVal attribute or was lifted by the
    // user, load the value from the pointer and store that as the new
    // operand. Don't include write-only ("out") arguments in the new operand
    // list.
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

      // Every written arg contributes a trailing tessera.call result, so
      // remember its pointer and pointee type for use after the call is
      // built. Whether it also keeps an operand slot depends on the
      // direction:
      //   - "out": the callee only writes it, so it is dropped from the
      //     operand list entirely. Supplying one would load the caller's
      //     not-yet-written storage and pass undef.
      //   - "inout": the callee reads the caller's existing object and writes
      //     back through the same pointer, so it falls through to the load
      //     below and keeps its operand slot carrying the incoming value.
      // Recorded in argument order either way, as CallOp's verifier expects.
      if (defineOp.argIsWritten(i)) {
        Type resultArgType = defineOp.getArgLiftedType(i);
        resultArgPtrs.push_back(operand);
        resultArgTypes.push_back(resultArgType);
        if (!defineOp.argIsRead(i))
          continue; // write-only: drop operand slot entirely
      }

      // If the callee takes this argument through a pointer, load the pointee
      // so the tessera.call carries a value; otherwise pass the pointer
      // through. newOperands.size() is the call-operand index this entry
      // lands at once pushed below, which is the index
      // getCallOperandPointeeType expects -- not the pre-strip loop index i.
      if (Type pointeeType =
              defineOp.getCallOperandPointeeType(newOperands.size())) {
        auto loadedVal = LLVM::LoadOp::create(rewriter, callOp.getLoc(),
                                              pointeeType, operand);
        newOperands.push_back(loadedVal);
      } else {
        newOperands.push_back(operand);
      }
    }

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
    llvm::DenseMap<unsigned, mlir::Type> argTypesByGlobalIndices;
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

      auto [it, inserted] =
          argTypesByGlobalIndices.try_emplace(idx, global.getType());
      if (!inserted) {
        llvm::errs()
            << "Tessera: found multiple globals matching argument type index "
            << idx << ":\n";
        return signalPassFailure();
      }
    }

    patterns.add<FuncOpRewrite>(ctx, argTypesByGlobalIndices);
    patterns.add<CallOpRewrite, ReturnOpRewrite>(ctx);

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
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
