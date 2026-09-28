//===----------------------------------------------------------------------===//
//
// This file extracts tessera_op, pure_tessera_op, tessera_optimize,
// tessera_guarantees and tessera_preserves global annotations and adds
// tessera_op / pure_tessera_op / tessera.guarantees / tessera.preserves
// attributes and tessera.optimization ops to the module.
//
//===----------------------------------------------------------------------===//

#include "Passes/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_LIFTTESSERAANNOTATIONSPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

// What the plugin's tessera::guarantees and tessera::preserves attributes
// become. Argument positions count as tessera_op argument lists do: `this`
// first for a member function, and no sret.
//
//   tessera_guarantees=SPD:return    the return value is SPD
//   tessera_guarantees=SPD:arg2      argument 2 is SPD once the call returns
//   tessera_preserves=SPD:0,1        the output is SPD if arguments 0 and 1 are
constexpr llvm::StringLiteral kGuaranteesPrefix = "tessera_guarantees=";
constexpr llvm::StringLiteral kPreservesPrefix = "tessera_preserves=";

bool isPropertyName(StringRef name) {
  return !name.empty() && llvm::all_of(name, [](char c) {
    return llvm::isAlnum(c) || c == '_';
  });
}

bool isOutputName(StringRef output) {
  unsigned index;
  return output == "return" ||
         (output.consume_front("arg") && !output.getAsInteger(10, index));
}

/// Parse the part after the prefix of a guarantees annotation into
/// {property, output}, or fail.
FailureOr<DictionaryAttr> parseGuarantee(StringRef text, MLIRContext *ctx) {
  auto [property, output] = text.split(':');
  if (!isPropertyName(property) || !isOutputName(output))
    return failure();
  Builder b(ctx);
  return b.getDictionaryAttr(
      {b.getNamedAttr("property", b.getStringAttr(property)),
       b.getNamedAttr("output", b.getStringAttr(output))});
}

/// Parse the part after the prefix of a preserves annotation into
/// {property, inputs}, or fail.
FailureOr<DictionaryAttr> parsePreserve(StringRef text, MLIRContext *ctx) {
  auto [property, list] = text.split(':');
  if (!isPropertyName(property) || list.empty())
    return failure();
  Builder b(ctx);
  SmallVector<Attribute> inputs;
  SmallVector<StringRef> parts;
  list.split(parts, ',');
  for (StringRef part : parts) {
    unsigned index;
    if (part.trim().getAsInteger(10, index))
      return failure();
    inputs.push_back(b.getI64IntegerAttr(index));
  }
  return b.getDictionaryAttr(
      {b.getNamedAttr("property", b.getStringAttr(property)),
       b.getNamedAttr("inputs", b.getArrayAttr(inputs))});
}

struct LiftTesseraAnnotationsPass
    : public enzyme::tessera::impl::LiftTesseraAnnotationsPassBase<
          LiftTesseraAnnotationsPass> {
  using LiftTesseraAnnotationsPassBase::LiftTesseraAnnotationsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    // Find string constants in metadata, locate annotations array, and build
    // optimization ops
    LLVM::GlobalOp annotationGlobal = nullptr;
    DenseMap<StringRef, std::string> stringGlobals;
    SmallVector<std::string> optimizationRules;

    for (auto global : module.getOps<LLVM::GlobalOp>()) {
      if (global.getSymName() == "llvm.global.annotations") {
        annotationGlobal = global;
      }
      if (global.getSection() && *global.getSection() == "llvm.metadata") {
        if (auto strAttr =
                dyn_cast_or_null<StringAttr>(global.getValueAttr())) {
          StringRef str = strAttr.getValue();
          stringGlobals[global.getSymName()] = str.str();
          if (str.starts_with("tessera_optimize=")) {
            StringRef rule =
                str.drop_front(StringRef("tessera_optimize=").size());
            if (rule.ends_with('\0'))
              rule = rule.drop_back(1);
            optimizationRules.push_back(rule.str());
          }
        }
      }
    }

    if (!annotationGlobal)
      return;

    Region &region = annotationGlobal.getInitializerRegion();
    if (region.empty())
      return;

    if (!optimizationRules.empty()) {
      OpBuilder builder(ctx);
      Location loc = builder.getUnknownLoc();
      builder.setInsertionPointToEnd(module.getBody());
      auto optimizationsOp = tessera::OptimizationsOp::create(builder, loc);
      Region &body = optimizationsOp.getBody();
      Block *block = builder.createBlock(&body);
      builder.setInsertionPointToStart(block);

      for (const std::string &rule : optimizationRules) {
        tessera::OptimizationOp::create(builder, loc,
                                        builder.getStringAttr(rule));
      }
    }

    DenseMap<Value, StringRef> valueToFunction;
    DenseMap<Value, StringRef> valueToAnnotation;

    // Find addressof operations
    for (Operation &op : region.front()) {
      if (auto addrOf = dyn_cast<LLVM::AddressOfOp>(&op)) {
        StringRef globalName = addrOf.getGlobalName();
        Value result = addrOf.getResult();

        if (module.lookupSymbol<LLVM::LLVMFuncOp>(globalName)) {
          valueToFunction[result] = globalName;
        } else if (stringGlobals.count(globalName)) {
          valueToAnnotation[result] = stringGlobals[globalName];
        }
      }
    }

    DenseMap<Value, StringRef> structToFunction;
    DenseMap<Value, StringRef> structToAnnotation;

    // Follow insertvalue chains to match functions with annotations
    for (Operation &op : region.front()) {
      if (auto insertValue = dyn_cast<LLVM::InsertValueOp>(&op)) {
        Value inserted = insertValue.getValue();
        Value container = insertValue.getContainer();
        Value result = insertValue.getResult();
        auto position = insertValue.getPosition();

        if (position.size() == 1) {
          if (position[0] == 0 && valueToFunction.count(inserted)) {
            structToFunction[result] = valueToFunction[inserted];
          } else if (position[0] == 1 && valueToAnnotation.count(inserted)) {
            structToAnnotation[result] = valueToAnnotation[inserted];
          }

          // Propagate information through the chain
          if (structToFunction.count(container)) {
            structToFunction[result] = structToFunction[container];
          }
          if (structToAnnotation.count(container)) {
            structToAnnotation[result] = structToAnnotation[container];
          }
        }
      }
    }

    // A function can carry more than one annotation -- a tessera op that is
    // also marked tessera_no_rewrite, say -- so keep all of them.
    DenseMap<StringRef, SmallVector<StringRef>> functionToAnnotations;

    for (auto [structValue, funcName] : structToFunction) {
      if (structToAnnotation.count(structValue)) {
        StringRef annotStr = structToAnnotation[structValue];
        functionToAnnotations[funcName].push_back(annotStr);
      }
    }

    // Apply annotations as attributes to functions
    for (auto &[funcName, annotStrs] : functionToAnnotations) {
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
      if (!func)
        continue;

      // The table is walked in no particular order, and clang can list an
      // annotation once per redeclaration, so settle both before the
      // annotations become ordered lists.
      llvm::sort(annotStrs);
      annotStrs.erase(llvm::unique(annotStrs), annotStrs.end());

      SmallVector<Attribute> guarantees, preserves;
      for (StringRef annot : annotStrs) {
        annot = annot.rtrim('\0');
        bool isGuarantee = annot.consume_front(kGuaranteesPrefix);
        if (!isGuarantee && !annot.consume_front(kPreservesPrefix))
          continue;
        FailureOr<DictionaryAttr> entry =
            isGuarantee ? parseGuarantee(annot, ctx) : parsePreserve(annot, ctx);
        if (failed(entry)) {
          func.emitError() << "malformed tessera "
                           << (isGuarantee ? "guarantees" : "preserves")
                           << " annotation '" << annot << "'";
          signalPassFailure();
          continue;
        }
        (isGuarantee ? guarantees : preserves).push_back(*entry);
      }
      if (!guarantees.empty())
        func->setAttr("tessera.guarantees", ArrayAttr::get(ctx, guarantees));
      if (!preserves.empty())
        func->setAttr("tessera.preserves", ArrayAttr::get(ctx, preserves));

      for (StringRef annot : annotStrs) {
        // Parse "tessera_op=string\0" and "pure_tessera_op=string\0"
        if (annot.consume_front("tessera_op=")) {
          annot = annot.rtrim('\0');
          func->setAttr("tessera_op",
                        StringAttr::get(func->getContext(), annot));
        } else if (annot.consume_front("pure_tessera_op=")) {
          annot = annot.rtrim('\0');
          func->setAttr("pure_tessera_op",
                        StringAttr::get(func->getContext(), annot));
        } else if (annot.rtrim('\0') == "tessera_no_rewrite") {
          // No rule may rewrite anything in this function's body. This is
          // how a replacement's fallback path reaches the op it replaces
          // without being rewritten back into a call to itself.
          //
          // It only protects a body that is still there: the pipeline runs
          // `inline` before this pass, so the function must also be
          // noinline, or its body is copied into callers where nothing marks
          // it.
          func->setAttr("tessera.no_rewrite",
                        UnitAttr::get(func->getContext()));
        }
      }
    }
  }
};
} // namespace
