//===----------------------------------------------------------------------===//
//
// This file extracts tessera_op and tessera_optimize global annotations
// and adds tessera_op attributes and tessera.optimization ops to the module.
//
//===----------------------------------------------------------------------===//

#include "Passes/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "llvm/ADT/SmallVector.h"
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

    DenseMap<StringRef, StringRef> functionToAnnotation;

    for (auto [structValue, funcName] : structToFunction) {
      if (structToAnnotation.count(structValue)) {
        StringRef annotStr = structToAnnotation[structValue];
        functionToAnnotation[funcName] = annotStr;
      }
    }

    // Apply annotations as attributes to functions
    for (auto [funcName, annotStr] : functionToAnnotation) {
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
      if (!func)
        continue;

      // Parse "tessera_op=string\0"
      StringRef annot(annotStr);
      if (annot.starts_with("tessera_op=")) {
        StringRef opName = annot.substr(11);
        opName = opName.take_while([](char c) { return c != '\0'; });
        func->setAttr("tessera_op",
                      StringAttr::get(func->getContext(), opName));
      }
    }
  }
};
} // namespace
