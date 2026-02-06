#include "Passes/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_TESSERAANNOTATIONTOATTRIBUTEPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

struct TesseraAnnotationToAttributePass
    : public enzyme::tessera::impl::TesseraAnnotationToAttributePassBase<
          TesseraAnnotationToAttributePass> {
  using TesseraAnnotationToAttributePassBase::
      TesseraAnnotationToAttributePassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Find string constants in metadata
    DenseMap<StringRef, std::string> stringGlobals;
    for (auto global : module.getOps<LLVM::GlobalOp>()) {
      if (global.getSection() && *global.getSection() == "llvm.metadata") {
        if (auto strAttr =
                dyn_cast_or_null<StringAttr>(global.getValueAttr())) {
          stringGlobals[global.getSymName()] = strAttr.getValue().str();
        }
      }
    }

    // Find annotations array
    LLVM::GlobalOp annotationGlobal = nullptr;
    for (auto global : module.getOps<LLVM::GlobalOp>()) {
      if (global.getSymName() == "llvm.global.annotations") {
        annotationGlobal = global;
        break;
      }
    }

    if (!annotationGlobal)
      return;

    Region &region = annotationGlobal.getInitializerRegion();
    if (region.empty())
      return;

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

    // Delete annotation array
    annotationGlobal.erase();
  }
};
} // namespace