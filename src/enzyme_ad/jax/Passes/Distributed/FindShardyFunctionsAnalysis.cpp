#include "src/enzyme_ad/jax/Passes/Distributed/FindShardyFunctionsAnalysis.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::enzyme::distributed {
#define GEN_PASS_DEF_PRINTSHARDYFUNCTIONNAMESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

static bool hasSdyPrefixedAttr(NamedAttribute attr) {
  return attr.getName().strref().starts_with("sdy.");
}

static bool hasSdyAttrs(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    if (hasSdyPrefixedAttr(attr)) {
      return true;
    }
  }
  return false;
}

static std::string stringifyAttribute(Attribute attr) {
  std::string printed;
  llvm::raw_string_ostream os(printed);
  os << attr;
  return printed;
}

static void insertUniqueMesh(llvm::SmallVectorImpl<sdy::MeshAttr> &out,
                             sdy::MeshAttr mesh) {
  if (llvm::is_contained(out, mesh)) {
    return;
  }
  out.push_back(std::move(mesh));
}

static LogicalResult
collectMeshFromSharding(sdy::TensorShardingAttr sharding, func::FuncOp funcOp,
                        const SymbolTable &symbolTable,
                        llvm::SmallVectorImpl<sdy::MeshAttr> &out) {
  sdy::MeshAttr mesh = sharding.getMesh(symbolTable);
  if (!mesh) {
    return funcOp.emitError()
           << "failed to resolve Shardy mesh for analysis from "
           << sharding.getMeshOrRef();
  }
  insertUniqueMesh(out, mesh);
  return success();
}

static LogicalResult
collectMeshesFromAttr(Attribute attr, func::FuncOp funcOp,
                      const SymbolTable &symbolTable,
                      llvm::SmallVectorImpl<sdy::MeshAttr> &out) {
  if (auto sharding = dyn_cast_or_null<sdy::TensorShardingAttr>(attr)) {
    return collectMeshFromSharding(sharding, funcOp, symbolTable, out);
  }
  if (auto shardings =
          dyn_cast_or_null<sdy::TensorShardingPerValueAttr>(attr)) {
    for (sdy::TensorShardingAttr sharding : shardings.getShardings()) {
      if (failed(collectMeshFromSharding(sharding, funcOp, symbolTable, out))) {
        return failure();
      }
    }
  }
  return success();
}

static FailureOr<llvm::SmallVector<sdy::MeshAttr>>
getShardyFunctionMeshes(func::FuncOp funcOp, const SymbolTable &symbolTable) {
  llvm::SmallVector<sdy::MeshAttr> meshes;

  for (NamedAttribute attr : funcOp->getAttrs()) {
    if (failed(collectMeshesFromAttr(attr.getValue(), funcOp, symbolTable,
                                     meshes))) {
      return failure();
    }
  }

  if (ArrayAttr argAttrs = funcOp.getArgAttrsAttr()) {
    for (Attribute attrsAttr : argAttrs) {
      auto attrs = dyn_cast_or_null<DictionaryAttr>(attrsAttr);
      if (!attrs) {
        continue;
      }
      for (NamedAttribute attr : attrs) {
        if (failed(collectMeshesFromAttr(attr.getValue(), funcOp, symbolTable,
                                         meshes))) {
          return failure();
        }
      }
    }
  }

  if (ArrayAttr resultAttrs = funcOp.getResAttrsAttr()) {
    for (Attribute attrsAttr : resultAttrs) {
      auto attrs = dyn_cast_or_null<DictionaryAttr>(attrsAttr);
      if (!attrs) {
        continue;
      }
      for (NamedAttribute attr : attrs) {
        if (failed(collectMeshesFromAttr(attr.getValue(), funcOp, symbolTable,
                                         meshes))) {
          return failure();
        }
      }
    }
  }

  WalkResult walkResult = funcOp.walk([&](Operation *op) {
    for (NamedAttribute attr : op->getAttrs()) {
      if (failed(collectMeshesFromAttr(attr.getValue(), funcOp, symbolTable,
                                       meshes))) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }

  return meshes;
}

static bool hasShardyFunctionInfo(func::FuncOp funcOp) {
  if (hasSdyAttrs(funcOp)) {
    return true;
  }

  if (ArrayAttr argAttrs = funcOp.getArgAttrsAttr()) {
    for (Attribute attrsAttr : argAttrs) {
      auto attrs = dyn_cast_or_null<DictionaryAttr>(attrsAttr);
      if (!attrs) {
        continue;
      }
      for (NamedAttribute attr : attrs) {
        if (hasSdyPrefixedAttr(attr)) {
          return true;
        }
      }
    }
  }

  if (ArrayAttr resultAttrs = funcOp.getResAttrsAttr()) {
    for (Attribute attrsAttr : resultAttrs) {
      auto attrs = dyn_cast_or_null<DictionaryAttr>(attrsAttr);
      if (!attrs) {
        continue;
      }
      for (NamedAttribute attr : attrs) {
        if (hasSdyPrefixedAttr(attr)) {
          return true;
        }
      }
    }
  }

  bool foundShardyBodyIR = false;
  funcOp.walk([&](Operation *op) {
    if (op == funcOp.getOperation()) {
      return WalkResult::advance();
    }
    if (op->getName().getStringRef().starts_with("sdy.") || hasSdyAttrs(op)) {
      foundShardyBodyIR = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (foundShardyBodyIR) {
    return true;
  }

  return false;
}

} // namespace

// Fragile, but works for now.
FindShardyFunctionsAnalysis::FindShardyFunctionsAnalysis(ModuleOp module) {
  SymbolTable symbolTable(module);

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    if (!hasShardyFunctionInfo(funcOp)) {
      continue;
    }

    FailureOr<llvm::SmallVector<sdy::MeshAttr>> meshes =
        getShardyFunctionMeshes(funcOp, symbolTable);
    if (failed(meshes)) {
      valid = false;
      return;
    }

    functionIndices[funcOp.getOperation()] = functions.size();
    functions.push_back(FindShardyFunctionsAnalysis::FunctionInfo{
        funcOp, funcOp.getSymNameAttr(), std::move(*meshes)});
  }
}

llvm::ArrayRef<sdy::MeshAttr>
FindShardyFunctionsAnalysis::getMeshes(func::FuncOp funcOp) const {
  auto it = functionIndices.find(funcOp.getOperation());
  if (it == functionIndices.end()) {
    return {};
  }
  return functions[it->second].meshes;
}

namespace {

// Testing / Debugging purposes.
struct PrintShardyFunctionNamesPass
    : public impl::PrintShardyFunctionNamesPassBase<
          PrintShardyFunctionNamesPass> {
  using PrintShardyFunctionNamesPassBase::PrintShardyFunctionNamesPassBase;

  void runOnOperation() override {
    const FindShardyFunctionsAnalysis &analysis =
        getAnalysis<FindShardyFunctionsAnalysis>();
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }

    for (const FindShardyFunctionsAnalysis::FunctionInfo &info :
         analysis.getShardyFunctions()) {
      llvm::outs() << info.symName.getValue();
      if (!info.meshes.empty()) {
        llvm::outs() << ": ";
        llvm::interleaveComma(info.meshes, llvm::outs(),
                              [&](sdy::MeshAttr mesh) {
                                llvm::outs() << stringifyAttribute(mesh);
                              });
      }
      llvm::outs() << "\n";
    }
  }
};

} // namespace
} // namespace mlir::enzyme::distributed
