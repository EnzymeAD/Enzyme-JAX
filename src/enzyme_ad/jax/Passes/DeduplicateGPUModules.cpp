//===- DeduplicateGPUModules.cpp - Deduplicate equivalent GPU modules -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

#include <cstdint>

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_DEDUPLICATEGPUMODULESPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

/// Returns true if any operation or block argument in `gpuModule` carries an
/// `LLVM::DIScopeAttr` location.
static bool hasDebugInfoScope(gpu::GPUModuleOp gpuModule) {
  WalkResult result = gpuModule.getBodyRegion().walk([](Operation *op) {
    if (op->getLoc()->findInstanceOf<FusedLocWith<LLVM::DIScopeAttr>>())
      return WalkResult::interrupt();
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (argument.getLoc()
                  ->findInstanceOf<FusedLocWith<LLVM::DIScopeAttr>>())
            return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// Returns true if `op` is a kernel entry function inside a `gpu.module`.
static bool isKernelFunction(Operation *op) {
  if (auto gpuFunc = dyn_cast<gpu::GPUFuncOp>(op))
    return gpuFunc.isKernel();
  if (auto llvmFunc = dyn_cast<LLVM::LLVMFuncOp>(op)) {
    if (llvmFunc.isExternal())
      return false;
    return llvmFunc->hasAttr(gpu::GPUDialect::getKernelFuncAttrName()) ||
           llvmFunc->hasAttr(NVVM::NVVMDialect::getKernelFuncAttrName()) ||
           llvmFunc->hasAttr(ROCDL::ROCDLDialect::getKernelFuncAttrName());
  }
  return false;
}

/// Returns true if `addrSpace` is a GPU shared/workgroup address space in NVVM
/// or ROCDL.
static bool isSharedMemoryAddressSpace(unsigned addrSpace) {
  return addrSpace == NVVM::NVVMMemorySpace::Shared ||
         addrSpace == ROCDL::ROCDLDialect::kSharedMemoryAddressSpace;
}

/// Returns true if `memRefType` resides in GPU shared/workgroup memory.
static bool isSharedMemoryMemRef(MemRefType memRefType) {
  if (gpu::GPUDialect::hasWorkgroupMemoryAddressSpace(memRefType))
    return true;
  Attribute memorySpace = memRefType.getMemorySpace();
  if (auto nvvmSpace =
          dyn_cast_if_present<NVVM::NVVMMemorySpaceAttr>(memorySpace))
    return nvvmSpace.getValue() == NVVM::NVVMMemorySpace::Shared;
  if (auto intSpace = dyn_cast_if_present<IntegerAttr>(memorySpace)) {
    int64_t space = intSpace.getInt();
    return space >= 0 &&
           isSharedMemoryAddressSpace(static_cast<unsigned>(space));
  }
  return false;
}

/// Returns true if `op`'s symbol name is local to its enclosing `gpu.module`
/// (or is a kernel entry whose host references are rewritten on deduplication)
/// and does not bind by name to external libraries or host shadow variables.
static bool canNormalizeInnerSymbol(Operation *op) {
  if (isKernelFunction(op))
    return true;

  if (auto gpuFunc = dyn_cast<gpu::GPUFuncOp>(op))
    return gpuFunc.isPrivate();
  if (auto funcOp = dyn_cast<func::FuncOp>(op))
    return !funcOp.isExternal() && funcOp.isPrivate();
  if (auto llvmFunc = dyn_cast<LLVM::LLVMFuncOp>(op)) {
    if (llvmFunc.isExternal())
      return false;
    return llvmFunc.getLinkage() == LLVM::Linkage::Internal ||
           llvmFunc.getLinkage() == LLVM::Linkage::Private;
  }

  // Conservatively normalize only kernel-local shared/workgroup memory globals
  // (such as pointer-named `@shared_mem_*` globals produced during parallel
  // lowering). Globals in other address spaces (e.g. device or constant
  // globals bound by name to host shadow variables) keep their exact names.
  if (auto llvmGlobal = dyn_cast<LLVM::GlobalOp>(op)) {
    if (!isSharedMemoryAddressSpace(llvmGlobal.getAddrSpace()))
      return false;
    return llvmGlobal.getLinkage() == LLVM::Linkage::Internal ||
           llvmGlobal.getLinkage() == LLVM::Linkage::Private;
  }
  if (auto memrefGlobal = dyn_cast<memref::GlobalOp>(op)) {
    if (memrefGlobal.isExternal())
      return false;
    return isSharedMemoryMemRef(memrefGlobal.getType());
  }

  return false;
}

/// Drops `polygeist.host_symbol` attributes on kernels inside `gpuModule` when
/// the referenced host stub function is no longer present in `hostSymbolTable`.
static void dropDeadHostSymbolAttrs(gpu::GPUModuleOp gpuModule,
                                    SymbolTable &hostSymbolTable) {
  for (Operation &op : gpuModule.getBody()->getOperations()) {
    auto hostSymbol =
        op.getAttrOfType<StringAttr>(enzyme::kPolygeistHostSymbolAttrName);
    if (!hostSymbol)
      continue;
    Operation *resolved = hostSymbolTable.lookup(hostSymbol);
    if (!isa_and_present<FunctionOpInterface>(resolved))
      op.removeAttr(enzyme::kPolygeistHostSymbolAttrName);
  }
}

/// Computes a structural hash for `rootOp` and its nested regions such that
/// `OperationEquivalence::isEquivalentTo(lhs, rhs, flags)` implies equal
/// hashes. `OperationEquivalence::computeHash` hashes only a single operation
/// without visiting its regions, blocks, or successors, so this walks the
/// nested IR to combine per-operation hashes.
static llvm::hash_code
computeOperationTreeHash(Operation *rootOp, OperationEquivalence::Flags flags) {
  // Pre-number all blocks and defined SSA values in lexical PreOrder before
  // hashing operations. CFG blocks in a region are stored in lexical rather
  // than dominance order, so an operation in an earlier lexical block can use
  // an SSA value defined in a later lexical block; pre-numbering ensures every
  // operand and branch target already has its positional ID when visited.
  llvm::DenseMap<Value, unsigned> valueIds;
  llvm::DenseMap<Block *, unsigned> blockIds;
  unsigned nextValueId = 0;
  unsigned nextBlockId = 0;

  rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (Value result : op->getResults())
      valueIds.try_emplace(result, ++nextValueId);
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        blockIds.try_emplace(&block, ++nextBlockId);
        for (BlockArgument argument : block.getArguments())
          valueIds.try_emplace(argument, ++nextValueId);
      }
    }
  });

  auto hashValue = [&](Value value) -> llvm::hash_code {
    auto it = valueIds.find(value);
    if (it != valueIds.end())
      return llvm::hash_value(it->second);
    return OperationEquivalence::directHashValue(value);
  };

  llvm::hash_code hash = {};
  rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    llvm::hash_code opHash =
        OperationEquivalence::computeHash(op, hashValue, hashValue, flags);
    for (Block *successor : op->getSuccessors())
      opHash = llvm::hash_combine(opHash, blockIds.lookup(successor));
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument argument : block.getArguments()) {
          opHash = llvm::hash_combine(opHash, argument.getType());
          if (!(flags & OperationEquivalence::IgnoreLocations))
            opHash = llvm::hash_combine(opHash, argument.getLoc());
        }
      }
    }
    hash = llvm::hash_combine(hash, opHash);
  });

  return hash;
}

/// Selects a symbol prefix disjoint from all symbols in `gpuModules` using
/// `SymbolTable::generateSymbolName`. `SymbolTable::renameToUnique` cannot pick
/// the target names directly because it retains each symbol's distinct original
/// name as a prefix, whereas structural comparison requires corresponding
/// symbols across modules to receive identical placeholder names.
static std::string
chooseDisjointSymbolPrefix(ArrayRef<gpu::GPUModuleOp> gpuModules) {
  unsigned uniquingCounter = 0;
  SmallString<64> prefix = SymbolTable::generateSymbolName<64>(
      "__enzymexla_dedup",
      [&](StringRef candidate) {
        return llvm::any_of(gpuModules, [&](gpu::GPUModuleOp gpuModule) {
          if (gpuModule.getName().starts_with(candidate))
            return true;
          return llvm::any_of(
              gpuModule.getBody()->getOperations(), [&](Operation &op) {
                auto symbolOp = dyn_cast<SymbolOpInterface>(op);
                return symbolOp && symbolOp.getName().starts_with(candidate);
              });
        });
      },
      uniquingCounter);
  prefix.push_back('_');
  return std::string(prefix);
}

namespace {

/// Original identity of an inner symbol renamed during normalization.
struct SavedSymbol {
  /// Inner symbol operation inside the GPU module.
  Operation *symbolOp;
  /// Symbol name prior to normalization.
  StringAttr originalName;
};

/// Metadata saved for a candidate `gpu.module` while its symbols are
/// normalized.
struct NormalizedModuleRecord {
  /// Candidate GPU module operation.
  gpu::GPUModuleOp gpuModule;
  /// Symbol table for `gpuModule`, reused for normalization and restoration.
  SymbolTable symbolTable;
  /// Original module symbol name.
  StringAttr originalModuleName;
  /// Inner symbols whose names were normalized, in module body order.
  SmallVector<SavedSymbol> savedSymbols;
};

/// Key for deduplicating normalized `gpu.module` operations in a `DenseMap`.
class ModuleEquivalenceKey {
public:
  ModuleEquivalenceKey(Operation *moduleOp, llvm::hash_code hash,
                       OperationEquivalence::Flags flags)
      : moduleOp(moduleOp), hash(hash), flags(flags) {
    assert(moduleOp && "expected non-null module operation");
  }

  /// Constructs a `DenseMap` empty or tombstone sentinel key.
  static ModuleEquivalenceKey createSentinel(Operation *sentinelOp) {
    return ModuleEquivalenceKey(sentinelOp);
  }

  Operation *getModuleOp() const { return moduleOp; }
  llvm::hash_code getHash() const { return hash; }
  OperationEquivalence::Flags getFlags() const { return flags; }

private:
  explicit ModuleEquivalenceKey(Operation *sentinelOp)
      : moduleOp(sentinelOp), hash{}, flags(OperationEquivalence::None) {}

  /// Normalized `gpu.module` operation, or a `DenseMap` pointer sentinel.
  Operation *moduleOp;
  /// Precomputed structural hash of `moduleOp`.
  llvm::hash_code hash;
  /// Equivalence flags used for both hashing and comparison.
  OperationEquivalence::Flags flags;
};

/// `DenseMapInfo` traits for `ModuleEquivalenceKey`.
/// Note: LLVM 24 switched `DenseMap` to sentinel-free occupancy bitmasks and
/// removed `getEmptyKey`/`getTombstoneKey` from `DenseMapInfo<T *>`, but older
/// LLVM revisions still invoke them; construct pointer sentinels using
/// `PointerLikeTypeTraits` alignment bits to match pointer `DenseMapInfo`.
struct ModuleEquivalenceInfo {
  static Operation *getEmptySentinel() {
    uintptr_t val = static_cast<uintptr_t>(-1);
    val <<= llvm::PointerLikeTypeTraits<Operation *>::NumLowBitsAvailable;
    return reinterpret_cast<Operation *>(val);
  }

  static Operation *getTombstoneSentinel() {
    uintptr_t val = static_cast<uintptr_t>(-2);
    val <<= llvm::PointerLikeTypeTraits<Operation *>::NumLowBitsAvailable;
    return reinterpret_cast<Operation *>(val);
  }

  static ModuleEquivalenceKey getEmptyKey() {
    return ModuleEquivalenceKey::createSentinel(getEmptySentinel());
  }

  static ModuleEquivalenceKey getTombstoneKey() {
    return ModuleEquivalenceKey::createSentinel(getTombstoneSentinel());
  }

  static unsigned getHashValue(const ModuleEquivalenceKey &key) {
    return static_cast<unsigned>(key.getHash());
  }

  static bool isEqual(const ModuleEquivalenceKey &lhs,
                      const ModuleEquivalenceKey &rhs) {
    Operation *lhsOp = lhs.getModuleOp();
    Operation *rhsOp = rhs.getModuleOp();
    if (lhsOp == rhsOp)
      return true;
    Operation *emptyKey = getEmptySentinel();
    Operation *tombstoneKey = getTombstoneSentinel();
    if (lhsOp == emptyKey || lhsOp == tombstoneKey || rhsOp == emptyKey ||
        rhsOp == tombstoneKey)
      return false;
    if (lhs.getHash() != rhs.getHash() || lhs.getFlags() != rhs.getFlags())
      return false;
    return OperationEquivalence::isEquivalentTo(lhsOp, rhsOp, lhs.getFlags());
  }
};

struct DeduplicateGPUModulesPass
    : public enzyme::impl::DeduplicateGPUModulesPassBase<
          DeduplicateGPUModulesPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // 1. Collect candidate `gpu.module` operations that have not yet generated
    // per-module host fatbin registration globals (`polygeist_stubs`).
    SmallVector<gpu::GPUModuleOp> candidateModules;
    for (gpu::GPUModuleOp gpuModule :
         moduleOp.getBody()->getOps<gpu::GPUModuleOp>()) {
      if (gpuModule->hasAttr(enzyme::kPolygeistStubsAttrName))
        continue;
      candidateModules.push_back(gpuModule);
    }

    if (candidateModules.size() < 2)
      return;

    // Collect all host-level symbol uses up front; bail out if `moduleOp`
    // contains an unregistered region-bearing op that could hide symbol uses.
    std::optional<SymbolTable::UseRange> hostSymbolUses =
        SymbolTable::getSymbolUses(&moduleOp.getBodyRegion());
    if (!hostSymbolUses)
      return;

    MLIRContext *context = &getContext();
    SymbolTable hostSymbolTable(moduleOp);
    std::string prefix = chooseDisjointSymbolPrefix(candidateModules);
    StringAttr normalizedModuleName =
        StringAttr::get(context, prefix + "module");

    llvm::DenseMap<ModuleEquivalenceKey, size_t, ModuleEquivalenceInfo>
        canonicalModuleIndexByKey;
    SmallVector<NormalizedModuleRecord> canonicalRecords;
    SmallVector<gpu::GPUModuleOp> duplicateModules;
    llvm::DenseMap<StringAttr, StringAttr> moduleRootReplacements;
    llvm::DenseMap<std::pair<StringAttr, StringAttr>, FlatSymbolRefAttr>
        nestedSymbolReplacements;

    // 2. Normalize local/kernel symbols inside each candidate module using
    // `SymbolTable::rename` and group structurally equivalent modules.
    for (gpu::GPUModuleOp gpuModule : candidateModules) {
      dropDeadHostSymbolAttrs(gpuModule, hostSymbolTable);

      NormalizedModuleRecord record{
          gpuModule, SymbolTable(gpuModule), gpuModule.getNameAttr(), {}};

      SmallVector<Operation *> normalizableOps;
      for (Operation &op : gpuModule.getBody()->getOperations()) {
        if (isa<SymbolOpInterface>(op) && canNormalizeInnerSymbol(&op))
          normalizableOps.push_back(&op);
      }

      bool normalizationFailed = false;
      for (auto [index, symbolOp] : llvm::enumerate(normalizableOps)) {
        StringAttr originalName = SymbolTable::getSymbolName(symbolOp);
        StringAttr normalizedName = StringAttr::get(
            context, llvm::formatv("{0}sym_{1}", prefix, index).sstr<64>());
        if (failed(record.symbolTable.rename(symbolOp, normalizedName))) {
          normalizationFailed = true;
          break;
        }
        record.savedSymbols.push_back({symbolOp, originalName});
      }

      // If `gpuModule` contains an unknown symbol table, roll back any partial
      // renames and leave this module untouched.
      if (normalizationFailed) {
        for (const SavedSymbol &saved : llvm::reverse(record.savedSymbols))
          (void)record.symbolTable.rename(saved.symbolOp, saved.originalName);
        continue;
      }

      gpuModule.setName(normalizedModuleName);

      OperationEquivalence::Flags flags =
          hasDebugInfoScope(gpuModule) ? OperationEquivalence::None
                                       : OperationEquivalence::IgnoreLocations;
      ModuleEquivalenceKey key(gpuModule.getOperation(),
                               computeOperationTreeHash(gpuModule, flags),
                               flags);

      auto [it, inserted] =
          canonicalModuleIndexByKey.try_emplace(key, canonicalRecords.size());
      if (inserted) {
        canonicalRecords.push_back(std::move(record));
        continue;
      }

      // Record symbol mappings from the duplicate module to its canonical
      // representative. Unnormalized inner symbols already share identical
      // names across equivalent modules, so only `savedSymbols` can differ.
      const NormalizedModuleRecord &canonicalRecord =
          canonicalRecords[it->second];
      moduleRootReplacements.try_emplace(record.originalModuleName,
                                         canonicalRecord.originalModuleName);
      for (auto [duplicateSaved, canonicalSaved] :
           llvm::zip_equal(record.savedSymbols, canonicalRecord.savedSymbols)) {
        if (duplicateSaved.originalName != canonicalSaved.originalName) {
          nestedSymbolReplacements.try_emplace(
              {record.originalModuleName, duplicateSaved.originalName},
              FlatSymbolRefAttr::get(canonicalSaved.originalName));
        }
      }
      duplicateModules.push_back(gpuModule);
    }

    // 3. Restore the original symbol names on all surviving canonical modules.
    for (NormalizedModuleRecord &record : canonicalRecords) {
      record.gpuModule.setName(record.originalModuleName);
      for (const SavedSymbol &saved : record.savedSymbols) {
        LogicalResult restored =
            record.symbolTable.rename(saved.symbolOp, saved.originalName);
        assert(succeeded(restored) &&
               "failed to restore original symbol name in canonical module");
        (void)restored;
      }
    }

    if (moduleRootReplacements.empty())
      return;

    // 4. Batch-rewrite host symbol references (such as `gpu.launch_func`,
    // `enzymexla.gpu_occupancy`, and `enzymexla.gpu_kernel_address`) on the
    // operations identified by `SymbolTable::getSymbolUses`, then erase the
    // duplicate modules.
    llvm::SmallPtrSet<Operation *, 32> duplicateModuleSet;
    for (gpu::GPUModuleOp duplicateModule : duplicateModules)
      duplicateModuleSet.insert(duplicateModule.getOperation());

    llvm::SmallPtrSet<Operation *, 32> hostUsersToUpdate;
    for (const SymbolTable::SymbolUse &symbolUse : *hostSymbolUses) {
      Operation *user = symbolUse.getUser();
      if (duplicateModuleSet.contains(user))
        continue;
      if (moduleRootReplacements.contains(
              symbolUse.getSymbolRef().getRootReference()))
        hostUsersToUpdate.insert(user);
    }

    AttrTypeReplacer hostReplacer;
    hostReplacer.addReplacement(
        [&](SymbolRefAttr symbolRef) -> std::pair<Attribute, WalkResult> {
          StringAttr root = symbolRef.getRootReference();
          auto rootIt = moduleRootReplacements.find(root);
          if (rootIt == moduleRootReplacements.end())
            return {symbolRef, WalkResult::skip()};

          ArrayRef<FlatSymbolRefAttr> nested = symbolRef.getNestedReferences();
          if (nested.empty())
            return {FlatSymbolRefAttr::get(rootIt->second), WalkResult::skip()};

          auto nestedIt =
              nestedSymbolReplacements.find({root, nested.front().getAttr()});
          if (nestedIt == nestedSymbolReplacements.end())
            return {SymbolRefAttr::get(rootIt->second, nested),
                    WalkResult::skip()};

          SmallVector<FlatSymbolRefAttr> updatedNested;
          updatedNested.reserve(nested.size());
          updatedNested.push_back(nestedIt->second);
          updatedNested.append(nested.begin() + 1, nested.end());
          return {SymbolRefAttr::get(rootIt->second, updatedNested),
                  WalkResult::skip()};
        });

    for (Operation *user : hostUsersToUpdate)
      hostReplacer.replaceElementsIn(user);

    for (gpu::GPUModuleOp duplicateModule : duplicateModules)
      duplicateModule.erase();
  }
};

} // namespace
