
#include "AffineUtils.h"
#include "Passes.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"

#define DEBUG_TYPE "gpu-launch-recognition"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_GPULAUNCHRECOGNITION
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

constexpr char gpuModuleName[] = "__mlir_gpu_module";

constexpr char nvshmemCollectiveLaunchName[] = "nvshmemx_collective_launch";

/// True if `op` is an `nvshmemx_collective_launch` call taking `addrOf` as its
/// kernel operand.
static bool isCollectiveLaunchOf(Operation *op, LLVM::AddressOfOp addrOf) {
  auto call = dyn_cast<LLVM::CallOp>(op);
  if (!call || call.getCallee() != nvshmemCollectiveLaunchName)
    return false;
  return !call.getArgOperands().empty() &&
         call.getArgOperands()[0] == addrOf.getResult();
}

/// True if `func`'s address is handed to `nvshmemx_collective_launch` as the
/// kernel to run. Such a function is a host stub whose launch this pass
/// rewrites itself, rather than an address that escapes out of its reach.
static bool isCollectiveLaunchStub(LLVM::LLVMFuncOp func, Operation *module) {
  auto uses = func.getSymbolUses(module);
  if (!uses)
    return false;
  for (const SymbolTable::SymbolUse &use : *uses) {
    auto addrOf = dyn_cast<LLVM::AddressOfOp>(use.getUser());
    if (!addrOf)
      continue;
    for (Operation *user : addrOf->getUsers())
      if (isCollectiveLaunchOf(user, addrOf))
        return true;
  }
  return false;
}

struct GPULaunchRecognitionPass
    : public enzyme::impl::GPULaunchRecognitionBase<GPULaunchRecognitionPass> {
  using GPULaunchRecognitionBase::GPULaunchRecognitionBase;

  // The kernel function is inlined and erased before the launch is lowered, so
  // the target it was compiled for has to travel on the launch itself.
  static void copyGPUTargetAttrs(LLVM::LLVMFuncOp from, Operation *to) {
    if (auto passthrough = from.getPassthrough())
      to->setAttr("passthrough", *passthrough);
    if (auto features = from.getTargetFeatures())
      to->setAttr("target_features", *features);
    if (auto cpu = from.getTargetCpuAttr())
      to->setAttr("target_cpu", cpu);
  }

  // Reads the device architecture off a kernel function.
  std::pair<std::string, std::string> getGPUTarget(LLVM::LLVMFuncOp func) {
    std::string sm; // NVIDIA Streaming Multiprocessor (sm_80)
    // The importer lifts `target-cpu` out of the function's attribute bag into
    // a first-class attribute, so that is where the device IR's architecture
    // actually lands; only fall back to scanning `passthrough`.
    if (auto cpu = func.getTargetCpuAttr()) {
      sm = cpu.getValue().str();
    } else if (auto attr =
                   dyn_cast_or_null<ArrayAttr>(func.getPassthroughAttr())) {
      for (auto a : attr) {
        if (auto ar = dyn_cast<ArrayAttr>(a)) {
          if (ar.size() != 2)
            continue;
          auto s0 = dyn_cast<StringAttr>(ar[0]);
          auto s1 = dyn_cast<StringAttr>(ar[1]);
          if (!s0 || !s1)
            continue;
          if (s0.getValue() == "target-cpu")
            sm = s1.getValue();
        }
      }
    }

    std::string feat;
    if (auto attr = dyn_cast_or_null<LLVM::TargetFeaturesAttr>(
            func.getTargetFeaturesAttr())) {
      feat = attr.getFeaturesString();
    }

    return {sm, feat};
  }

  void initGPUModule(gpu::GPUModuleOp &gpuModule, LLVM::LLVMFuncOp func) {
    if (gpuModule)
      return;
    auto ctx = getOperation()->getContext();
    auto moduleBuilder =
        OpBuilder::atBlockBegin(cast<ModuleOp>(getOperation()).getBody());
    gpuModule = gpu::GPUModuleOp::create(
        moduleBuilder, getOperation()->getLoc(), gpuModuleName);

    auto [sm, feat] = getGPUTarget(func);

    Attribute target;
    if (backend == "rocm") {
      auto chip = "gfx1030";
      auto features = "+wavefrontsize64";

      target = moduleBuilder.getAttr<ROCDL::ROCDLTargetAttr>(
          /*optLevel=*/3, /*triple=*/"amdgcn-amd-amdhsa", chip, features,
          /*abiVersion=*/"600",
          /*flags=*/nullptr,
          /*linkLibs=*/nullptr);
    } else {
      // Default to CUDA/NVVM. A host function's target-cpu can leak in
      // here through an unresolved stub; never let a non-GPU chip through.
      auto chip = sm;
      if (!StringRef(chip).starts_with("sm_"))
        chip = "sm_80";
      auto features = feat;
      if (features.size() == 0)
        features = "+ptx73";
      target = moduleBuilder.getAttr<NVVM::NVVMTargetAttr>(
          /*optLevel=*/3, /*triple=*/"nvptx64-nvidia-cuda", chip, features,
          /*flags=*/nullptr,
          /*linkLibs=*/nullptr);
    }
    gpuModule.setTargetsAttr(moduleBuilder.getArrayAttr({target}));

    DataLayoutSpecInterface dataLayout = {};
    // Set index type size to 32 bits
    {
      llvm::DenseMap<mlir::TypeAttr, mlir::DataLayoutEntryInterface>
          typeEntries;
      auto type = IndexType::get(ctx);
      auto key = mlir::TypeAttr::get(type);
      uint64_t size = 32;
      auto params = IntegerAttr::get(mlir::IntegerType::get(ctx, 64), size);
      typeEntries.try_emplace(key, DataLayoutEntryAttr::get(type, params));
      SmallVector<DataLayoutEntryInterface> entries;
      entries.reserve(typeEntries.size());
      for (const auto &it : typeEntries)
        entries.push_back(it.second);
      dataLayout = DataLayoutSpecAttr::get(ctx, entries);
    }
    // gpuModule->setAttr(
    //     LLVM::LLVMDialect::getDataLayoutAttrName(),
    //     deviceModule->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName()));
    gpuModule->setAttr(DLTIDialect::kDataLayoutAttrName, dataLayout);
  }
  void replaceRuntime() {
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());
    StringSet<> seenErrors;
    // With exception handling preserved these runtime calls arrive in
    // invoke form; none of them throw, so turn each into a call plus a
    // branch to the normal destination so the rewrites below see them.
    {
      SmallVector<LLVM::InvokeOp> invokes;
      getOperation()->walk([&](LLVM::InvokeOp inv) {
        auto callee = inv.getCallee();
        if (!callee)
          return;
        for (StringRef name :
             {"cudaMalloc", "cudaFree",
              "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
              "cudaFuncGetAttributes", "cudaFuncSetCacheConfig", "cudaMemcpy",
              "cudaMemset", "cudaMemsetAsync", "cudaMemcpy2D"})
          if (*callee == name) {
            invokes.push_back(inv);
            return;
          }
      });
      for (auto inv : invokes) {
        OpBuilder builder(inv);
        auto call =
            LLVM::CallOp::create(builder, inv.getLoc(), inv.getResultTypes(),
                                 inv.getCalleeAttr(), inv.getCalleeOperands());
        inv->replaceAllUsesWith(call->getResults());
        LLVM::BrOp::create(builder, inv.getLoc(), inv.getNormalDestOperands(),
                           inv.getNormalDest());
        inv->erase();
      }
    }
    getOperation()->walk([&](LLVM::CallOp call) {
      auto callee = call.getCallee();
      OpBuilder builder(call);
      auto i8 = builder.getIntegerType(8);
      if (callee == "cudaMalloc") {

        Value arg = call->getOperand(1);
        if (!isa<IndexType>(arg.getType()))
          arg = arith::IndexCastOp::create(builder, call->getLoc(),
                                           builder.getIndexType(), arg);

        auto res = gpu::AllocOp::create(
            builder, call.getLoc(),
            MemRefType::get({ShapedType::kDynamic}, i8,
                            MemRefLayoutAttrInterface{},
                            builder.getI64IntegerAttr(1)),
            (mlir::Type) nullptr, ValueRange(), ValueRange(arg), ValueRange());
        auto ptr = enzymexla::Memref2PointerOp::create(
            builder, call.getLoc(),
            LLVM::LLVMPointerType::get(call.getContext()), res.getResult(0));
        LLVM::StoreOp::create(builder, call.getLoc(), ptr, call->getOperand(0));
        auto replace =
            LLVM::ZeroOp::create(builder, call.getLoc(), call.getType(0));
        call->replaceAllUsesWith(replace);
        call->erase();
        return;
      }

      if (callee == "cudaFree") {
        Value arg = call->getOperand(0);
        auto src = enzymexla::Pointer2MemrefOp::create(
            builder, call->getLoc(),
            MemRefType::get({ShapedType::kDynamic}, i8,
                            MemRefLayoutAttrInterface{},
                            builder.getI64IntegerAttr(1)),
            arg);
        gpu::DeallocOp::create(builder, call.getLoc(), (mlir::Type) nullptr,
                               ValueRange(), src);
        auto replace =
            LLVM::ZeroOp::create(builder, call.getLoc(), call.getType(0));
        call->replaceAllUsesWith(replace);
        call->erase();
        return;
      }

      if (callee == "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") {
        auto intType = call.getArgOperands()[2].getType();

        auto fnop = call.getArgOperands()[1].getDefiningOp<LLVM::AddressOfOp>();
        if (!fnop)
          return;

        auto curfn = fnop.getFunction(symbolTable);
        if (!curfn)
          return;

        auto repOp = enzymexla::GPUOccupancyOp::create(
            builder, call.getLoc(), intType,
            mlir::SymbolRefAttr::get(curfn.getContext(),
                                     curfn.getSymName().str()),
            call.getArgOperands()[2], call.getArgOperands()[3],
            call.getArgOperands()[4]);
        LLVM::StoreOp::create(builder, call.getLoc(), repOp->getResult(0),
                              call.getArgOperands()[0]);
        auto replace =
            LLVM::ZeroOp::create(builder, call.getLoc(), call.getType(0));
        call->replaceAllUsesWith(replace);
        call->erase();
        return;
      }

      /*
      if (callee == "cudaFuncGetAttributes" ||
          callee == "cudaFuncSetCacheConfig") {
        if (!seenErrors.count(*callee))
          call->emitWarning()
              << " Unsupported runtime function: " << *callee << "\n";
        seenErrors.insert(*callee);
        auto replace =
            LLVM::ZeroOp::create(builder, call.getLoc(), call.getType(0));
        call->replaceAllUsesWith(replace);
        call->erase();
        return;
      }
      */
      if (callee == "cudaMemcpy") {
        APInt directionA;

#if 0 
enum __device_builtin__ cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};
#endif
        if (matchPattern(call->getOperand(3), m_ConstantInt(&directionA))) {
          auto dst = call->getOperand(0);
          if (directionA == 0 || directionA == 2)
            dst = enzymexla::Pointer2MemrefOp::create(
                builder, call->getLoc(),
                MemRefType::get({ShapedType::kDynamic}, i8,
                                MemRefLayoutAttrInterface{}),
                dst);
          else
            dst = enzymexla::Pointer2MemrefOp::create(
                builder, call->getLoc(),
                MemRefType::get({ShapedType::kDynamic}, i8,
                                MemRefLayoutAttrInterface{},
                                builder.getI64IntegerAttr(1)),
                dst);

          auto src = call->getOperand(1);
          if (directionA == 0 || directionA == 1)
            src = enzymexla::Pointer2MemrefOp::create(
                builder, call->getLoc(),
                MemRefType::get({ShapedType::kDynamic}, i8,
                                MemRefLayoutAttrInterface{}),
                src);
          else
            src = enzymexla::Pointer2MemrefOp::create(
                builder, call->getLoc(),
                MemRefType::get({ShapedType::kDynamic}, i8,
                                MemRefLayoutAttrInterface{},
                                builder.getI64IntegerAttr(1)),
                src);

          Value arg = call->getOperand(2);
          if (!isa<IndexType>(arg.getType()))
            arg = arith::IndexCastOp::create(builder, call->getLoc(),
                                             builder.getIndexType(), arg);

          enzymexla::MemcpyOp::create(builder, call.getLoc(),
                                      (mlir::Type) nullptr, ValueRange(), dst,
                                      src, arg);
          auto replace =
              LLVM::ZeroOp::create(builder, call.getLoc(), call.getType(0));
          call->replaceAllUsesWith(replace);
          call->erase();
          return;
        }
      }

      if (callee == "cudaMemset" || callee == "cudaMemsetAsync") {
        Value devPtr = call->getOperand(0);
        devPtr = enzymexla::Pointer2MemrefOp::create(
            builder, call->getLoc(),
            MemRefType::get({ShapedType::kDynamic}, i8,
                            MemRefLayoutAttrInterface{},
                            builder.getI64IntegerAttr(1)),
            devPtr);

        Value value = call->getOperand(1);
        Value count = call->getOperand(2);
        if (!isa<IndexType>(count.getType()))
          count = arith::IndexCastOp::create(builder, call->getLoc(),
                                             builder.getIndexType(), count);

        SmallVector<Value, 1> asyncDeps;
        if (callee == "cudaMemsetAsync") {
          Value stream = call->getOperand(3);
          if (!stream.getDefiningOp<LLVM::ZeroOp>()) {
            auto token = enzymexla::StreamToTokenOp::create(
                builder, call.getLoc(),
                gpu::AsyncTokenType::get(call.getContext()), stream);
            asyncDeps.push_back(token.getResult());
          }
        }

        enzymexla::MemsetOp::create(builder, call.getLoc(),
                                    (mlir::Type) nullptr, ValueRange(asyncDeps),
                                    devPtr, value, count);
        auto replace =
            LLVM::ZeroOp::create(builder, call.getLoc(), call.getType(0));
        call->replaceAllUsesWith(replace);
        call->erase();
        return;
      }

      if (callee == "cudaMemcpy2D") {
        APInt directionA;
        if (matchPattern(call->getOperand(6), m_ConstantInt(&directionA))) {
          auto dst = call->getOperand(0);
          if (directionA == 0 || directionA == 2)
            dst = enzymexla::Pointer2MemrefOp::create(
                builder, call->getLoc(),
                MemRefType::get({ShapedType::kDynamic}, i8,
                                MemRefLayoutAttrInterface{}),
                dst);
          else
            dst = enzymexla::Pointer2MemrefOp::create(
                builder, call->getLoc(),
                MemRefType::get({ShapedType::kDynamic}, i8,
                                MemRefLayoutAttrInterface{},
                                builder.getI64IntegerAttr(1)),
                dst);

          Value dpitch = call->getOperand(1);
          if (!isa<IndexType>(dpitch.getType()))
            dpitch = arith::IndexCastOp::create(builder, call->getLoc(),
                                                builder.getIndexType(), dpitch);

          auto src = call->getOperand(2);
          if (directionA == 0 || directionA == 1)
            src = enzymexla::Pointer2MemrefOp::create(
                builder, call->getLoc(),
                MemRefType::get({ShapedType::kDynamic}, i8,
                                MemRefLayoutAttrInterface{}),
                src);
          else
            src = enzymexla::Pointer2MemrefOp::create(
                builder, call->getLoc(),
                MemRefType::get({ShapedType::kDynamic}, i8,
                                MemRefLayoutAttrInterface{},
                                builder.getI64IntegerAttr(1)),
                src);

          Value spitch = call->getOperand(3);
          if (!isa<IndexType>(spitch.getType()))
            spitch = arith::IndexCastOp::create(builder, call->getLoc(),
                                                builder.getIndexType(), spitch);

          Value width = call->getOperand(4);
          if (!isa<IndexType>(width.getType()))
            width = arith::IndexCastOp::create(builder, call->getLoc(),
                                               builder.getIndexType(), width);

          Value height = call->getOperand(5);
          if (!isa<IndexType>(height.getType()))
            height = arith::IndexCastOp::create(builder, call->getLoc(),
                                                builder.getIndexType(), height);

          enzymexla::Memcpy2DOp::create(builder, call.getLoc(),
                                        (mlir::Type) nullptr, ValueRange(), dst,
                                        dpitch, src, spitch, width, height);
          auto replace =
              LLVM::ZeroOp::create(builder, call.getLoc(), call.getType(0));
          call->replaceAllUsesWith(replace);
          call->erase();
          return;
        }
      }
    });
  }
  void runOnOperation() override {
    replaceRuntime();
    llvm::SmallVector<LLVM::LLVMFuncOp> launchFuncs;
    getOperation()->walk([&](LLVM::LLVMFuncOp funcOp) {
      auto symName = funcOp.getName();
      if (symName == "__mlir_cuda_caller_phase3")
        launchFuncs.push_back(funcOp);
    });

    auto ctx = getOperation()->getContext();

    gpu::GPUModuleOp gpuModule = nullptr;

    OpBuilder builder(getOperation()->getContext());

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());

    SetVector<Operation *> tocopy;

    DenseMap<LLVM::LLVMFuncOp, SmallVector<CallOpInterface>> kernelLaunches;

    // phase3 calls bypassed in favour of the collective launch that reaches
    // them. They still launch their kernel, so the capture scan below must not
    // read them as an address escaping out of this pass's reach.
    DenseSet<Operation *> stubLaunches;

    for (auto launchFunc : launchFuncs) {
      auto launchFuncUses = launchFunc.getSymbolUses(getOperation());
      for (auto use : *launchFuncUses) {
        if (auto cop = dyn_cast<CallOpInterface>(use.getUser())) {
          if (cop.getArgOperands().size() == 0)
            continue;
          auto argop =
              cop.getArgOperands()[0].getDefiningOp<LLVM::AddressOfOp>();
          if (!argop)
            continue;
          auto cur = argop.getFunction(symbolTable);
          if (!cur)
            continue;

          // A phase3 call inside a host stub that nvshmemx_collective_launch
          // takes by address is reached through the collective-launch walker
          // below; registering it here as well would launch the kernel twice.
          // The test is the collective launch specifically, not "address taken
          // and never called" -- a stub whose address escapes elsewhere is
          // still the launch site for its own phase3 call.
          if (auto parentFunc = cop->getParentOfType<LLVM::LLVMFuncOp>())
            if (isCollectiveLaunchStub(parentFunc, getOperation())) {
              stubLaunches.insert(cop.getOperation());
              continue;
            }

          kernelLaunches[cur].push_back(cop);
        }
      }
    }

    // Map of runtime function, index of the entry fn
    std::pair<const char *, int> runtime_fns[] = {
        {"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", 1},
        {"cudaFuncGetAttributes", 1},
        {"cudaFuncGetName", 1},
        {"cudaFuncSetAttribute", 0},
        {"cudaFuncSetCacheConfig", 0},
        {"cudaLaunchKernelExC", 1},
    };
    for (auto &pair : runtime_fns) {
      if (auto occupancy = symbolTable.getSymbolTable(getOperation())
                               .lookup<LLVM::LLVMFuncOp>(pair.first)) {
        auto launchFuncUses = occupancy.getSymbolUses(getOperation());
        for (auto use : *launchFuncUses) {
          if (auto cop = dyn_cast<CallOpInterface>(use.getUser())) {
            if (cop.getArgOperands().size() < pair.second + 1)
              continue;
            auto argop = cop.getArgOperands()[pair.second]
                             .getDefiningOp<LLVM::AddressOfOp>();
            if (!argop)
              continue;
            auto cur = argop.getFunction(symbolTable);
            if (!cur)
              continue;

            kernelLaunches[cur];
          }
        }
      }
    }

    // ── nvshmemx_collective_launch → kernelLaunches ──────────────────────
    //
    // The kernel this launches is two indirections away: the call takes the
    // address of clang's host stub, and the device stub is the first argument
    // of the phase3 call inside it.
    getOperation()->walk([&](LLVM::CallOp call) {
      if (call.getCallee() != nvshmemCollectiveLaunchName)
        return;
      // (kernel, gridXY, gridZ, blockXY, blockZ, args, sharedMem, stream);
      // anything else is a different function wearing the same name.
      if (call.getArgOperands().size() != 8)
        return;

      auto addrOf = call.getArgOperands()[0].getDefiningOp<LLVM::AddressOfOp>();
      if (!addrOf)
        return;
      auto hostStub = addrOf.getFunction(symbolTable);
      if (!hostStub)
        return;

      LLVM::LLVMFuncOp deviceFunc = nullptr;
      hostStub->walk([&](LLVM::CallOp inner) {
        if (inner.getCallee() != "__mlir_cuda_caller_phase3")
          return WalkResult::advance();
        if (inner.getArgOperands().empty())
          return WalkResult::advance();
        auto innerAddr =
            inner.getArgOperands()[0].getDefiningOp<LLVM::AddressOfOp>();
        if (!innerAddr)
          return WalkResult::advance();
        deviceFunc = innerAddr.getFunction(symbolTable);
        if (!deviceFunc)
          return WalkResult::advance();
        return WalkResult::interrupt();
      });

      if (!deviceFunc) {
        call.emitWarning()
            << "nvshmemx_collective_launch: could not trace device function "
               "through host stub '"
            << hostStub.getName() << "'";
        return;
      }

      kernelLaunches[deviceFunc].push_back(
          cast<CallOpInterface>(call.getOperation()));
    });

    SmallVector<Operation *> toErase;
    for (auto &launch : kernelLaunches) {
      bool captured = false;
      auto kernelUses = launch.first.getSymbolUses(getOperation());
      for (auto use : *kernelUses) {
        auto user = dyn_cast<LLVM::AddressOfOp>(use.getUser());
        if (!user) {
          captured = true;
          break;
        }
        for (auto user2 : user->getResult(0).getUsers()) {
          auto user3 = dyn_cast<CallOpInterface>(user2);
          if (!user3) {
            captured = true;
            break;
          }
          if (!llvm::is_contained(launch.second, user3) &&
              !stubLaunches.contains(user3.getOperation())) {
            captured = true;
            break;
          }
        }
      }
      // The scan above only sees uses of the device-side function. An address
      // user code takes is that of clang's host stub, and when it escapes --
      // handed to a call rather than straight to a runtime query -- nothing
      // rewrites it to the device symbol, so the kernel would never reach a
      // gpu.module and never be registered. An address-taken host stub is a
      // capture of its kernel.
      if (!captured) {
        StringRef hostStubName;
        if (auto attr = dyn_cast_or_null<ArrayAttr>(
                launch.first.getPassthroughAttr())) {
          for (auto a : attr) {
            auto ar = dyn_cast<ArrayAttr>(a);
            if (!ar || ar.size() != 2)
              continue;
            auto s0 = dyn_cast<StringAttr>(ar[0]);
            auto s1 = dyn_cast<StringAttr>(ar[1]);
            if (s0 && s1 && s0.getValue() == "polygeist.host_symbol")
              hostStubName = s1.getValue();
          }
        }
        if (!hostStubName.empty()) {
          if (auto hostStub = symbolTable.getSymbolTable(getOperation())
                                  .lookup<LLVM::LLVMFuncOp>(hostStubName)) {
            if (auto hostStubUses = hostStub.getSymbolUses(getOperation()))
              for (auto use : *hostStubUses) {
                auto addrOf = dyn_cast<LLVM::AddressOfOp>(use.getUser());
                if (!addrOf)
                  continue;
                // An address that only ever reaches
                // nvshmemx_collective_launch is the launch itself, rewritten
                // below, not an escape -- the same exemption the scan over the
                // device symbol's own uses already makes for its launches.
                if (llvm::all_of(addrOf->getUsers(), [&](Operation *user) {
                      return isCollectiveLaunchOf(user, addrOf);
                    }))
                  continue;
                captured = true;
                break;
              }
          }
        }
      }

      auto cur = launch.first;
      if (cur.isExternal())
        continue;
      gpu::GPUFuncOp gpufunc = nullptr;
      bool local_use_launch_func = use_launch_func || captured;
      if (local_use_launch_func) {

        FunctionType gpuTy0 = dyn_cast<FunctionType>(cur.getFunctionType());
        if (!gpuTy0) {
          if (auto lty =
                  dyn_cast<LLVM::LLVMFunctionType>(cur.getFunctionType())) {
            SmallVector<Type> restys;
            if (!isa<LLVM::LLVMVoidType>(lty.getReturnType()))
              restys.push_back(lty.getReturnType());

            gpuTy0 = builder.getFunctionType(lty.getParams(), restys);
          } else {
            cur.emitError("Require target operand to have functiontype or "
                          "llvmfunctiontype");
            continue;
          }
        }
        initGPUModule(gpuModule, launch.first);
        builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
        gpufunc = gpu::GPUFuncOp::create(builder, cur->getLoc(), cur.getName(),
                                         gpuTy0);
        {
          // The plugin records which host symbol each imported kernel was
          // registered for; carry it so the registration can bind the
          // address the program actually passes around instead of a
          // synthetic stub.
          StringRef host;
          if (auto attr =
                  dyn_cast_or_null<ArrayAttr>(cur.getPassthroughAttr())) {
            for (auto a : attr) {
              auto ar = dyn_cast<ArrayAttr>(a);
              if (!ar || ar.size() != 2)
                continue;
              auto s0 = dyn_cast<StringAttr>(ar[0]);
              auto s1 = dyn_cast<StringAttr>(ar[1]);
              if (s0 && s1 && s0.getValue() == "polygeist.host_symbol")
                host = s1.getValue();
            }
          }
          if (!host.empty())
            gpufunc->setAttr("polygeist.host_symbol",
                             builder.getStringAttr(host));
        }
        if (auto attrs = cur.getAllArgAttrs()) {
          gpufunc.setAllArgAttrs(attrs);
        }
        if (auto attrs = cur.getAllResultAttrs()) {
          gpufunc.setAllResultAttrs(attrs);
        }
        auto entry = &gpufunc.getBody().front();
        builder.setInsertionPointToEnd(entry);
        IRMapping map;
        gpufunc.getBody().getBlocks().clear();
        cur.getFunctionBody().cloneInto(&gpufunc.getBody(), map);

        if (auto comdat = cur.getComdat()) {
          cur.setComdatAttr({});
          auto comdatSelector =
              SymbolTable::lookupNearestSymbolFrom(cur, *comdat);
          if (auto cselect = dyn_cast<LLVM::ComdatSelectorOp>(comdatSelector)) {
            cselect->erase();
          }
        }

        gpufunc.setKernelAttr(builder.getUnitAttr());

        gpufunc->walk([](LLVM::ReturnOp op) {
          OpBuilder rewriter(op);
          gpu::ReturnOp::create(rewriter, op.getLoc());
          op.erase();
        });

        gpufunc->walk([](LLVM::UnreachableOp op) {
          OpBuilder rewriter(op);
          gpu::ReturnOp::create(rewriter, op.getLoc());
          op.erase();
        });

        gpufunc->walk([](func::ReturnOp op) {
          OpBuilder rewriter(op);
          gpu::ReturnOp::create(rewriter, op.getLoc());
          op.erase();
        });

        cur->walk([&](CallOpInterface cop) {
          if (auto op2 = cop.resolveCallable())
            tocopy.insert(op2);
        });
        cur->walk([&](LLVM::AddressOfOp cop) {
          if (auto op2 = cop.getGlobal(symbolTable))
            tocopy.insert(op2);
          else if (auto op2 = cop.getFunction(symbolTable))
            tocopy.insert(op2);
        });

        auto kernelSymbol =
            SymbolRefAttr::get(gpuModule.getNameAttr(),
                               {SymbolRefAttr::get(gpufunc.getNameAttr())});
        for (auto use : *kernelUses) {
          if (auto occ = dyn_cast<enzymexla::GPUOccupancyOp>(use.getUser())) {
            occ.setFnAttr(kernelSymbol);
            continue;
          }
          auto user = dyn_cast<LLVM::AddressOfOp>(use.getUser());
          if (!user) {
            llvm::errs()
                << " Error, could not replace kernel symbol in user(1): "
                << *use.getUser() << "\n";
            continue;
          }
          builder.setInsertionPoint(user);
          auto k2 = enzymexla::GPUKernelAddressOp::create(
              builder, user->getLoc(), user.getType(), kernelSymbol);
          for (auto &user2 :
               llvm::make_early_inc_range(user->getResult(0).getUses())) {
            auto user3 = dyn_cast<CallOpInterface>(user2.getOwner());
            if (user3 && llvm::is_contained(launch.second, user3)) {
              continue;
            }
            user2.assign(k2);
          }
          toErase.push_back(user);
        }
      }

      for (auto cop : launch.second) {
        gpu::LaunchFuncOp launchFuncOp = nullptr;
        auto loc = cop->getLoc();
        builder.setInsertionPointAfter(cop);

        // ── nvshmemx_collective_launch branch ──────────────────────────────
        if (auto llvmCall = dyn_cast<LLVM::CallOp>(cop.getOperation())) {
          if (llvmCall.getCallee() == nvshmemCollectiveLaunchName) {

            auto unpackDim =
                [&](Value packed_i64,
                    Value z_i32) -> std::tuple<Value, Value, Value> {
              Value c32 = arith::ConstantOp::create(
                  builder, loc, builder.getI64IntegerAttr(32));
              Value x_i32 = arith::TruncIOp::create(
                  builder, loc, builder.getI32Type(), packed_i64);
              Value y_i64 =
                  arith::ShRUIOp::create(builder, loc, packed_i64, c32);
              Value y_i32 = arith::TruncIOp::create(
                  builder, loc, builder.getI32Type(), y_i64);
              Value x = arith::IndexCastOp::create(
                  builder, loc, builder.getIndexType(), x_i32);
              Value y = arith::IndexCastOp::create(
                  builder, loc, builder.getIndexType(), y_i32);
              Value z = arith::IndexCastOp::create(
                  builder, loc, builder.getIndexType(), z_i32);
              return {x, y, z};
            };

            auto [gridX, gridY, gridZ] = unpackDim(
                llvmCall.getArgOperands()[1], llvmCall.getArgOperands()[2]);
            auto [blockX, blockY, blockZ] = unpackDim(
                llvmCall.getArgOperands()[3], llvmCall.getArgOperands()[4]);

            auto curFuncTy =
                dyn_cast<LLVM::LLVMFunctionType>(cur.getFunctionType());
            unsigned numParams = curFuncTy ? curFuncTy.getNumParams() : 0;

            Value argsArrayPtr = llvmCall.getArgOperands()[5];

            // nullptr is valid when the kernel has no parameters
            bool argsIsNull =
                argsArrayPtr.getDefiningOp<LLVM::ZeroOp>() != nullptr;

            SmallVector<Value> kernelArgs;

            if (argsIsNull) {
              if (numParams != 0) {
                llvmCall.emitError()
                    << "nvshmemx_collective_launch: void** args is null but "
                       "kernel expects "
                    << numParams << " parameter(s)";
                signalPassFailure();
                return;
              }
              // numParams == 0: kernelArgs stays empty.
            } else {
              auto argsAlloca = argsArrayPtr.getDefiningOp<LLVM::AllocaOp>();
              if (!argsAlloca) {
                llvmCall.emitError()
                    << "nvshmemx_collective_launch: void** args is not a "
                       "static alloca";
                signalPassFailure();
                return;
              }

              // Which slot of the args array a pointer into it refers to,
              // i.e. which kernel parameter a store through it writes. Clang
              // emits either a store straight to the alloca (slot 0) or a
              // getelementptr into it, so those are the shapes accepted here;
              // anything else is reported rather than guessed at.
              std::function<bool(Value, int64_t &)> getSlotIndex =
                  [&](Value ptr, int64_t &slot) -> bool {
                slot = 0;
                while (ptr != argsAlloca->getResult(0)) {
                  if (auto bitcast = ptr.getDefiningOp<LLVM::BitcastOp>()) {
                    ptr = bitcast.getArg();
                    continue;
                  }
                  if (auto addrcast =
                          ptr.getDefiningOp<LLVM::AddrSpaceCastOp>()) {
                    ptr = addrcast.getArg();
                    continue;
                  }
                  auto gep = ptr.getDefiningOp<LLVM::GEPOp>();
                  if (!gep)
                    return false;

                  SmallVector<int64_t> indices;
                  for (auto gepIdx : gep.getIndices()) {
                    if (auto attr = gepIdx.template dyn_cast<IntegerAttr>()) {
                      indices.push_back(attr.getValue().getSExtValue());
                      continue;
                    }
                    auto val = gepIdx.template dyn_cast<Value>();
                    APInt intVal;
                    if (!val || !matchPattern(val, m_ConstantInt(&intVal)))
                      return false;
                    indices.push_back(intVal.getSExtValue());
                  }

                  // `gep ptr, i` walks the slots directly; `gep [N x ptr], p,
                  // 0, i` walks into the array object. Both land on slot i,
                  // and nothing else indexes an array of void*.
                  if (indices.size() == 1 &&
                      isa<LLVM::LLVMPointerType>(gep.getElemType()))
                    slot += indices[0];
                  else if (indices.size() == 2 && indices[0] == 0 &&
                           isa<LLVM::LLVMArrayType>(gep.getElemType()))
                    slot += indices[1];
                  else
                    return false;

                  ptr = gep.getBase();
                }
                return true;
              };

              SmallVector<Value> slotValues(numParams, nullptr);
              bool slotFailed = false;
              bool zeroInit = false;

              auto recordSlot = [&](int64_t slot, Value val) {
                if (slotFailed)
                  return;
                // A slot written twice means the array is built up
                // conditionally, which this reconstruction cannot follow.
                if (slot < 0 || (unsigned)slot >= numParams || slotValues[slot])
                  slotFailed = true;
                else
                  slotValues[slot] = val;
              };

              // Every write into the array has to be accounted for, so an
              // unrecognised user is a failure rather than something to skip:
              // it may be the very store the launch arguments come from.
              std::function<void(Value)> collectWrites = [&](Value ptr) {
                for (Operation *user : ptr.getUsers()) {
                  if (auto memset = dyn_cast<LLVM::MemsetOp>(user)) {
                    APInt fillVal;
                    if (matchPattern(memset.getVal(),
                                     m_ConstantInt(&fillVal)) &&
                        fillVal.isZero())
                      zeroInit = true;
                    else
                      slotFailed = true;
                  } else if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
                    int64_t slot = 0;
                    // The array itself being stored somewhere is an escape,
                    // not a write into a slot.
                    if (store.getAddr() != ptr || !getSlotIndex(ptr, slot))
                      slotFailed = true;
                    else
                      recordSlot(slot, store.getValue());
                  } else if (isa<LLVM::GEPOp, LLVM::BitcastOp,
                                 LLVM::AddrSpaceCastOp>(user)) {
                    collectWrites(user->getResult(0));
                  } else if (auto call = dyn_cast<LLVM::CallOp>(user)) {
                    // A collective launch consumes the array, it does not
                    // write to it. Any other callee might.
                    if (call.getCallee() != nvshmemCollectiveLaunchName)
                      slotFailed = true;
                  } else if (!isa<LLVM::LoadOp, LLVM::LifetimeStartOp,
                                  LLVM::LifetimeEndOp>(user)) {
                    slotFailed = true;
                  }
                }
              };

              collectWrites(argsAlloca->getResult(0));

              // Applied after the walk, so it does not depend on where the
              // memset sits relative to the stores that overwrite it.
              if (zeroInit)
                for (unsigned i = 0; i < numParams; i++)
                  if (!slotValues[i])
                    slotValues[i] = LLVM::ZeroOp::create(
                        builder, loc, curFuncTy.getParamType(i));

              if (slotFailed) {
                llvmCall.emitError()
                    << "nvshmemx_collective_launch: could not statically "
                       "resolve args array";
                signalPassFailure();
                return;
              }

              // Build the final kernel argument list.
              for (unsigned i = 0; i < numParams; i++) {
                if (!slotValues[i]) {
                  llvmCall.emitError()
                      << "nvshmemx_collective_launch: missing args[" << i
                      << "]";
                  signalPassFailure();
                  return;
                }
                Type paramTy = curFuncTy.getParamType(i);
                if (slotValues[i].getType() == paramTy) {
                  kernelArgs.push_back(slotValues[i]);
                } else {
                  // Slot holds a void* pointing to the actual arg — load it.
                  auto slotMemref = enzymexla::Pointer2MemrefOp::create(
                      builder, loc, MemRefType::get({1}, paramTy),
                      slotValues[i]);
                  Value idx0 = arith::ConstantIndexOp::create(builder, loc, 0);
                  kernelArgs.push_back(memref::LoadOp::create(
                      builder, loc, slotMemref, ValueRange{idx0}));
                }
              }
            } // end else (!argsIsNull)

            Value shMem =
                arith::TruncIOp::create(builder, loc, builder.getI32Type(),
                                        llvmCall.getArgOperands()[6]);
            Value stream = llvmCall.getArgOperands()[7];

            Value result = llvmCall.getResult();
            Value zero = nullptr;
            if (result) {
              zero = LLVM::ConstantOp::create(
                  builder, loc, result.getType(),
                  builder.getIntegerAttr(result.getType(), 0));
            }

            if (local_use_launch_func) {
              if (stream.getDefiningOp<LLVM::ZeroOp>()) {
                launchFuncOp = gpu::LaunchFuncOp::create(
                    builder, loc, gpufunc, gpu::KernelDim3{gridX, gridY, gridZ},
                    gpu::KernelDim3{blockX, blockY, blockZ}, shMem,
                    ValueRange(kernelArgs));
              } else {
                assert(isa<LLVM::LLVMPointerType>(stream.getType()));
                Value token = enzymexla::StreamToTokenOp::create(
                    builder, loc, gpu::AsyncTokenType::get(ctx), stream);
                launchFuncOp = gpu::LaunchFuncOp::create(
                    builder, loc, gpufunc, gpu::KernelDim3{gridX, gridY, gridZ},
                    gpu::KernelDim3{blockX, blockY, blockZ}, shMem,
                    ValueRange(kernelArgs), token.getType(), ValueRange(token));
              }
            } else {
              if (stream.getDefiningOp<LLVM::ZeroOp>()) {
                auto op = mlir::gpu::LaunchOp::create(
                    builder, loc, gridX, gridY, gridZ, blockX, blockY, blockZ,
                    shMem, nullptr, ValueRange());
                builder.setInsertionPointToStart(&op.getRegion().front());
                LLVM::CallOp::create(builder, loc, cur, kernelArgs);
                gpu::TerminatorOp::create(builder, loc);
              } else {
                assert(isa<LLVM::LLVMPointerType>(stream.getType()));
                Value token = enzymexla::StreamToTokenOp::create(
                    builder, loc, gpu::AsyncTokenType::get(ctx), stream);
                auto op = mlir::gpu::LaunchOp::create(
                    builder, loc, gridX, gridY, gridZ, blockX, blockY, blockZ,
                    shMem, token.getType(), ValueRange(token));
                builder.setInsertionPointToStart(&op.getRegion().front());
                LLVM::CallOp::create(builder, loc, cur, kernelArgs);
                gpu::TerminatorOp::create(builder, loc);
              }
            }

            if (zero)
              result.replaceAllUsesWith(zero);
            cop->erase();
            continue;
          }
        }

        auto shMemSize = LLVM::TruncOp::create(
            builder, loc, builder.getI32Type(), cop.getArgOperands()[7]);
        auto stream = cop.getArgOperands()[8];
        llvm::SmallVector<mlir::Value> args;
        for (unsigned i = 9; i < cop.getArgOperands().size(); i++) {
          mlir::Value arg = cop.getArgOperands()[i];
          auto gpuTy0 = cur.getFunctionType();
          mlir::Type expectedTy;
          if (auto funcTy = dyn_cast<FunctionType>(gpuTy0)) {
            expectedTy = funcTy.getInput(i - 9);
          } else if (auto llvmFuncTy =
                         dyn_cast<LLVM::LLVMFunctionType>(gpuTy0)) {
            expectedTy = llvmFuncTy.getParamType(i - 9);
          } else {
            expectedTy =
                arg.getType(); // Should not happen given earlier checks
          }

          if (arg.getType() != expectedTy) {
            if (isa<LLVM::LLVMPointerType>(arg.getType()) &&
                isa<LLVM::LLVMPointerType>(expectedTy)) {
              arg =
                  LLVM::AddrSpaceCastOp::create(builder, loc, expectedTy, arg);
            } else if (arg.getType().isIntOrIndexOrFloat() &&
                       expectedTy.isIntOrIndexOrFloat() &&
                       arg.getType().getIntOrFloatBitWidth() ==
                           expectedTy.getIntOrFloatBitWidth()) {
              arg = LLVM::BitcastOp::create(builder, loc, expectedTy, arg);
            } else if (arg.getType().isIntOrIndex() &&
                       isa<LLVM::LLVMPointerType>(expectedTy)) {
              arg = LLVM::IntToPtrOp::create(builder, loc, expectedTy, arg);
            } else if (isa<LLVM::LLVMPointerType>(arg.getType()) &&
                       expectedTy.isIntOrIndex()) {
              arg = LLVM::PtrToIntOp::create(builder, loc, expectedTy, arg);
            } else {
              arg = LLVM::BitcastOp::create(builder, loc, expectedTy,
                                            arg); // Fallback
            }
          }
          args.push_back(arg);
        }

        Value grid[3];
        for (int i = 0; i < 3; i++) {
          if (local_use_launch_func)
            grid[i] = LLVM::SExtOp::create(builder, loc, builder.getI64Type(),
                                           cop.getArgOperands()[i + 1]);
          else
            grid[i] =
                arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                           cop.getArgOperands()[i + 1]);
        }
        Value block[3];
        for (int i = 0; i < 3; i++) {
          if (local_use_launch_func)
            block[i] = LLVM::SExtOp::create(builder, loc, builder.getI64Type(),
                                            cop.getArgOperands()[i + 4]);
          else
            block[i] =
                arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                           cop.getArgOperands()[i + 4]);
        }
        if (stream.getDefiningOp<LLVM::ZeroOp>()) {
          if (local_use_launch_func) {
            launchFuncOp = gpu::LaunchFuncOp::create(
                builder, loc, gpufunc,
                gpu::KernelDim3{grid[0], grid[1], grid[2]},
                gpu::KernelDim3{block[0], block[1], block[2]}, shMemSize,
                ValueRange(args));
          } else {
            auto op = mlir::gpu::LaunchOp::create(
                builder, launch.first->getLoc(), grid[0], grid[1], grid[2],
                block[0], block[1], block[2], shMemSize, nullptr, ValueRange());
            copyGPUTargetAttrs(cur, op);
            builder.setInsertionPointToStart(&op.getRegion().front());
            LLVM::CallOp::create(builder, loc, cur, args);
            gpu::TerminatorOp::create(builder, loc);
          }
        } else {
          if (local_use_launch_func) {
            assert(isa<LLVM::LLVMPointerType>(stream.getType()));
            // The stream-based async form: dependency operands without a
            // result token no longer verify, the stream rides the
            // asyncObject operand instead.
            launchFuncOp = gpu::LaunchFuncOp::create(
                builder, loc, gpufunc,
                gpu::KernelDim3{grid[0], grid[1], grid[2]},
                gpu::KernelDim3{block[0], block[1], block[2]}, shMemSize,
                ValueRange(args), /*asyncTokenType=*/nullptr,
                /*asyncDependencies=*/ValueRange(), /*asyncObject=*/stream);
          } else {
            assert(isa<LLVM::LLVMPointerType>(stream.getType()));
            stream = enzymexla::StreamToTokenOp::create(
                builder, loc, gpu::AsyncTokenType::get(ctx), stream);
            auto op = mlir::gpu::LaunchOp::create(
                builder, launch.first->getLoc(), grid[0], grid[1], grid[2],
                block[0], block[1], block[2], shMemSize, stream.getType(),
                ValueRange(stream));
            copyGPUTargetAttrs(cur, op);
            builder.setInsertionPointToStart(&op.getRegion().front());
            LLVM::CallOp::create(builder, loc, cur, args);
            gpu::TerminatorOp::create(builder, loc);
          }
        }
        if (launchFuncOp) {

          // A kernel with no argument attributes has no attribute list at
          // all, and the optional is empty rather than holding an empty array.
          SmallVector<Attribute> newArgAttrs;
          ArrayAttr curArgAttrs =
              cur.getArgAttrs().value_or(ArrayAttr::get(cur->getContext(), {}));
          for (auto [i, argAttrs] : llvm::enumerate(curArgAttrs)) {
            if (std::optional<NamedAttribute> attr =
                    cast<DictionaryAttr>(argAttrs).getNamed(
                        LLVM::LLVMDialect::getByValAttrName())) {
              newArgAttrs.push_back(
                  NamedAttrList(*attr).getDictionary(gpufunc->getContext()));
            } else {
              newArgAttrs.push_back(
                  NamedAttrList().getDictionary(gpufunc->getContext()));
            }
          }
          launchFuncOp->setAttr(
              "reactant.arg_attrs",
              ArrayAttr::get(gpufunc->getContext(), newArgAttrs));
        }
        cop->erase();
      }
    }

    if (gpuModule) {
      builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      llvm::SmallSet<Operation *, 1> done;
      while (tocopy.size()) {
        auto cur = tocopy.pop_back_val();
        if (done.count(cur))
          continue;
        done.insert(cur);
        auto cloned = builder.clone(*cur);
        if (auto glob = dyn_cast<LLVM::GlobalOp>(cur)) {
          if (auto comdat = glob.getComdat()) {
            glob.setComdatAttr({});

            auto comdatSelector =
                SymbolTable::lookupNearestSymbolFrom(cur, *comdat);
            if (auto cselect =
                    dyn_cast<LLVM::ComdatSelectorOp>(comdatSelector)) {
              cselect->erase();
            }

            cast<LLVM::GlobalOp>(cloned).setComdatAttr({});
          }
        }
        if (auto glob = dyn_cast<LLVM::LLVMFuncOp>(cur)) {
          if (auto comdat = glob.getComdat()) {
            glob.setComdatAttr({});

            auto comdatSelector =
                SymbolTable::lookupNearestSymbolFrom(cur, *comdat);
            if (auto cselect =
                    dyn_cast<LLVM::ComdatSelectorOp>(comdatSelector)) {
              cselect->erase();
            }

            cast<LLVM::LLVMFuncOp>(cloned).setComdatAttr({});
          }
        }
        cur->walk([&](CallOpInterface cop) {
          if (auto op2 = cop.resolveCallable())
            tocopy.insert(op2);
        });
        cur->walk([&](LLVM::AddressOfOp cop) {
          if (auto op2 = cop.getGlobal(symbolTable))
            tocopy.insert(op2);
          else if (auto op2 = cop.getFunction(symbolTable))
            tocopy.insert(op2);
        });
      }
    }

    for (auto e : toErase)
      e->erase();

    if (launchFuncs.size() && gpuModule)
      getOperation()->setAttr("gpu.container_module",
                              OpBuilder(ctx).getUnitAttr());
  }
};
