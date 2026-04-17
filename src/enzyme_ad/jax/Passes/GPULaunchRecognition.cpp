
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

struct GPULaunchRecognitionPass
    : public enzyme::impl::GPULaunchRecognitionBase<GPULaunchRecognitionPass> {
  using GPULaunchRecognitionBase::GPULaunchRecognitionBase;

  void initGPUModule(gpu::GPUModuleOp &gpuModule, LLVM::LLVMFuncOp func) {
    if (gpuModule)
      return;
    auto ctx = getOperation()->getContext();
    auto moduleBuilder =
        OpBuilder::atBlockBegin(cast<ModuleOp>(getOperation()).getBody());
    gpuModule = gpu::GPUModuleOp::create(
        moduleBuilder, getOperation()->getLoc(), gpuModuleName);

    std::string sm; // NVIDIA Streaming Multiprocessor (sm_80)
    if (auto attr = dyn_cast_or_null<ArrayAttr>(func.getPassthroughAttr())) {
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
      // Default to CUDA/NVVM
      auto chip = sm;
      if (chip.size() == 0)
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

         // Skip phase3 calls whose enclosing host stub is never directly
          // called — it is only referenced via addressof for
          // nvshmemx_collective_launch. Those kernels are handled by the
          // collective-launch walker below and must not be double-registered.
          auto parentFunc = cop->getParentOfType<LLVM::LLVMFuncOp>();
          if (parentFunc) {
            // Only skip if parentFunc is a host stub — i.e. it has no direct
            // callers AND it IS referenced via addressof (as a launch function
            // pointer). Plain entry points like main also have no direct callers
            // but are never used via addressof, so they must not be skipped.
            bool hasDirectCaller = false;
            bool hasAddressOfUse = false;

            if (auto parentUses = parentFunc.getSymbolUses(getOperation())) {
              for (auto &puse : *parentUses) {
                if (isa<LLVM::CallOp>(puse.getUser()))
                  hasDirectCaller = true;
                if (isa<LLVM::AddressOfOp>(puse.getUser()))
                  hasAddressOfUse = true;
              }
            }

            // Skip only genuine host stubs: no direct calls, but IS used as a
            // function pointer (for nvshmemx_collective_launch or similar).
            if (!hasDirectCaller && hasAddressOfUse)
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
    getOperation()->walk([&](LLVM::CallOp call) {
      if (call.getCallee() != "nvshmemx_collective_launch") return;
      if (call.getArgOperands().empty()) return;

      auto addrOf = call.getArgOperands()[0]
                        .getDefiningOp<LLVM::AddressOfOp>();
      if (!addrOf) return;
      auto hostStub = addrOf.getFunction(symbolTable);
      if (!hostStub) return;

      LLVM::LLVMFuncOp deviceFunc = nullptr;
      hostStub->walk([&](LLVM::CallOp inner) {
        if (inner.getCallee() != "__mlir_cuda_caller_phase3")
          return WalkResult::advance();
        if (inner.getArgOperands().empty())
          return WalkResult::advance();
        auto innerAddr = inner.getArgOperands()[0]
                            .getDefiningOp<LLVM::AddressOfOp>();
        if (!innerAddr) return WalkResult::advance();
        deviceFunc = innerAddr.getFunction(symbolTable);
        return WalkResult::interrupt();
      });

      if (!deviceFunc) {
        call.emitWarning()
            << "nvshmemx_collective_launch: could not trace device function "
               "through host stub '" << hostStub.getName() << "'\n";
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
          if (!llvm::is_contained(launch.second, user3)) {
            captured = true;
            break;
          }
        }
      }
      auto cur = launch.first;
      gpu::GPUFuncOp gpufunc = nullptr;
      bool allLaunchesAreNvshmem = llvm::all_of(launch.second, [](CallOpInterface cop) {
          auto llvmCall = dyn_cast<LLVM::CallOp>(cop.getOperation());
          return llvmCall && llvmCall.getCallee() == "nvshmemx_collective_launch";
      });
      bool local_use_launch_func = allLaunchesAreNvshmem 
          ? use_launch_func 
          : (use_launch_func || captured);
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

        gpufunc->setAttr("gpu.kernel", builder.getUnitAttr());

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
          if (llvmCall.getCallee() == "nvshmemx_collective_launch") {

            auto unpackDim = [&](Value packed_i64, Value z_i32)
                -> std::tuple<Value, Value, Value> {
              Value c32 = LLVM::ConstantOp::create(builder, loc,
                              builder.getI64Type(),
                              builder.getI64IntegerAttr(32));
              Value x_i32 = arith::TruncIOp::create(builder, loc,
                                builder.getI32Type(), packed_i64);
              Value y_i64 = arith::ShRUIOp::create(builder, loc, packed_i64, c32);
              Value y_i32 = arith::TruncIOp::create(builder, loc,
                                builder.getI32Type(), y_i64);
              Value x = arith::IndexCastOp::create(builder, loc,
                            builder.getIndexType(), x_i32);
              Value y = arith::IndexCastOp::create(builder, loc,
                            builder.getIndexType(), y_i32);
              Value z = arith::IndexCastOp::create(builder, loc,
                            builder.getIndexType(), z_i32);
              return {x, y, z};
            };

            auto [gridX, gridY, gridZ] =
                unpackDim(llvmCall.getArgOperands()[1],
                          llvmCall.getArgOperands()[2]);
            auto [blockX, blockY, blockZ] =
                unpackDim(llvmCall.getArgOperands()[3],
                          llvmCall.getArgOperands()[4]);

            // ── Hoist numParams/curFuncTy so nullptr case can use them ──
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
                       "kernel expects " << numParams << " parameter(s)\n";
                signalPassFailure();
                return;
              }
              // numParams == 0: kernelArgs stays empty.
            } else {
              auto argsAlloca =
                  argsArrayPtr.getDefiningOp<LLVM::AllocaOp>();
              if (!argsAlloca) {
                llvmCall.emitError()
                    << "nvshmemx_collective_launch: void** args is not a "
                       "static alloca\n";
                signalPassFailure();
                return;
              }

              // ── Helpers ────────────────────────────────────────────────

              // Returns the byte size of an LLVM type on a 64-bit target.
              std::function<int64_t(Type)> getTypeSizeBytes =
                  [&](Type t) -> int64_t {
                if (auto intTy = dyn_cast<IntegerType>(t))
                  return intTy.getWidth() / 8;
                if (isa<LLVM::LLVMPointerType>(t))
                  return 8;
                if (auto arrTy = dyn_cast<LLVM::LLVMArrayType>(t))
                  return arrTy.getNumElements() *
                         getTypeSizeBytes(arrTy.getElementType());
                if (auto structTy = dyn_cast<LLVM::LLVMStructType>(t)) {
                  int64_t total = 0;
                  for (auto field : structTy.getBody()) {
                    int64_t sz = getTypeSizeBytes(field);
                    if (sz <= 0) return 0;
                    total += sz;
                  }
                  return total;
                }
                if (t.isF32()) return 4;
                if (t.isF64()) return 8;
                return 0;
              };

              // Walk backwards through transparent pointer ops to compute
              // the byte offset of ptr relative to argsAlloca->getResult(0).
             std::function<bool(Value, int64_t &)> getByteOffset =
                  [&](Value ptr, int64_t &byteOffset) -> bool {
                byteOffset = 0;
                while (ptr != argsAlloca->getResult(0)) {
                  if (auto bitcast = ptr.getDefiningOp<LLVM::BitcastOp>()) {
                    ptr = bitcast.getArg();
                  } else if (auto addrcast = ptr.getDefiningOp<LLVM::AddrSpaceCastOp>()) {
                    ptr = addrcast.getArg();
                  } else if (auto gep = ptr.getDefiningOp<LLVM::GEPOp>()) {
                    int64_t elemSize = getTypeSizeBytes(gep.getElemType());
                    if (elemSize <= 0)
                      return false;
                    int64_t gepOffset = 0;
                    for (auto gepIdx : gep.getIndices()) {
                      int64_t idxVal = 0;
                      bool found = false;
                      if (auto attr = gepIdx.template dyn_cast<IntegerAttr>()) {
                        idxVal = attr.getValue().getSExtValue();
                        found = true;
                      } else if (auto val = gepIdx.template dyn_cast<Value>()) {
                        APInt intVal;
                        if (matchPattern(val, m_ConstantInt(&intVal))) {
                          idxVal = intVal.getSExtValue();
                          found = true;
                        } else if (auto c = val.getDefiningOp<arith::ConstantIndexOp>()) {
                          idxVal = c.value();
                          found = true;
                        } else if (auto c = val.getDefiningOp<LLVM::ConstantOp>()) {  // <-- ADD THIS
                          if (auto intAttr = dyn_cast<IntegerAttr>(c.getValue())) {
                            idxVal = intAttr.getValue().getSExtValue();
                            found = true;
                          }
                        }
                      }
                      if (!found)
                        return false;
                      gepOffset += idxVal * elemSize;
                    }
                    byteOffset += gepOffset;
                    ptr = gep.getBase();
                  } else {
                    return false;
                  }
                }
                return true;
              };

              constexpr int64_t kPtrSize = 8;

              SmallVector<Value> slotValues(numParams, nullptr);
              bool slotFailed = false;

              auto recordSlot = [&](int64_t byteOff, Value val) {
                if (slotFailed) return;
                if ((byteOff % kPtrSize) != 0) {
                  slotFailed = true;
                  return;
                }
                int64_t idx = byteOff / kPtrSize;
                if (idx < 0 || (unsigned)idx >= numParams) {
                  slotFailed = true;
                  return;
                }
                if (slotValues[idx]) {
                  slotFailed = true;
                  return;
                }
                slotValues[idx] = val;
              };

              std::function<void(Value)> collectWrites = [&](Value ptr) {
                for (Operation *user : ptr.getUsers()) {
                  if (auto memset = dyn_cast<LLVM::MemsetOp>(user)) {
                    APInt fillVal;
                    if (matchPattern(memset.getVal(), m_ConstantInt(&fillVal)) &&
                        fillVal.isZero()) {
                      for (unsigned i = 0; i < numParams; i++)
                        if (!slotValues[i])
                          slotValues[i] = LLVM::ZeroOp::create(
                              builder, loc, curFuncTy.getParamType(i));
                    }
                  } else if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
                    int64_t byteOff = 0;
                    if (getByteOffset(store.getAddr(), byteOff))
                      recordSlot(byteOff, store.getValue());
                    else
                      slotFailed = true;
                  } else if (isa<LLVM::GEPOp, LLVM::BitcastOp,
                                LLVM::AddrSpaceCastOp>(user)) {
                    collectWrites(user->getResult(0));

                  // ── NEW: trace writes that flow through a function call ───────────
                  } else if (auto call = dyn_cast<LLVM::CallOp>(user)) {
                    // Skip the launch call itself and any other collective launch —
                    // they consume the args array, they don't write into it.
                    if (call == llvmCall) continue;
                    if (call.getCallee() == "nvshmemx_collective_launch") continue;

                    // Only handle direct calls to functions we can inspect.
                    std::optional<StringRef> calleeName = call.getCallee();
                    if (!calleeName) { slotFailed = true; continue; }
                    auto calleeFn = symbolTable.getSymbolTable(getOperation())
                                        .lookup<LLVM::LLVMFuncOp>(*calleeName);
                    if (!calleeFn || calleeFn.isExternal()) {
                      slotFailed = true;
                      continue;
                    }

                    // Byte offset of `ptr` from the top of argsAlloca.
                    int64_t ptrBaseOffset = 0;
                    if (!getByteOffset(ptr, ptrBaseOffset)) {
                      slotFailed = true;
                      continue;
                    }

                    // Find which argument positions carry our tracked ptr.
                    for (auto [argIdx, operand] :
                        llvm::enumerate(call.getArgOperands())) {
                      if (slotFailed) break;
                      if (operand != ptr) continue;
                      if ((unsigned)argIdx >=
                          calleeFn.getBody().front().getNumArguments()) {
                        slotFailed = true;
                        break;
                      }
                      BlockArgument calleeParam =
                          calleeFn.getBody().front().getArgument(argIdx);

                      // Walk callee looking for stores *through* calleeParam.
                      calleeFn->walk([&](LLVM::StoreOp store) {
                        if (slotFailed) return;

                        // Strip transparent casts from the store address.
                        Value addr = store.getAddr();
                        while (auto bc = addr.getDefiningOp<LLVM::BitcastOp>())
                          addr = bc.getArg();
                        while (auto ac = addr.getDefiningOp<LLVM::AddrSpaceCastOp>())
                          addr = ac.getArg();
                        if (addr != calleeParam) return; // not our slot

                        Value sv = store.getValue();

                        // Pattern A: store (load paramK), paramJ
                        //   → in caller: load(callerArgK) → slot at ptrBaseOffset
                        if (auto ld = sv.getDefiningOp<LLVM::LoadOp>()) {
                          Value lsrc = ld.getAddr();
                          while (auto bc = lsrc.getDefiningOp<LLVM::BitcastOp>())
                            lsrc = bc.getArg();
                          while (auto ac = lsrc.getDefiningOp<LLVM::AddrSpaceCastOp>())
                            lsrc = ac.getArg();
                          if (auto ba = dyn_cast<BlockArgument>(lsrc)) {
                            Value callerSrcPtr =
                                call.getArgOperands()[ba.getArgNumber()];
                            OpBuilder b(call); // insert before the call
                            Value callerVal = LLVM::LoadOp::create(
                                b, call.getLoc(), sv.getType(), callerSrcPtr);
                            recordSlot(ptrBaseOffset, callerVal);
                            return;
                          }
                        }

                        // Pattern B: store paramK, paramJ  (direct arg value)
                        //   → in caller: callerArgK → slot at ptrBaseOffset
                        if (auto ba = dyn_cast<BlockArgument>(sv)) {
                          recordSlot(ptrBaseOffset,
                                    call.getArgOperands()[ba.getArgNumber()]);
                          return;
                        }

                        slotFailed = true; // unrecognised pattern
                      });
                    }
                  }
                  // All other users (loads, lifetime markers, etc.) silently ignored.
                }
              };

              collectWrites(argsAlloca->getResult(0));   

              if (slotFailed) {
                llvmCall.emitError()
                    << "nvshmemx_collective_launch: could not statically "
                       "resolve args array\n";
                signalPassFailure();
                return;
              }

              // Build the final kernel argument list.
              for (unsigned i = 0; i < numParams; i++) {
                if (!slotValues[i]) {
                  llvmCall.emitError()
                      << "nvshmemx_collective_launch: missing args[" << i
                      << "]\n";
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
                  Value idx0 =
                      arith::ConstantIndexOp::create(builder, loc, 0);
                  kernelArgs.push_back(memref::LoadOp::create(
                      builder, loc, slotMemref, ValueRange{idx0}));
                }
              }
            } // end else (!argsIsNull)

            Value shMem = arith::TruncIOp::create(builder, loc,
                              builder.getI32Type(),
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
                    builder, loc, gpufunc,
                    gpu::KernelDim3{gridX, gridY, gridZ},
                    gpu::KernelDim3{blockX, blockY, blockZ},
                    shMem, ValueRange(kernelArgs));
              } else {
                assert(isa<LLVM::LLVMPointerType>(stream.getType()));
                Value token = enzymexla::StreamToTokenOp::create(
                    builder, loc, gpu::AsyncTokenType::get(ctx), stream);
                launchFuncOp = gpu::LaunchFuncOp::create(
                    builder, loc, gpufunc,
                    gpu::KernelDim3{gridX, gridY, gridZ},
                    gpu::KernelDim3{blockX, blockY, blockZ},
                    shMem, ValueRange(kernelArgs),
                    token.getType(), ValueRange(token));
              }
            } else {
              if (stream.getDefiningOp<LLVM::ZeroOp>()) {
                auto op = mlir::gpu::LaunchOp::create(
                    builder, loc, gridX, gridY, gridZ,
                    blockX, blockY, blockZ, shMem, nullptr, ValueRange());
                builder.setInsertionPointToStart(&op.getRegion().front());
                LLVM::CallOp::create(builder, loc, cur, kernelArgs);
                gpu::TerminatorOp::create(builder, loc);
              } else {
                assert(isa<LLVM::LLVMPointerType>(stream.getType()));
                Value token = enzymexla::StreamToTokenOp::create(
                    builder, loc, gpu::AsyncTokenType::get(ctx), stream);
                auto op = mlir::gpu::LaunchOp::create(
                    builder, loc, gridX, gridY, gridZ,
                    blockX, blockY, blockZ, shMem,
                    token.getType(), ValueRange(token));
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
        } // closes: if (auto llvmCall = dyn_cast<LLVM::CallOp>(...))

        // ── existing __mlir_cuda_caller_phase3 path (unchanged) ────────────

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
            expectedTy = arg.getType();
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
            } else {
              arg = LLVM::BitcastOp::create(builder, loc, expectedTy, arg);
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
            builder.setInsertionPointToStart(&op.getRegion().front());
            LLVM::CallOp::create(builder, loc, cur, args);
            gpu::TerminatorOp::create(builder, loc);
          }
        } else {
          if (local_use_launch_func) {
            assert(isa<LLVM::LLVMPointerType>(stream.getType()));
            stream = enzymexla::StreamToTokenOp::create(
                builder, loc, gpu::AsyncTokenType::get(ctx), stream);
            launchFuncOp = gpu::LaunchFuncOp::create(
                builder, loc, gpufunc,
                gpu::KernelDim3{grid[0], grid[1], grid[2]},
                gpu::KernelDim3{block[0], block[1], block[2]}, shMemSize,
                ValueRange(args), stream.getType(), ValueRange(stream));
          } else {
            assert(isa<LLVM::LLVMPointerType>(stream.getType()));
            stream = enzymexla::StreamToTokenOp::create(
                builder, loc, gpu::AsyncTokenType::get(ctx), stream);
            auto op = mlir::gpu::LaunchOp::create(
                builder, launch.first->getLoc(), grid[0], grid[1], grid[2],
                block[0], block[1], block[2], shMemSize, stream.getType(),
                ValueRange(stream));
            builder.setInsertionPointToStart(&op.getRegion().front());
            LLVM::CallOp::create(builder, loc, cur, args);
            gpu::TerminatorOp::create(builder, loc);
          }
        }
        if (launchFuncOp) {
          SmallVector<Attribute> newArgAttrs;
          for (auto [i, argAttrs] : llvm::enumerate(*cur.getArgAttrs())) {
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
