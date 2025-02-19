#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/enzyme_ad/jax/Dialect/Comm/Comm.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#define DEBUG_TYPE "enzyme"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_COMMLOWER
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace comm;
using namespace enzyme;

namespace {

// TODO find better tag than static global
static int tagCounter = 0;

std::string sendFuncName = "commSendF";
std::string recvFuncName = "commRecvF";

struct LowerMessageOpsPattern
    : public OpRewritePattern<comm::CommSimpleMessage> {

  LowerMessageOpsPattern(mlir::MLIRContext *ctx) : OpRewritePattern(ctx) {}

  static uint32_t getNewTag() { return tagCounter++; }

  static LogicalResult getStaticByteSize(mlir::Value v, unsigned &num_bytes) {
    if(TensorType tt = dyn_cast<TensorType>(v.getType())){
      if(tt.hasStaticShape()){
        unsigned numel = tt.getNumElements();
        unsigned bitwidth = tt.getElementTypeBitWidth();
        assert(bitwidth % 8 == 0 && "Currently support integer byte multiple bitwidths");
        num_bytes = numel * bitwidth / 8;
        return LogicalResult::success();
      }
    }
    return emitError(v.getLoc(), "Unsupported message type: currently only able to get message size for staticly sized tensors");
  }

  LogicalResult matchAndRewrite(CommSimpleMessage msg,
                                PatternRewriter &rewriter) const override {
    llvm::dbgs() << "Pattern called \n";
    // Collect the send, recvs of the message
    auto tok = msg.getToken();
    CommSend send;
    llvm::SmallVector<CommRecv> recvs;

    for (auto user : tok.getUsers()) {
      if (CommSend s = dyn_cast<CommSend>(user)) {
        if (send) {
          return s.emitOpError("Multiply send ops defined for same token!");
        }
        send = s;
      } else if (CommRecv r = dyn_cast<CommRecv>(user)) {
        recvs.push_back(r);
      } else {
        return user->emitOpError("not a handled case: was this supposed to be "
                                 "removed prior to lowering?");
      }
    }
    llvm::dbgs() << "C" << send << "\n";

    // Check that there is only one sending device
    auto send_devices = getOpDevices(*send);
    llvm::dbgs() << "C2\n";
    if (send_devices.size() != 1) {
      return send.emitOpError(
          "should be scheduled on exactly one device when lowering");
    }
    auto send_device = send_devices.front();
    rewriter.setInsertionPoint(msg.getParentSplit());
    mlir::Value send_device_val = rewriter.create<mlir::arith::ConstantIntOp>(send.getLoc(), send_device, 32);


    // Create a UID for the tag and other message attributes
    uint32_t tag = getNewTag();
    mlir::Value tag_val = rewriter.create<mlir::arith::ConstantIntOp>(msg.getLoc(), tag, 32);


    llvm::dbgs() << "D\n";

    unsigned message_size;
    LogicalResult try_get_size = getStaticByteSize(send.getData(), message_size);
    if(try_get_size.failed()) return try_get_size;
    mlir::Value message_size_val = rewriter.create<mlir::arith::ConstantIntOp>(send.getLoc(), message_size, 64);

    llvm::dbgs() << "E\n";

    for (CommRecv recv : recvs) {
      llvm::dbgs() << "I\n";
      // Likewise get single recieving device
      auto recv_devices = getOpDevices(*recv);
      if (recv_devices.size() != 1) {
        return recv.emitOpError(
            "should be scheduled on exactly one device when lowering");
      }
      auto recv_device = recv_devices.front();
      llvm::dbgs() << "F\n";
      
      rewriter.setInsertionPoint(msg.getParentSplit());
      mlir::Value recv_device_val = rewriter.create<mlir::arith::ConstantIntOp>(recv.getLoc(), recv_device, 32);
      llvm::dbgs() << "G\n";
      // Replace send with a JIT call to the send function for each receiver.
      // TODO: make this async, maybe make this one function instead of one per recv
      rewriter.setInsertionPoint(send);
      rewriter.create<enzymexla::JITCallOp>(
        send.getLoc(),
        (mlir::TypeRange){},
        mlir::FlatSymbolRefAttr::get(send.getContext(), sendFuncName),
        (mlir::ValueRange){recv_device_val, tag_val, send.getData(), message_size_val},
        nullptr, // Backend config (use default)
        nullptr, // Operand layouts
        nullptr, // result layouts
        nullptr  // output aliases
      );
      llvm::dbgs() << "H\n";

      // Create a recv call for the appropriate device
      llvm::dbgs() << "H2\n";
      rewriter.setInsertionPoint(recv);
      auto recv_call = rewriter.create<enzymexla::JITCallOp>(
        recv.getLoc(),
        recv->getResultTypes(),
        mlir::FlatSymbolRefAttr::get(send.getContext(), recvFuncName),
        (mlir::ValueRange){send_device_val, tag_val, message_size_val},
        nullptr, // Backend config (use default)
        nullptr, // Operand layouts
        nullptr, // result layouts
        nullptr  // output aliases
      );
      rewriter.replaceOp(recv, recv_call);

    }
    llvm::dbgs() << "J\n";
    llvm::dbgs() << *msg.getParentSplit()->getParentOp();

    rewriter.eraseOp(send);
    rewriter.eraseOp(msg);
    llvm::dbgs() << "K\n";

    return LogicalResult::success();
  }
};

struct CommLower : public enzyme::impl::CommLowerBase<CommLower> {

  /**
   * Reassigns each use of this multiplex's token to one of the contributing
   * tokens.
   *
   * TODO: this can potentially be a complex decision based on device load,
   * communication latency, potential for removing communcations/computations
   * outright, etc.
   */
  static void chooseMultiplexMapping(CommMultiplexMessage plex) {
    plex.getToken().replaceAllUsesWith(plex.getInTokens().front());
  }

  void runOnOperation() override {
    mlir::Operation *op = getOperation();

    mlir::RewritePatternSet patterns(op->getContext());
    llvm::dbgs() << "A\n";
    patterns.add<LowerMessageOpsPattern>(op->getContext());
    llvm::dbgs() << "B\n";
    FrozenRewritePatternSet frozen(std::move(patterns));
    llvm::dbgs() << "Calling apply patterns\n";
    (void)mlir::applyPatternsAndFoldGreedily(op, frozen);
    llvm::dbgs() << "\n\n final: \n" << *op;

  }
};

} // end anonymous namespace