#include "mlir/Support/LogicalResult.h"
#include "src/enzyme_ad/jax/Dialect/Comm/CommOps.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::comm;

// Split
LogicalResult CommSplit::verify() {
  for(Operation &op : getDeclarations().getOps()){
    // Check that all ops are allowable as members
    if (!op.hasTrait<SplitMemberOp>()){
      return op.emitOpError("not allowed as immediate split op member");
    }

    // check that all branches have disjoint device sets
    DenseSet<unsigned> used_devices;
    for(CommBranch branch : getBranches()){
      for(unsigned device : branch.getDeviceIds()){
        if (used_devices.contains(device)){
          return branch.emitError("uses device already accounted for in same split");
        }
        used_devices.insert(device);
      }
    }
  }
  return success();
}
// CommSend
LogicalResult CommSend::verify(){
  auto op = getToken().getDefiningOp();
  if(!isa<CommSimpleMessage>(op)) return emitError("can only send to tokens from simple messages");
  return success();
}

CommSimpleMessage CommSend::getMessage(){
  return dyn_cast<CommSimpleMessage>(getToken().getDefiningOp());
}

// CommSimpleMessage
mlir::Type CommSimpleMessage::getInputType() {
  return getDataType();
}

mlir::Type CommSimpleMessage::getOutputType() {
  return getDataType();
}

// CommMultiplexMessage
LogicalResult CommMultiplexMessage::verify() {
  for (mlir::Value input_token : getInTokens()) { 
    auto input_op = input_token.getDefiningOp();
    if(CommMessage input_msg = dyn_cast<CommMessage>(input_op)){
      // check that the data types of the input message and this message match
      if(input_msg.getOutputType() != getDataType()){
        return emitError("includes message with return type different than declared");
      }
    } else {
      // TODO write verification to ensure all tokens are defined by messages only
      return input_op->emitError("message tokens should only be defined by message declarations");
    }
  }
  return success();
}

mlir::Type CommMultiplexMessage::getInputType() {
  // cannot send to a multiplex message!
  return NoneType();
}

mlir::Type CommMultiplexMessage::getOutputType() {
  return getDataType();
}

// Custom parsers

/**
 * Want to see either <<empty string>> or:
 *  `loop` $loop `reenter` $reentry `exit` $exit
 */
static ParseResult parseSplitRecurrenceRegions(OpAsmParser &p, Region &loop, Region &reentry, Region &exit){
  auto res = p.parseOptionalKeyword("loop");
  if(res.succeeded()) {
    // Keyword indicates presence of items in IR, which we need to (non-optionally) parse
    res = p.parseRegion(loop);            if(res.failed()) return failure();
    res = p.parseKeyword("reenter");      if(res.failed()) return failure();
    res = p.parseRegion(reentry);         if(res.failed()) return failure();
    res = p.parseKeyword("exit");         if(res.failed()) return failure();
    res = p.parseRegion(exit);            if(res.failed()) return failure();
  } else {
    // No keyword indicates no recurrence items, fill region defaults
    auto loop_block = new Block();
    loop.push_back(loop_block);
    auto reentry_block = new Block();
    reentry.push_back(reentry_block);
    auto exit_block = new Block();
    exit.push_back(exit_block);

    // Add the correct terminators to the parsed blocks'
    OpBuilder builder(p.getContext());
    mlir::Location loc(p.getEncodedSourceLoc(p.getCurrentLocation()));
    
    builder.setInsertionPointToEnd(loop_block);
    builder.create<CommContinue>(loc);

    builder.setInsertionPointToEnd(reentry_block);
    builder.create<CommContinue>(loc);

    builder.setInsertionPointToEnd(exit_block);
    builder.create<CommJoin>(loc);
  
    return success();
  }
}

/**
 * Just print everything- optionality is there for convenience and an empty region has same semantics as a missing one
 */
static void printSplitRecurrenceRegions(OpAsmPrinter &p, Operation* op, Region &loop, Region &reentry, Region &exit){
  p.printKeywordOrString("loop");
  p.printRegion(loop);
  p.printKeywordOrString("reenter");
  p.printRegion(reentry);
  p.printKeywordOrString("exit");
  p.printRegion(exit);
}

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/CommOps.cpp.inc"