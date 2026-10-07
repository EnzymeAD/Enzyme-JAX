//===- CheckpointSchedule.h - Loop checkpointing requests -------*- C++ -*-===//
//
// The loop annotations of Enzyme's frontends (enzyme/checkpoint_schedule.h)
// as the attributes Enzyme-MLIR's LoopCheckpointing reads.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEXLA_CHECKPOINT_SCHEDULE_H
#define ENZYMEXLA_CHECKPOINT_SCHEDULE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include "enzyme/checkpoint_schedule.h"

namespace mlir {
namespace enzyme {

/// Asks for `loop` to be checkpointed with `schedule`, a schedule of
/// enzyme/checkpoint_schedule.h, keeping `budget` checkpoints (0 or less for
/// the schedule's default). Enzyme-MLIR has one binomial schedule, which
/// stands for Revolve too; storing every step is what differentiating the loop
/// without checkpointing does. False, and nothing set, for a schedule the
/// header does not name.
inline bool setCheckpointSchedule(Operation *loop, int64_t schedule,
                                  int64_t budget) {
  bool enable = false, binomial = false;
  switch (schedule) {
  case ENZYME_CKPT_SCHEDULE_NONE:
  case ENZYME_CKPT_SCHEDULE_STORE_ALL:
    break;
  case ENZYME_CKPT_SCHEDULE_PERIODIC:
    enable = true;
    break;
  case ENZYME_CKPT_SCHEDULE_REVOLVE:
  case ENZYME_CKPT_SCHEDULE_BINOMIAL:
    enable = binomial = true;
    break;
  default:
    return false;
  }
  Builder builder(loop->getContext());
  loop->setAttr("enzyme.enable_checkpointing", builder.getBoolAttr(enable));
  if (binomial)
    loop->setAttr("enzyme.binomial_checkpointing", builder.getUnitAttr());
  if (enable && budget > 0)
    loop->setAttr("enzyme.checkpoint_period",
                  builder.getI64IntegerAttr(budget));
  return true;
}

} // namespace enzyme
} // namespace mlir

#endif // ENZYMEXLA_CHECKPOINT_SCHEDULE_H
