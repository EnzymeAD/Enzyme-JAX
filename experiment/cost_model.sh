#!/bin/bash

configs=(
  "export FUSION_COSTS=true"
  "export FUSION_COSTS=false"
  "export FUSION_COSTS=false ZERO_COSTS=false"
)
platforms=("cpu" "gpu")
models=("llama" "maxtext" "jaxmd")
filename=cost_model_$(date '+%Y-%m-%d_%H:%M:%S').txt

echo "Cost model ablation" > $filename
echo "--------------------------" >> $filename

for model in "${models[@]}"; do
  for platform in "${platforms[@]}"; do
    for config in "${configs[@]}"; do
      eval "$config"
      export EXPERIMENT_NAME="${model}_${config// /_}-${platform}"
      export KERAS_BACKEND="jax"
      export EQSAT_PLATFORM=$platform
      export EQSAT_ONLY=true

      if [ "$platform" == "gpu" ]; then
        COMMAND="CUDA_VISIBLE_DEVICES=2 python test/${model}.py"
      else
        COMMAND="JAX_PLATFORMS=cpu python test/${model}.py"
      fi

      # Time the command and redirect output to eqsat_vs_enzyme.txt
      echo "Running $EXPERIMENT_NAME..." | tee -a $filename
      START_TIME=$(date +%s)
      eval "$COMMAND" >> $filename 2>&1
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))

      # Append the timing to eqsat_vs_enzyme.txt
      echo "$EXPERIMENT_NAME: ${DURATION} seconds" >> $filename
      echo "--------------------------------" >> $filename
    done
  done
done
