#!/bin/bash

configs=(
  "export FUSION_COSTS=true"
  "export FUSION_COSTS=false"
  "export FUSION_COSTS=true ZERO_COSTS=false"
)
config_names=(
  "baseline"
  "no-fusion"
  "no-zero"
)
platforms=("cpu") # "gpu")
models=("llama" "nasrnn" "maxtext") # "jaxmd")
datetime=$(date '+%Y-%m-%d_%H:%M:%S')
filename=cost_model_$datetime.txt

export STATS_FILENAME=stats_cost_model_$datetime.csv
touch $STATS_FILENAME
echo "experiment_name,eqsat_time,segments" > $STATS_FILENAME

echo "Cost model ablation" > $filename
echo "--------------------------" >> $filename

for model in "${models[@]}"; do
  for platform in "${platforms[@]}"; do
    for i in "${!configs[@]}"; do
      config="${configs[$i]}"
      config_name="${config_names[$i]}"
      eval "$config"
      export EXPERIMENT_NAME="${model}_cost-model_${config_name}-${platform}_${datetime}"
      export KERAS_BACKEND="jax"
      export EQSAT_PLATFORM=$platform

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
