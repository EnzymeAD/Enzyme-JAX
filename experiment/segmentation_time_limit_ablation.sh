#!/bin/bash

thresholds=(100 500 1000 5000)
time_limits=(10 50 100 500)
platforms=("cpu" "gpu")
models=("llama")

echo "Segmentation time limit ablation" > segmentation_time_limit.txt
echo "--------------------------" >> segmentation_time_limit.txt

run_experiment() {
  local model=$1
  local platform=$2
  local experiment_name=$3
  local command=$4

  echo "Running $experiment_name..." | tee -a segmentation_time_limit.txt
  START_TIME=$(date +%s)
  eval "$command" >> segmentation_time_limit.txt 2>&1
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  echo "$experiment_name: ${DURATION} seconds" >> segmentation_time_limit.txt
  echo "--------------------------------" >> segmentation_time_limit.txt
}

for model in "${models[@]}"; do
  for platform in "${platforms[@]}"; do
    for i in "${!thresholds[@]}"; do
      threshold=${thresholds[i]}
      time_limit=${time_limits[i]}
      
      export EXPERIMENT_NAME="${model}_tau=${threshold}_sat_time_limit=${time_limit}-${platform}"
      export SEGMENTATION_THRESHOLD=$threshold
      export SATURATION_TIME_LIMIT=$time_limit
      export EQSAT_PLATFORM=$platform

      if [ "$platform" == "gpu" ]; then
        COMMAND="CUDA_VISIBLE_DEVICES=2 python test/${model}.py"
      else
        COMMAND="JAX_PLATFORMS=cpu python test/${model}.py"
      fi

      run_experiment "$model" "$platform" "$EXPERIMENT_NAME" "$COMMAND"
    done
  done
done
