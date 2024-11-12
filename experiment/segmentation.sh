#!/bin/bash

thresholds=(100 500 1000 5000)
platforms=("cpu" "gpu")
models=("llama" "maxtext" "jaxmd")

echo "Segmentation" > segmentation.txt
echo "--------------------------" >> segmentation.txt

run_experiment() {
  local model=$1
  local platform=$2
  local experiment_name=$3
  local command=$4

  echo "Running $experiment_name..." | tee -a segmentation.txt
  START_TIME=$(date +%s)
  eval "$command" >> segmentation.txt 2>&1
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  echo "$experiment_name: ${DURATION} seconds" >> segmentation.txt
  echo "--------------------------------" >> segmentation.txt
}

for model in "${models[@]}"; do
  for platform in "${platforms[@]}"; do
    for threshold in "${thresholds[@]}"; do
      export EXPERIMENT_NAME="${model}_tau=${threshold}-${platform}"
      export SEGMENTATION_THRESHOLD=$threshold
      export EQSAT_PLATFORM=$platform

      if [ "$platform" == "gpu" ]; then
        COMMAND="CUDA_VISIBLE_DEVICES=2 python test/${model}.py"
      else
        COMMAND="JAX_PLATFORMS=cpu python test/${model}.py"
      fi

      run_experiment "$model" "$platform" "$EXPERIMENT_NAME" "$COMMAND"
    done

    export SEGMENTATION_OFF=true
    export EXPERIMENT_NAME="${model}_no_segmentation-${platform}"

    run_experiment "$model" "$platform" "$EXPERIMENT_NAME" "$COMMAND"
    unset SEGMENTATION_OFF
  done
done
