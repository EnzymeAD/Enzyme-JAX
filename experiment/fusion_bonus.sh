#!/bin/bash

platforms=("gpu")
fusion_bonuses=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)
models=("llama" "maxtext" "jaxmd")

echo "Fusion bonus" > fusion_bonus.txt
echo "--------------------------" >> fusion_bonus.txt

run_experiment() {
  local model=$1
  local platform=$2
  local experiment_name=$3
  local command=$4

  echo "Running $experiment_name..." | tee -a fusion_bonus.txt
  START_TIME=$(date +%s)
  eval "$command" >> fusion_bonus.txt 2>&1
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  echo "$experiment_name: ${DURATION} seconds" >> fusion_bonus.txt
  echo "--------------------------------" >> fusion_bonus.txt
}

for model in "${models[@]}"; do
  for platform in "${platforms[@]}"; do
    for i in "${!fusion_bonuses[@]}"; do
      fusion_bonus=${fusion_bonuses[i]}
      export EXPERIMENT_NAME="fusion_bonus=${fusion_bonus}-${platform}"
      export FUSION_BONUS_FRACTION_IN_OUTPUT=$fusion_bonus
      export EQSAT_PLATFORM=$platform

      if [ "$platform" == "gpu" ]; then
        COMMAND="python test/${model}.py"
      else
        echo "WARNING: THIS EXPERIMENT SHOULD NOT RUN WITH CPU"
      fi

      run_experiment "$model" "$platform" "$EXPERIMENT_NAME" "$COMMAND"
    done
  done
done
