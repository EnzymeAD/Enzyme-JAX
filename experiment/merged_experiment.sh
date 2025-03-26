#!/bin/bash

# Define the combinations of environment variables
configs=(
  "export FUSION_COSTS=true ZERO_COSTS=true ENZYME_RULES=true MULTI_RULES=false EQSAT_RULES=false"
  "export FUSION_COSTS=true ZERO_COSTS=true ENZYME_RULES=true MULTI_RULES=false EQSAT_RULES=false"
  "export FUSION_COSTS=false ZERO_COSTS=true ENZYME_RULES=true MULTI_RULES=true EQSAT_RULES=true"
  "export FUSION_COSTS=true ZERO_COSTS=false ENZYME_RULES=true MULTI_RULES=true EQSAT_RULES=true"
  "export FUSION_COSTS=true ZERO_COSTS=false ENZYME_RULES=true MULTI_RULES=true EQSAT_RULES=true"
)
config_names=(
  "enzyme_ablation"
  "enzyme_ablation"
  "cost-model_no_fusion"
  "cost-model_no_zero"
  "cost-model_no_zero"
)
platforms=("cpu" "gpu" "gpu" "cpu" "gpu")
models=("bert" "gemma" "gpt2" "jaxmd" "kan1" "kan2" "llama" "maxtext" "nasrnn" "resnet" "searchlesschess" )
datetime=$(date '+%Y-%m-%d_%H:%M:%S')
filename=merged_$datetime.txt
num_repeats=12

export STATS_FILENAME=stats_merged_$datetime.csv
touch $STATS_FILENAME
echo "experiment_name,eqsat_time,segments" > $STATS_FILENAME

echo "Eqsat vs Enzyme AND cost model ablation" > $filename
echo "--------------------------" >> $filename


for repeat in $(seq 1 $num_repeats); do
    for model in "${models[@]}"; do
        for i in "${!configs[@]}"; do
            config="${configs[$i]}"
            platform="${platforms[$i]}"
            config_name="${config_names[$i]}"
            eval "$config"
            export KERAS_BACKEND="jax"
            export EQSAT_PLATFORM=$platform
            export LIMIT_RULES="true"
            export ILP_TIME_LIMIT=10
            export SATURATION_TIME_LIMIT=10

            if [ "$platform" == "gpu" ]; then
                COMMAND="python test/${model}.py"
            else
                COMMAND="JAX_PLATFORMS=cpu python test/${model}.py"
            fi

            export EXPERIMENT_NAME="${model}_${config_name}-${platform}_${datetime}_run${repeat}"
            echo "Running $EXPERIMENT_NAME (repeat $repeat/$num_repeats)..." | tee -a $filename

            START_TIME=$(date +%s)
            eval "$COMMAND" >> $filename 2>&1
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))

            echo "$EXPERIMENT_NAME: ${DURATION} seconds" >> $filename
            echo "--------------------------------" >> $filename
        done
    done
done
