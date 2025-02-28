#!/bin/bash

# Define arrays of parameters
CLASSIFIERS=("random_forest" "MLP" "GaussianNB" "CategoricalNB" "SVM" "DT" "KNN" "CN2" "countcp")
BENCHMARKS=("sudoku" "golomb" "exam_timetabling" "nurse_rostering")
NOISE_MODES=("fp" "fn")
NOISE_PERCENTAGES=("0" "0.05" "0.1" "0.15" "0.2")
TRAINING_SIZE=1

# Count total experiments
total=$((${#CLASSIFIERS[@]} * ${#BENCHMARKS[@]} * ${#NOISE_MODES[@]} * ${#NOISE_PERCENTAGES[@]}))
current=0

echo "Starting $total experiments..."

# Loop through all combinations
for classifier in "${CLASSIFIERS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        for noise_mode in "${NOISE_MODES[@]}"; do
            for noise_percentage in "${NOISE_PERCENTAGES[@]}"; do
                ((current++))
                echo "Running experiment $current/$total"
                echo "Classifier: $classifier"
                echo "Benchmark: $benchmark"
                echo "Noise mode: $noise_mode"
                echo "Noise percentage: $noise_percentage"
                echo "Training size: $TRAINING_SIZE"
                echo "----------------------------------------"
                
                python main.py -c "$classifier" -b "$benchmark" -n "$noise_mode" -p "$noise_percentage" -ts "$TRAINING_SIZE"
                
                echo "========================================="
            done
        done
    done
done

echo "All experiments completed!" 