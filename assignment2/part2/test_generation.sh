#!/bin/bash
# Test script for evaluating GPT model's knowledge on Grimm's Fairy Tales

MODEL_PATH="./logs/gpt-mini/version_0/checkpoints"
TOKENS=100
SAMPLES=3

echo "=========================================="
echo "GPT Generation Test - Grimm's Fairy Tales"
echo "=========================================="

# Test 1: Classic fairy tale openings
echo -e "\n>>> Test 1: Classic Opening (temperature=0.8)"
python generate.py --model_weights_folder $MODEL_PATH \
    --prompt "Once upon a time there was a" \
    --num_samples $SAMPLES --num_generated_tokens $TOKENS \
    --temperature 0.8 --top_p 0.9

# Test 2: Character prompts
echo -e "\n>>> Test 2: King Character (temperature=0.8)"
python generate.py --model_weights_folder $MODEL_PATH \
    --prompt "The king had three daughters" \
    --num_samples $SAMPLES --num_generated_tokens $TOKENS \
    --temperature 0.8 --top_p 0.9

# Test 3: Forest setting
echo -e "\n>>> Test 3: Forest Setting (temperature=0.8)"
python generate.py --model_weights_folder $MODEL_PATH \
    --prompt "In a great forest there lived" \
    --num_samples $SAMPLES --num_generated_tokens $TOKENS \
    --temperature 0.8 --top_p 0.9

# Test 4: Low temperature (more conservative)
echo -e "\n>>> Test 4: Low Temperature (temperature=0.5)"
python generate.py --model_weights_folder $MODEL_PATH \
    --prompt "The princess went into the" \
    --num_samples $SAMPLES --num_generated_tokens $TOKENS \
    --temperature 0.5 --top_p 0.9

# Test 5: High temperature (more creative)
echo -e "\n>>> Test 5: High Temperature (temperature=1.2)"
python generate.py --model_weights_folder $MODEL_PATH \
    --prompt "The princess went into the" \
    --num_samples $SAMPLES --num_generated_tokens $TOKENS \
    --temperature 1.2 --top_p 0.9

# Test 6: Witch character
echo -e "\n>>> Test 6: Witch Character (temperature=0.8)"
python generate.py --model_weights_folder $MODEL_PATH \
    --prompt "The old witch said" \
    --num_samples $SAMPLES --num_generated_tokens $TOKENS \
    --temperature 0.8 --top_p 0.9

# Test 7: Wolf character
echo -e "\n>>> Test 7: Wolf Character (temperature=0.8)"
python generate.py --model_weights_folder $MODEL_PATH \
    --prompt "The wolf" \
    --num_samples $SAMPLES --num_generated_tokens $TOKENS \
    --temperature 0.8 --top_p 0.9

# Test 8: Happy ending
echo -e "\n>>> Test 8: Happy Ending (temperature=0.8)"
python generate.py --model_weights_folder $MODEL_PATH \
    --prompt "And they lived happily" \
    --num_samples $SAMPLES --num_generated_tokens $TOKENS \
    --temperature 0.8 --top_p 0.9

# Test 9: Top-k sampling
echo -e "\n>>> Test 9: Top-k Sampling (k=50)"
python generate.py --model_weights_folder $MODEL_PATH \
    --prompt "Once upon a time" \
    --num_samples $SAMPLES --num_generated_tokens $TOKENS \
    --temperature 0.8 --top_k 50

# Test 10: Greedy decoding (deterministic)
echo -e "\n>>> Test 10: Greedy Decoding (no sampling)"
python generate.py --model_weights_folder $MODEL_PATH \
    --prompt "Once upon a time" \
    --num_samples 1 --num_generated_tokens $TOKENS \
    --do_sample False

echo -e "\n=========================================="
echo "Generation Tests Complete!"
echo "=========================================="
