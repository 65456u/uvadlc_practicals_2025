#!/usr/bin/env bash
set -euo pipefail

# Automation for clean/FGSM/PGD eval experiments
# Results are appended to results_adv.txt

OUT_FILE="results_adv.txt"
BATCH_SIZE=64
VALID_RATIO=0.75
EPS_FGSM=0.1
ALPHA_FGSM=0.5
EPS_PGD=0.03
ALPHA_PGD=0.007
NITER_PGD=10

echo "======================================" | tee -a ${OUT_FILE}
echo "Experiment started at $(date)" | tee -a ${OUT_FILE}
echo "======================================" | tee -a ${OUT_FILE}

# ---------------------------------------------
# Experiment 1: Pretrained ResNet18, NO augmentations
# ---------------------------------------------
echo "" | tee -a ${OUT_FILE}
echo "=== Exp 1: Pretrained, No Augmentations ===" | tee -a ${OUT_FILE}
python run_adv_eval.py \
  --pretrained \
  --num_epochs 0 \
  --batch_size ${BATCH_SIZE} \
  --valid_ratio ${VALID_RATIO} \
  --epsilon_fgsm ${EPS_FGSM} \
  --alpha_fgsm ${ALPHA_FGSM} \
  --epsilon_pgd ${EPS_PGD} \
  --alpha_pgd ${ALPHA_PGD} \
  --num_iter_pgd ${NITER_PGD} \
  --out_file ${OUT_FILE}

# ---------------------------------------------
# Experiment 2: Pretrained ResNet18, WITH augmentations (fine-tune a few epochs)
# ---------------------------------------------
echo "" | tee -a ${OUT_FILE}
echo "=== Exp 2: Pretrained + Augmentations (fine-tune 5 epochs) ===" | tee -a ${OUT_FILE}
python run_adv_eval.py \
  --pretrained \
  --augmentations \
  --num_epochs 5 \
  --defense standard \
  --batch_size ${BATCH_SIZE} \
  --valid_ratio ${VALID_RATIO} \
  --epsilon_fgsm ${EPS_FGSM} \
  --alpha_fgsm ${ALPHA_FGSM} \
  --epsilon_pgd ${EPS_PGD} \
  --alpha_pgd ${ALPHA_PGD} \
  --num_iter_pgd ${NITER_PGD} \
  --out_file ${OUT_FILE}

# ---------------------------------------------
# Experiment 3: Pretrained + FGSM adversarial training
# ---------------------------------------------
echo "" | tee -a ${OUT_FILE}
echo "=== Exp 3: Pretrained + FGSM Defense (5 epochs) ===" | tee -a ${OUT_FILE}
python run_adv_eval.py \
  --pretrained \
  --num_epochs 5 \
  --defense fgsm \
  --batch_size ${BATCH_SIZE} \
  --valid_ratio ${VALID_RATIO} \
  --epsilon_fgsm ${EPS_FGSM} \
  --alpha_fgsm ${ALPHA_FGSM} \
  --epsilon_pgd ${EPS_PGD} \
  --alpha_pgd ${ALPHA_PGD} \
  --num_iter_pgd ${NITER_PGD} \
  --out_file ${OUT_FILE}

# ---------------------------------------------
# Experiment 4: Different epsilon values for FGSM
# ---------------------------------------------
for eps in 0.05 0.1 0.2 0.3; do
  echo "" | tee -a ${OUT_FILE}
  echo "=== Exp 4: Pretrained, epsilon_fgsm=${eps} ===" | tee -a ${OUT_FILE}
  python run_adv_eval.py \
    --pretrained \
    --num_epochs 0 \
    --batch_size ${BATCH_SIZE} \
    --valid_ratio ${VALID_RATIO} \
    --epsilon_fgsm ${eps} \
    --alpha_fgsm ${ALPHA_FGSM} \
    --epsilon_pgd ${EPS_PGD} \
    --alpha_pgd ${ALPHA_PGD} \
    --num_iter_pgd ${NITER_PGD} \
    --out_file ${OUT_FILE}
done

echo "" | tee -a ${OUT_FILE}
echo "======================================" | tee -a ${OUT_FILE}
echo "All experiments finished at $(date)" | tee -a ${OUT_FILE}
echo "Results saved to ${OUT_FILE}"