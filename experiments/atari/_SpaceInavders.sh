SEED=42
ENV="ALE/SpaceInvaders-v5"
TRAIN_MODES="--first-mode 4 --last-mode 9"
export CUDA_VISIBLE_DEVICES=1
# python run_experiments.py --env $ENV --method_type Baseline  $TRAIN_MODES --seed $SEED --tag main
# python run_experiments.py --env $ENV --method_type Finetune  $TRAIN_MODES --seed $SEED  --tag main
# python run_experiments.py --env $ENV --method_type ProgNet   $TRAIN_MODES --seed $SEED  --tag main
# python run_experiments.py --env $ENV --method_type PackNet   $TRAIN_MODES --seed $SEED  --tag main
# python run_experiments.py --env $ENV --method_type MaskNet  $TRAIN_MODES --seed $SEED  --tag main
# python run_experiments.py --env $ENV --method_type CReLUs  $TRAIN_MODES --seed $SEED  --tag main
# python run_experiments.py --env $ENV --method_type CompoNet  $TRAIN_MODES --seed $SEED  --tag main
# python run_experiments.py --env $ENV --method_type CbpNet  $TRAIN_MODES --seed $SEED  --tag main
# python run_experiments.py --env $ENV --method_type CKA-RL   $TRAIN_MODES --seed $SEED  --tag main