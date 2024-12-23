SEED=42
ENV="ALE/SpaceInvaders-v5"
TRAIN_MODES="--first-mode 0 --last-mode 9"
STEPS=1400000
python run_experiments.py --env $ENV  --tag "WoResetActor" --method_type Finetune  --total_timesteps $STEPS $TRAIN_MODES --seed $SEED
# python run_experiments.py --env $ENV  --method_type CompoNet  $TRAIN_MODES --seed $SEED --debug
                            # --debug
# python run_experiments.py --env $ENV  --method_type PackNet   $TRAIN_MODES --seed $SEED
# python run_experiments.py --env $ENV  --method_type ProgNet   $TRAIN_MODES --seed $SEED
