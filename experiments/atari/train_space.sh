SEED=42
ENV="ALE/SpaceInvaders-v5"
TRAIN_MODES="--first-mode 0 --last-mode 9"
STEPS=1000000
# python run_experiments.py   --env $ENV  \
#                             --method_type FuseNetwMerge   \
#                             $TRAIN_MODES \
#                             --seed $SEED \
#                             --total_timesteps $STEPS \
#                             --alpha_learning_rate 2.5e-3 \
#                             --delta_theta_mode T \
#                             --pool_size 3 \
#                             --fuse_actor \
#                             --tag "Randn2.5e-3" \
# python run_experiments.py --env $ENV  --method_type Finetune  --total_timesteps $STEPS $TRAIN_MODES --seed $SEED
# python run_experiments.py   --env $ENV  \
#                             --method_type FuseNet   \
#                             $TRAIN_MODES \
#                             --seed $SEED \
#                             --total_timesteps $STEPS \
#                             --alpha_learning_rate 2.5e-1 \
#                             --delta_theta_mode T \
#                             --fuse_actor \
#                             --tag "FuseActorOnlyTRandn2.5e-1" \
# python run_experiments.py --env $ENV  --method_type Baseline  --total_timesteps $STEPS $TRAIN_MODES --seed $SEED
# python run_experiments.py --env $ENV  --method_type CompoNet  --total_timesteps $STEPS $TRAIN_MODES --seed $SEED
                            # --debug
# python run_experiments.py --env $ENV  --method_type PackNet   $TRAIN_MODES --seed $SEED
# python run_experiments.py --env $ENV  --method_type ProgNet   $TRAIN_MODES --seed $SEED
# python run_experiments.py --env $ENV  --method_type MaskNet  $TRAIN_MODES --seed $SEED --tag Baseline
python run_experiments.py --env $ENV  --method_type CbpNet  $TRAIN_MODES --seed $SEED --tag Baseline
