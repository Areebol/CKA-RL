export CUDA_VISIBLE_DEVICES=4

python run_ppo.py   --no-track                      \
                    --model-type=cnn-simple         \
                    --env-id=ALE/SpaceInvaders-v5   \
                    --seed=42                       \
                    --mode=0                        \
                    --save-dir=tmp                  \
                    --total-timesteps=1000          \
                    --wandb_mode=offline            \
                    --no-capture-video

