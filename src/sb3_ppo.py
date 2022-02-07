# sb3_ppo.py
import torch as th
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    "CartPole-v0",
    verbose=0,
    device="cpu",
    seed=1,
    # match A2C's parameters
    n_steps=5,
    ent_coef=0.0,
    learning_rate=7e-4,  # turn off learning rate anneling
    normalize_advantage=False,  # don't normalize advantages
    gae_lambda=1,  # disable GAE
    n_epochs=1,  # match PPO's and A2C's objective
    batch_size=5,  # perform update on the entire batch
    clip_range_vf=None,  # disable value function clipping
    policy_kwargs=dict(
        optimizer_class=th.optim.RMSprop,  # use RMSProp
        optimizer_kwargs=dict(
            alpha=0.99,
            eps=1e-5,
            weight_decay=0,
        ),  # use with A2C's optimizer settings
    ),
)
model.learn(total_timesteps=3000)
for name, param in model.policy.named_parameters():
    if param.requires_grad:
        layer_param_sum = round(param.data.sum().item(), 4)
        print(f"{name}'s sum = {layer_param_sum}")
