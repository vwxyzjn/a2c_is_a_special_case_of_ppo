# sb3_ppo.py
import torch as th
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy", "CartPole-v0", verbose=0,
    device="cpu", seed=1,
    policy_kwargs=dict(
        optimizer_class=th.optim.RMSprop,
        optimizer_kwargs=dict(
            alpha=0.99, eps=1e-5, weight_decay=0,
        ),
    ),  # match A2C's optimizer settings
    learning_rate=7e-4,  # turn off learning rate anneling
    n_steps=5,  # match A2C's number of steps
    gae_lambda=1,  # disable GAE
    n_epochs=1,  # match PPO's and A2C's objective
    batch_size=5,  # perform update on the entire batch
    normalize_advantage=False,  # don't normalize advantages
    

    clip_range_vf=None,  # disable value function clipping

)
model.learn(total_timesteps=3000)
for name, param in model.policy.named_parameters():
    if param.requires_grad:
        layer_param_sum = round(param.data.sum().item(), 4)
        print(f"{name}'s sum = {layer_param_sum}")
