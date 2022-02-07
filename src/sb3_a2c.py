# sb3_a2c.py
from stable_baselines3 import A2C

model = A2C(
    "MlpPolicy",
    "CartPole-v0",
    verbose=0,
    device="cpu",
    seed=1,
)
model.learn(total_timesteps=3000)
for name, param in model.policy.named_parameters():
    if param.requires_grad:
        layer_param_sum = round(param.data.sum().item(), 4)
        print(f"{name}'s sum = {layer_param_sum}")
