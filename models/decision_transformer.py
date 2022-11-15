from transformers import DecisionTransformerModel
import torch as t
import gym

device = "cuda" if t.cuda.is_available() else "cpu"
model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
TARGET_RETURN = 1

model = model.to(device)
model.eval()

env = gym.make("Hopper-v3")
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Generate a new starting state.
state = env.reset()
states = t.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=t.float32)

actions = t.zeros((1, 1, act_dim),
                  device=device,
                  dtype=t.float32)

rewards = t.zeros((1, 1),
                  device=device,
                  dtype=t.float32)

target_return = t.tensor(TARGET_RETURN, dtype=t.float32).reshape(1, 1)
timesteps = t.tensor(0,
                     device=device,
                     dtype=t.long).reshape(1, 1)

attention_mask = t.zeros((1, 1),
                         device=device,
                         dtype=t.float32)

# Inference mode forward pass.
with t.no_grad():
    state_preds, action_preds, return_preds = model(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=target_return,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
    )

# Printing out the results from the forward pass.
    print(state_preds)
