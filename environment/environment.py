import random
import pyspiel
import numpy as np

game = pyspiel.load_game("chess")

state = game.new_initial_state()

# Play the game.
while not state.is_terminal():
    current_player = state.current_player()
    legal_actions = state.legal_actions()
    legible_actions = [state.action_to_string(current_player, action) for action in legal_actions]
    print(f"{legible_actions=}")

    if state.is_chance_node():
        outcomes_with_probabilities = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes_with_probabilities)
        action = np.random.choice(action_list, p=prob_list) # Randomly sample an action according to probs.

        # Actually take that action.
        state.apply_action(action)
    else:
        action = legal_actions[0] # Hardcoded, but this is where your model chooses the action.
        print(f"Player: {current_player} is taking action: {legible_actions[0]}")
        state.apply_action(action)

print(f"Game finished! State: {state}")
print(f"Utility functions: {state.returns()}")
