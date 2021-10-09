import numpy as np
from ple import PLE
from ple.games.snake import Snake

agent = Snake(width=256, height=256)

env = PLE(agent, fps=15, force_fps=False, display_screen=True)

env.init()
actions = env.getActionSet()
loc_count = 0

q_state = {}
gamma = 0.5
learning_rate = 0.5

snake_state = agent.getGameState()
current_state = hash(str(snake_state))

for i in range(1000):
    if env.game_over():
        env.reset_game()
    else:
        action = actions[np.random.randint(0, len(actions))]
        reward = env.act(action)
        next_state_snake = agent.getGameState()
        next_state = hash(str(next_state_snake))

        if reward == -5:
            sample = -1
        elif reward == 1:
            sample = 1
        else:
            if (abs(snake_state["snake_head_x"] - snake_state["food_x"]) + abs(snake_state["snake_head_y"] - snake_state["food_y"])
                    > abs(next_state_snake["snake_head_x"] - next_state_snake["food_x"]) + abs(next_state_snake["snake_head_y"] - next_state_snake["food_y"])):
                sample = 1
            else:
                sample = -1

        if (current_state, action) not in q_state:
            loc_count += 1
            q_state[(current_state, action)] = 0.0

        best_action = actions[np.random.randint(0, len(actions))]
        maxy = 0
        if (next_state, best_action) in q_state:
            maxy = q_state[(next_state, best_action)]
        else:
            q_state[(next_state, best_action)] = 0.0

        for a in actions:
            if (next_state, a) in q_state:
                best = q_state[(next_state, a)]
                if best > maxy:
                    maxy = best
                    best_action = a

        # based on Reinforcement learning formula on p65
        q_state[(current_state, action)] += learning_rate * (
                sample + gamma * q_state[(next_state, best_action)] - q_state[(current_state, action)])

        current_state = next_state
        snake_state = next_state_snake

        print("iteration number : {} , Number of states added : {} ".format(i, loc_count))
