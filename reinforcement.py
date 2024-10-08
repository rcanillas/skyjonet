import numpy as np
import matplotlib.pyplot as plt


## There's an issue with the algorithm, it's clearly not doing what it should.

ACTIONS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
MAZE_SIZE = 6


class Maze:
    def __init__(self) -> None:
        self.maze = np.zeros((MAZE_SIZE, MAZE_SIZE))
        self.maze[0, 0] = 2
        self.maze[5, :5] = 1
        self.maze[:4, 5] = 1
        self.maze[2, 2:] = 1
        self.maze[3, 2] = 1

        self.robot_position = (0, 0)  # current robot position
        self.steps = 0  # num steps of the robot
        self.allowed_states = None
        self.construct_allowed_states()

    def print_maze(self):
        for row in self.maze:
            print(" ".join([str(int(v)) for v in row]))

    def is_allowed_move(self, state, action):
        y, x = state
        y += ACTIONS[action][0]
        x += ACTIONS[action][1]
        # Moving off the board is not allowed
        if y < 0 or x < 0 or y > MAZE_SIZE - 1 or x > MAZE_SIZE - 1:
            return False
        # Moving to empty place or staying in the same position is allowed
        if self.maze[y, x] == 0 or self.maze[y, x] == 2:
            return True
        else:
            return False

    def construct_allowed_states(self):
        allowed_states = {}
        for y, row in enumerate(self.maze):
            for x, col in enumerate(row):
                # iterate through all valid spaces
                if self.maze[(y, x)] != 1:
                    allowed_states[(y, x)] = []
                    for action in ACTIONS:
                        if self.is_allowed_move((y, x), action):
                            allowed_states[(y, x)].append(action)
        self.allowed_states = allowed_states

    def update_maze(self, action):
        y, x = self.robot_position
        self.maze[y, x] = 0  # set the current position to empty
        y += ACTIONS[action][0]
        x += ACTIONS[action][1]
        self.robot_position = (y, x)
        self.maze[y, x] = 2
        self.steps += 1

    def is_game_over(self):
        if self.robot_position == (MAZE_SIZE - 1, MAZE_SIZE - 1):
            return True
        return False

    def give_reward(self):
        if self.robot_position == (MAZE_SIZE - 1, MAZE_SIZE - 1):
            return 0
        else:
            return -1

    def get_state_and_reward(self):
        return self.robot_position, self.give_reward()


class Agent:
    def __init__(self, states, alpha=0.15, random_factor=0.2) -> None:
        self.state_history = [((0, 0), 0)]  # state, reward
        self.alpha = alpha
        self.random_factor = random_factor

        # start the rewards table
        self.G = {}
        self.init_reward(states)

    def init_reward(self, states):
        for i, row in enumerate(states):
            for j, cl in enumerate(row):
                self.G[(j, i)] = np.random.uniform(high=1.0, low=0.1)

    def update_state_history(self, state, reward):
        self.state_history.append((state, reward))

    def learn(self):
        target = 0  # we know the "ideal" reward
        a = self.alpha

        for state, reward in reversed(self.state_history):
            self.G[state] = self.G[state] + a * (target - self.G[state])

        self.state_history = []  # reset the state_history
        self.random_factor = -10e-5  # decrease random_factor

    def choose_action(self, state, allowed_moves):
        next_move = None
        n = np.random.random()
        # print("exploration") if n < self.random_factor else print("exploitation")
        # print(state)
        # print(allowed_moves)
        # print(
        #   [
        #       self.G[tuple([sum(x) for x in zip(state, ACTIONS[a])])]
        #       for a in allowed_moves
        #   ]
        # )
        if n < self.random_factor:
            next_move = np.random.choice(allowed_moves)
        else:
            maxG = -10e15  # some really small random number
            for action in allowed_moves:
                new_state = tuple([sum(x) for x in zip(state, ACTIONS[action])])
                if self.G[new_state] >= maxG:
                    next_move = action
                    maxG = self.G[new_state]
        # print(next_move)
        return next_move


if __name__ == "__main__":
    maze = Maze()
    robot = Agent(maze.maze, alpha=0.1, random_factor=0.25)
    moveHistory = []

    for i in range(20000):
        if i % 1000 == 0:
            print(i)
        while not maze.is_game_over():
            # print(maze.steps, end=" ")
            state, _ = maze.get_state_and_reward()
            action = robot.choose_action(state, maze.allowed_states[state])
            maze.update_maze(action)
            state, reward = maze.get_state_and_reward()
            robot.update_state_history(state, reward)
            if maze.steps > 1000:
                maze.robot_position = (MAZE_SIZE - 1, MAZE_SIZE - 1)
        robot.learn()
        # print(maze.steps)
        moveHistory.append(maze.steps)
        maze = Maze()
    # print(moveHistory)
    plt.plot(moveHistory, "b--")
    plt.show()
