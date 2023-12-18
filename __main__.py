import tensorflow as tf
import numpy as np
import random as rng

from scipy.signal import convolve2d

ROW_COUNT = 6
COL_COUNT = 7

HORIZONTAL_KERNEL = np.array([[1, 1, 1, 1]])
VERTICAL_KERNEL = np.transpose(HORIZONTAL_KERNEL)
DIAGONAL1_KERNEL = np.eye(4, dtype=np.uint8)
DIAGONAL2_KERNEL = np.fliplr(DIAGONAL1_KERNEL)
DETECTION_KERNELS = [
	HORIZONTAL_KERNEL,
	VERTICAL_KERNEL,
	DIAGONAL1_KERNEL,
	DIAGONAL2_KERNEL
]


class Connect4:
	def __init__(self):
		self.board = np.zeros((ROW_COUNT, COL_COUNT))
		self.player = 1

	def get_open_row(self, col):
		for row in reversed(range(ROW_COUNT)):
			if self.board[row][col] == 0:
				return row

		return None

	def place(self, col):
		if self.player is None:
			raise PermissionError("Game already done")

		if col >= COL_COUNT:
			raise OverflowError(f"Column {col} does not exist")

		row = self.get_open_row(col)
		if row is None:
			raise OverflowError(f"Column {col} already full")

		self.board[row][col] = self.player

		# https://stackoverflow.com/questions/29949169/how-to-implement-the-function-that-checks-for-a-win-in-a-python-based-connect-fo
		for kernel in DETECTION_KERNELS:
			if (convolve2d(self.board == self.player, kernel, mode='valid') == 4).any():
				self.player = None
				return True

		self.player = 2 if self.player == 1 else 1
		return False


def play_compete(player1, player2):
	board = Connect4()
	players = [player1, player2]
	player = 0
	turns = 0

	try:
		while board.player is not None:
			# because players are 1 and 2 subtract 1 to get the index
			player = board.player - 1

			# Count the number of turns
			turns += 1

			# Auto switches the board player index to the next
			# Also passes the board and the number to which belongs to the player
			if board.place(players[player].evaluate(board.board, player + 1)):
				break

	except OverflowError:
		# When an exception is raised we need the other player
		player = 1 if player == 0 else 0

	return player, turns


class Connect4Player:
	def __init__(self, model):
		self.model = model
		self.states = []
		self.labels = []

	def evaluate(self, board, self_num):
		opponent = 2 if self_num == 1 else 1

		board_copy = np.copy(board)
		board_copy[board == self_num] = 1
		board_copy[board == opponent] = 2

		prediction = np.argmax(self.model(board_copy.reshape((1, ROW_COUNT, COL_COUNT))))
		self.states.append(board_copy)
		self.labels.append(prediction)
		return prediction


def check_fitness(model, opponent) -> float:
	model_player = Connect4Player(model)
	opponent_player = Connect4Player(opponent)
	(winner, turns) = play_compete(model_player, opponent_player)

	# When winning give values above 1, but reward shorter games (shorter -> higher value)
	fitness = (ROW_COUNT * COL_COUNT) / turns

	# When losing give values below 1, but reward longer games (longer -> higher value)
	if winner == 1:
		fitness = turns / (ROW_COUNT * COL_COUNT)

	return fitness


def create_neural_network():
	return tf.keras.models.Sequential([
		tf.keras.layers.Input(shape=(ROW_COUNT, COL_COUNT)),
		tf.keras.layers.Rescaling(1 / 2),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(ROW_COUNT, activation=tf.keras.activations.softmax),
	])


GAMES_FOR_AVERAGE = 10
GENERATIONS = 100
POPULATION = 100
MUTATION_RATE = 0.1


def mutate(model: tf.keras.Model):
	for layer in model.layers:
		if np.random.rand() < MUTATION_RATE:
			layer.set_weights([weight + 0.1 * np.random.randn(*weight.shape) for weight in layer.get_weights()])


def crossover(parents: list[tf.keras.Model]) -> tf.keras.Model:
	child = create_neural_network()

	for i, parent_layers in enumerate(zip([parent.layers for parent in parents])):
		child.layers[i].set_weights(rng.choice(parent_layers)[i].get_weights())

	return child


def train_neural_network(population: list[tf.keras.Model], games: int, epochs: int):
	for individual in population:

		player = Connect4Player(individual)
		for _ in range(games):
			play_compete(player, player)

		individual.compile(
			optimizer=tf.keras.optimizers.Adam(),
			loss=tf.keras.losses.SparseCategoricalCrossentropy(),
			metrics=[
				tf.keras.metrics.SparseCategoricalAccuracy(),
			]
		)
		states = np.array(player.states)
		labels = np.array(player.labels)

		individual.fit(states, labels, epochs=epochs)


def random_valid_column(board):
	connect4 = Connect4()
	connect4.board = np.copy(board[0])

	valid_moves = list(filter(lambda col: connect4.get_open_row(col) is not None, range(COL_COUNT)))
	index = rng.choice(valid_moves) if len(valid_moves) > 0 else 0  # Return 0 when no valid moves, we lose these

	categories = [0] * COL_COUNT
	categories[index] = 1
	return categories


def evaluate_average_fitness(population: list[tf.keras.Model], games: int):
	average_fitness = 0

	for _ in range(games):
		population_fitness = [check_fitness(i, random_valid_column) for i in population]
		population_fittest = np.argmax(population_fitness)
		average_fitness += population_fitness[population_fittest]

	return average_fitness / games


def main():
	population = [create_neural_network() for _ in range(POPULATION)]

	for generation in range(GENERATIONS):
		fitness_scores = [check_fitness(individual, rng.choice(population)) for individual in population]

		selected_indices = np.argsort(fitness_scores)[-POPULATION // 2:]
		selected_population = [population[i] for i in selected_indices]

		while len(selected_population) < POPULATION:
			child = crossover([
				population[rng.choice(selected_indices)],
				population[rng.choice(selected_indices)],
			])
			mutate(child)
			selected_population.append(child)

		population = selected_population
		#train_neural_network(population, 100, 10)

		average_fitness = evaluate_average_fitness(population, GAMES_FOR_AVERAGE)
		print(f"Generation {generation + 1}, Average Fitness: {average_fitness}")


if __name__ == '__main__':
	main()
