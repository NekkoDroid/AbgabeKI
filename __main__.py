import tensorflow as tf
import numpy as np

from scipy.signal import convolve2d
from pathlib import Path

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

	def place(self, col):
		if self.player is None:
			raise PermissionError("Game already done")

		if col >= COL_COUNT:
			raise OverflowError(f"Column {col} does not exist")

		row = self.get_open_row(col)
		if not row:
			raise OverflowError(f"Column {col} already full")

		self.board[row][col] = self.player

		# https://stackoverflow.com/questions/29949169/how-to-implement-the-function-that-checks-for-a-win-in-a-python-based-connect-fo
		for kernel in DETECTION_KERNELS:
			if (convolve2d(self.board == self.player, kernel, mode='valid') == 4).any():
				self.player = None
				return True

		self.player = 2 if self.player == 1 else 1
		return False


def main(model_path: Path):
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

	"""
	The images in the MNIST dataset have pixel values that range from [0, 255]
	Reduce the pixel values to [0.0, 1.0) floating point numbers to give to the model
	"""
	train_images = train_images / 255.0
	test_images = test_images / 255.0

	if not model_path.exists():
		"""
		Sequential network models feed the results from one layer directly to the following layer

		Layers:
			1 -> Flatten the image from a 28x28 2d-matrix to a 784 1d-matrix
			2 -> Reduce the input to a fully connected 128 1d-matrix
			3 -> Drop 20% of the input to prevent over-fitting to the training dataset
			4 -> Reduce the output to 10 values mapping to the 10 labels of the dataset
		"""
		model = tf.keras.models.Sequential([
			tf.keras.layers.Flatten(input_shape=train_images.shape[1:]),
			tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax),
		])

		"""
		Compile the model with certain settings during the training:
		optimizer: <I-couldn't-figure-out-what-the-fuck-this-is-used-for>
		loss: Calculates the loss (how wrong) of the model during training
		metrics: Used to decides how the results should be evaluated (relates to 'loss')
		"""
		model.compile(
			optimizer=tf.keras.optimizers.Adam(),
			loss=tf.keras.losses.SparseCategoricalCrossentropy(),
			metrics=[
				tf.keras.metrics.SparseCategoricalAccuracy(),
			]
		)

		"""Train the model on the training dataset"""
		model.fit(train_images, train_labels, epochs=5)

		"""Serialize the model to be loaded later again"""
		model.save(model_path)

	"""Deserialize a previously save model"""
	model = tf.keras.models.load_model(model_path)

	"""Evaluate the trained model on the test dataset"""
	model.evaluate(test_images, test_labels, verbose=2)


if __name__ == '__main__':
	main(Path("./model.keras"))
