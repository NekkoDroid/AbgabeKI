import tensorflow as tf

from pathlib import Path


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
