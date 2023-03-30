
import os

import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer

datasetPath = "models/hagrid_120k/dataset"
exportDir = "models/hagrid_120k/model"

trainingRatio = .8

datasetHyperParams = gesture_recognizer.HandDataPreprocessingParams(
    shuffle = True,
    min_detection_confidence = 0.7
)

modelHyperParams = gesture_recognizer.HParams(

    # Parameters for train configuration
    learning_rate = 0.001,
    batch_size = 2,
    epochs = 10,
    steps_per_epoch = None,

    lr_decay = 0.99,
    gamma = 2,

    # Dataset-related parameters
    shuffle = False,

    # Parameters for model / checkpoint files
    export_dir=exportDir,

    # Parameters for hardware acceleration
    # Note: MediaPipe only supports CPU for python
    distribution_strategy = 'off',
    num_gpus = -1,  # default value of -1 means use all available GPUs
    tpu = ''
)

model_options = gesture_recognizer.ModelOptions(
    dropout_rate = 0.05,
    layer_widths = [],
)

def main():

    print("------- STARTING ---------")

    print(f"Searching: '{datasetPath}'")
    labels = []
    for i in os.listdir(datasetPath):
        if os.path.isdir(os.path.join(datasetPath, i)):
            labels.append(i)

    print(f"Labels: '{labels}'")

    data = gesture_recognizer.Dataset.from_folder(
        dirname = datasetPath,
        hparams = datasetHyperParams
    )
    train_data, rest_data = data.split(trainingRatio)
    validation_data, test_data = rest_data.split(0.5)

    print(f"Created Dataset - Train: {len(train_data)} | Validate: {len(validation_data)} | Test: {len(test_data)}")

    options = gesture_recognizer.GestureRecognizerOptions(model_options=model_options, hparams=modelHyperParams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data = train_data,
        validation_data = validation_data,
        options = options
    )

    loss, accuracy = model.evaluate(test_data, batch_size=32)
    print(f"Test loss:{loss}, Test accuracy:{accuracy}")

    print(f"Exporting Model to: '{exportDir}'")
    model.export_model()

    print("------- DONE! ---------")


if __name__ == "__main__":
    main()