import argparse
import os

import tensorflow as tf

assert tf.__version__.startswith("2")

from mediapipe_model_maker import gesture_recognizer
from mediapipe_model_maker.python.core.utils import loss_functions


def getLabels(datasetPath):

    print(f"Searching: '{datasetPath}'")

    #Warn: This code needs to be structured like this so function
    #      returns same ordering as gesture_recognizer.Dataset.from_folder 

    labels = sorted(
        name for name in os.listdir(datasetPath)
        if os.path.isdir(os.path.join(datasetPath, name))
    )

    lowercaseLabels = [v.lower() for v in labels]
    try:
        none_idx = lowercaseLabels.index('none')        
    except ValueError as e:
      raise ValueError('Label set does not contain label "None".')

    # Move label 'none' to the front of label list.
    none_value = labels.pop(none_idx)
    labels.insert(0, none_value)

    return labels

def createDateset(datasetPath, trainingRatio = 0.8):

    datasetHyperParams = gesture_recognizer.HandDataPreprocessingParams(
        shuffle=True, min_detection_confidence=0.7
    )

    print(f"Building Dataset...")
    data = gesture_recognizer.Dataset.from_folder(
        dirname=datasetPath, hparams=datasetHyperParams
    )
    train_data, rest_data = data.split(trainingRatio)
    validation_data, test_data = rest_data.split(0.5)

    print(
        f"Created Dataset - Train: {len(train_data)} | Validate: {len(validation_data)} | Test: {len(test_data)}"
    )

    return train_data, validation_data, test_data


def createModel(datasetPath, exportDir, trainEpochs=0) -> tuple[gesture_recognizer.GestureRecognizer, list, list, list]:
    
    labels = getLabels(datasetPath)
    print(f"Labels: '{labels}'")

    hparams = gesture_recognizer.HParams(
        # Parameters for train configuration
        learning_rate=0.001,
        batch_size=2,
        epochs=25,
        steps_per_epoch=None,
        lr_decay=0.99,
        gamma=2,
        # Dataset-related parameters
        shuffle=False,
        # Parameters for model / checkpoint files
        export_dir=exportDir,
        # Parameters for hardware acceleration
        # Note: MediaPipe only supports CPU for python
        distribution_strategy="off",
        num_gpus=-1,  # default value of -1 means use all available GPUs
        tpu="",
    )

    model_options = gesture_recognizer.ModelOptions(
        dropout_rate=0.05,
        layer_widths=[],
    )

    options = gesture_recognizer.GestureRecognizerOptions(
        model_options=model_options, hparams=hparams
    )

    if trainEpochs > 0:
        trainData, validationData, testData = createDateset(datasetPath)

        model = gesture_recognizer.GestureRecognizer.create(
            train_data=trainData, validation_data=validationData, options=options
        )

    else:
        # Just load checkpoint
        trainData      = None
        validationData = None
        testData       = None

        model = gesture_recognizer.GestureRecognizer(
            label_names=labels,
            model_options=options.model_options,
            hparams=options.hparams
        )

        model._create_model()

        model._model.compile(
            optimizer='adam',
            loss=loss_functions.FocalLoss(gamma=model._hparams.gamma),
            metrics=['categorical_accuracy']
        )

        checkpoint_path = os.path.join(hparams.export_dir, 'epoch_models')
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint:
            print(f'Resuming from {latest_checkpoint}')
            model._model.load_weights(latest_checkpoint)

    return model, trainData, validationData, testData


def main():
    print("------- STARTING ---------")

    
    argParser = argparse.ArgumentParser(
        prog="gestureTrainer",
        description="Gesture Trainer for Remote Free TV",
    )

    argParser.add_argument(
        "-d",
        "--dir",
        action="store",
        required=True,
        help="Specifies the directory to load dataset from and/or save model to",
    )

    argParser.add_argument(
        "-t",
        "--train",
        metavar="int:epochs",
        action="store",
        default="0",
        required=False,
        help="Specifies the number of epochs to train the model for. Default value is 0",
    )

    args = argParser.parse_args()

    datasetPath = f"{args.dir}/dataset"
    exportDir = f"{args.dir}/model"

    trainEpochs = int(args.train)
    
    model, trainData, validationData, testData = createModel(datasetPath, exportDir, trainEpochs)

    print(f"Exporting Model to: '{exportDir}'")
    model.export_model()

    if testData is not None:
        loss, accuracy = model.evaluate(testData, batch_size=32)
        print(f"Test loss:{loss}, Test accuracy:{accuracy}")

    print("------- DONE! ---------")


if __name__ == "__main__":
    main()
