import argparse


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter,  # preserve formatting of help text
    argparse.MetavarTypeHelpFormatter,  # provide type information in help text
):
    pass


def get_args():
    """
    Get the arguments from the command line.
    """
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
    parser.add_argument(
        "-task",
        help="Task to run for the training of the model",
        type=str,
        default="Task04_Hippocampus",
    )
    parser.add_argument(
        "-google_id",
        help="Google drive id for the datas",
        type=str,
        default="1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C",
    )
    parser.add_argument(
        "-batch_size", help="Batch size for the training", type=int, default=16
    )
    parser.add_argument(
        "-train_val_ratio",
        help="Ratio of the training set to the validation set",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "-epochs", help="Number of epochs for the training", type=int, default=100
    )
    parser.add_argument(
        "-lr", help="Learning rate for the training", type=float, default=0.01
    )
    parser.add_argument(
        "-early_stopping",
        help="Early stopping for the training. None if you don't want to use Early Stopping, else choose an integer number > 0 that will represent the patience.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-train_from_checkpoint",
        help="Train from a checkpoint. Insert the path to the weights you wish to load",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-fine_tune",
        help="Fine tune the model, training only the last layer and freezing the others.",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-best_models_dir",
        help="Directory where the best models will be saved",
        type=str,
        default="best_models",
    )
    parser.add_argument(
        "-mixed_precision",
        help="Use mixed precision for the training",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "-Nit", help="Number of iterations for the training", type=int, default=None
    )
    parser.add_argument(
        "-random_seed", help="Random seed for the training", type=int, default=42
    )

    args = parser.parse_args()

    return args
