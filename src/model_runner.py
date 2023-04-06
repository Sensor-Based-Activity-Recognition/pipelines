# Internal Libraries
from models.MLP import MLP
from models.utils.DataLoaderTabular import DataModuleTabular

# 3rd Party Libraries
from pytorch_lightning import Trainer
from dvclive.lightning import DVCLiveLogger
from lightning.pytorch.cli import LightningArgumentParser


if __name__ == "__main__":
    # Arguments
    parser = LightningArgumentParser(description="MLP Model")
    parser.add_argument(
        "data_filename",
        help="Path of the file containing the training and test dataset",
    )
    parser.add_argument(
        "train_test_split_filename",
        help="Path of the file containing the train/test split",
    )
    parser.add_argument(
        "model", help='Name of the model to be used in training etc., e.g. "MLP"'
    )
    parser.add_argument(
        "datamodule",
        help='Name of the datamodule to be used in training etc., e.g. "DataModuleTabular"',
    )
    parser.add_argument(
        "input_size",
        type=int,
        help="Number of features in the input Data",
    )
    parser.add_argument(
        "output_size",
        type=int,
        help="Number of classes to predict / Number of labels",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for the SGD optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="Weight decay / L2 penalty for the SGD optimizer",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for the SGD optimizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the dataloader",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.8,
        help="Percentage of the training data to use for training, the rest is used for validation",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs to train the model",
    )
    args = parser.parse_args()

    # Define model
    if args.model.lower() == "mlp":
        model = MLP(args)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    # Define datamodule
    if args.datamodule.lower() == "tabular":
        datamodule = DataModuleTabular(args)
    else:
        raise NotImplementedError(f"Datamodule {args.datamodule} not implemented")

    # Define trainer
    trainer = Trainer(
        accelerator="auto",
        logger=DVCLiveLogger(),
        max_epochs=args.num_epochs,
        enable_progress_bar=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )

    # Train model
    trainer.fit(model, datamodule)

    # Test model
    trainer.test(model, datamodule)
