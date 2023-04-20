# Standard Libraries
import sys
import yaml
from argparse import Namespace

# Internal Libraries
from models.MLP import MLP
from models.utils.DataLoaderTabular import DataModuleTabular

# 3rd Party Libraries
from pytorch_lightning import Trainer
from dvclive.lightning import DVCLiveLogger
from lightning.pytorch.cli import LightningArgumentParser


# get args
stagename = sys.argv[1]
input_filename_data = sys.argv[2]
input_filename_train_test_split = sys.argv[3]
output_filename = sys.argv[4]

hparams = Namespace(**yaml.safe_load(open("params.yaml"))[stagename])

# Define model
if hparams.model == "MLP":
    model = MLP(hparams)
else:
    raise NotImplementedError(f"Model {hparams.model} not implemented")

# Define datamodule
if hparams.data["type"] == "Tabular":
    datamodule = DataModuleTabular(
        hparams, input_filename_data, input_filename_train_test_split
    )
else:
    raise NotImplementedError(f"Datamodule {hparams.type} not implemented")

# Define trainer
trainer = Trainer(
    accelerator="auto",
    logger=DVCLiveLogger(),
    max_epochs=hparams.model_hparams["num_epochs"],
    enable_progress_bar=True,
    log_every_n_steps=1,
    check_val_every_n_epoch=1,
)

# Train model
trainer.fit(model, datamodule)

# Test model
trainer.test(model, datamodule)
