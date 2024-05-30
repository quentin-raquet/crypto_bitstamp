import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from utils import str_to_datetime
from api_client import BitstampClient
from data import TimeSeriesDataModule
from model import TimeSeriesPredictionModel


# Data parameters
curr_symbs = ["btceur", "etheur", "ltceur"]
context_len = 24
input_cols = ["close", "volume"]
start_date = str_to_datetime("2021-01-01")
end_date = str_to_datetime("2021-02-01")
api_key = os.getenv("BITSTAMP_API_KEY")
api_secret = os.getenv("BITSTAMP_API_SECRET")
client = BitstampClient(api_key, api_secret)
step_sec = 3600
max_res = 1000
batch_size = 64
val_size = 0.2

# Initialize the data module
data_module = TimeSeriesDataModule(
    client,
    curr_symbs,
    context_len,
    input_cols,
    start_date,
    end_date,
    step_sec,
    max_res,
    batch_size,
    val_size,
)
data_module.setup()

# Hyperparameters
num_inputs = len(curr_symbs)  # Number of input features (BTC/EUR and ETH/EUR)
num_channels = [32, 32, 16, 16]  # Number of channels in TCN layers
kernel_size = 3  # Kernel size in TCN layers
learning_rate = 0.001  # Learning rate for Adam optimizer
max_epochs = 100
model_checkpoint_path = "model_checkpoints"
patience = 5

# Initialize the model
model = TimeSeriesPredictionModel(num_inputs, num_channels, kernel_size, learning_rate, batch_size)

# Define the callbacks
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
model_checkpoint_callback = ModelCheckpoint(
    dirpath=model_checkpoint_path,
    filename="model-{epoch:02d}-{val_loss:.2f}",
    monitor="val_loss",
    mode="min",
)

# Train the model
trainer = Trainer(
    max_epochs=max_epochs, callbacks=[early_stopping_callback, model_checkpoint_callback]
)
trainer.fit(model, datamodule=data_module)
