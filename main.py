# imports
from utils.data_class import MedicalDecathlonDataModule
import torch
import monai
from utils.trainer_class import Trainer
from utils.args import get_args
from utils.other_utils import assert_config

# Get the arguments from the command line
args = get_args()
assert_config(args.__dict__)

# set the random seed
torch.manual_seed(args.random_seed)

# Data definition and preparation following the steps in the tutorial
# https://colab.research.google.com/github/fepegar/torchio-notebooks/blob/main/notebooks/TorchIO_MONAI_PyTorch_Lightning.ipynb#scrollTo=pHXXLvDM8g6U
print("\nLoading and preparing the dataloaders...")
data = MedicalDecathlonDataModule(
    task=args.task,
    google_id=args.google_id,
    batch_size=args.batch_size,
    train_val_ratio=args.train_val_ratio,
)

data.prepare_data()
data.setup()

# Get the dataloaders
train_data_loader = data.train_dataloader()
val_data_loader = data.val_dataloader()
test_data_loader = data.test_dataloader()

# model, loss and optimizer definition
print("\nDefining the model, the loss and the optimizer...")
model = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=3,
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
)

criterion = monai.losses.DiceCELoss(softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define trainer
trainer = Trainer(
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    **args.__dict__,
)

# training loop
trainer.training_loop()
