# Exercises-MRI-segmentation

Here is the structure of the following repo:
- main.py &rarr; contains the functions to run the training from the command prompt, but in order to see the possible arguments for the function you might refer to the file "utils/args.py" or the list below.
- main.ipynb --> a simple notebook where the code can be run step-by-step. On top you can find a dictionary called "config" where you can define the values you want to use for your run. So after preparing the data, it runs the training, and after follows the testing of the model on some unseen data.
- The "utils" folder contains a series of .py files:
  * trainer_class.py -> contains the the most important part of the code. Here is where the Trainer class is defined, and after definig the data, model, loss, optimizer etc. this has to be used to start the training. The code does the same as in this [tutorial](https://colab.research.google.com/github/fepegar/torchio-notebooks/blob/main/notebooks/TorchIO_MONAI_PyTorch_Lightning.ipynb#scrollTo=KuhTaRl3vf37) wrt to the training, but is made using normal pytorch code. I decided to make it a bit modular for clarity, so the main part is the function "training_loop", but "forward_pass" and "backward_pass" are separate functions. So first it's necessary to define a Trainer object, and then to start the training use "Trainer.training_loop()".
  * data_class.py --> contain the same Lightning module for the dataloader as in the [tutorial](https://colab.research.google.com/github/fepegar/torchio-notebooks/blob/main/notebooks/TorchIO_MONAI_PyTorch_Lightning.ipynb#scrollTo=KuhTaRl3vf37), I simply copied the code so that I could load the DataLoaders with the relative transformations.
  * args.py --> it containes the argparse function to run the code from terminal when using "main.py".
  * other_utils.py --> file with a utility function to assert that the values in the config dict or argparse are correct, and a plotting function.
  
  Argparse arguments for main.py
  ```
  usage: args.py [-h] [-task str] [-google_id str] [-batch_size int]
               [-train_val_ratio float] [-epochs int] [-lr float]
               [-early_stopping int] [-train_from_checkpoint str]
               [-fine_tune bool] [-best_models_dir str]
               [-mixed_precision bool] [-Nit int] [-random_seed int]

    optional arguments:
    -h, --help            show this help message and exit
    -task str             Task to run for the training of the model (default:
                            Task04_Hippocampus)
    -google_id str        Google drive id for the datas (default:
                            1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C)
    -batch_size int       Batch size for the training (default: 16)
    -train_val_ratio float
                            Ratio of the training set to the validation set
                            (default: 0.8)
    -epochs int           Number of epochs for the training (default: 100)
    -lr float             Learning rate for the training (default: 0.01)
    -early_stopping int   Early stopping for the training. None if you don't
                            want to use Early Stopping, else choose an integer
                            number > 0 that will represent the patience. (default:
                            None)
    -train_from_checkpoint str
                            Train from a checkpoint. Insert the path to the
                            weights you wish to load (default: None)
    -fine_tune bool       Fine tune the model, training only the last layer and
                            freezing the others. (default: False)
    -best_models_dir str  Directory where the best models will be saved
                            (default: best_models)
    -mixed_precision bool
                            Use mixed precision for the training (default: True)
    -Nit int              Number of iterations for the training (default: None)
    -random_seed int      Random seed for the training (default: 42)
# Proposed solutions
## Exercise 1 
- Write a training code for a similar training as in the tutorial, but without the pytorch_lightning library.

The code where I made the training loop without using PytorchLightning can be found inside "utils/trainer_class.py" under the Trainer class inside the training_loop function.
- Make one script with a command line for training.

It is possible to run the code as a single command line script using "main.py" and the possible flags can be seen in the list above. 
- In the training loop use the automatic mixed precision from Pytorch (with autocast and
GradScaler) in order to train with FP16 precision instead of the default FP32.

If you want to start the training using mixed precision, you just have to run the code with the code setting up the flag for it `python main.py -mixed_precision True`. Here are the parts where this is implemented

In the training setup, the scaler is defined, however the argument "mixed_precision" enables the scaler or not so during the training this remains flexible.
![image](https://user-images.githubusercontent.com/63954877/228507397-cc7c40dc-1e95-46b6-beb8-d4935b6c6490.png)
In the training loop we can use the context manager "with torch.cuda.amp.autocast(); even here it is used only if it is enabled, otherwise it doesn't use it.
![image](https://user-images.githubusercontent.com/63954877/228506000-641c7dd9-8722-4fd8-a336-83c2382c70c7.png)
Finally when performing the backward pass, only if the scaler is enabled, the gradients are scaled to prevent gradient underflow.
![image](https://user-images.githubusercontent.com/63954877/228506503-dc9f438d-abae-405d-b80f-a218050f8355.png)


## Exercise 2
- Implement an option to perform a fine-tuning strategy: load a previously saved model, or start from random weights, and freeze all layer parameters for the Unet model except the last classification layer.

This is implemented inside the Trainer class in the training_setup function. 

In order to start the training from a chekpoint, it is simply necessary to set up an argument in the argparser (or config file in the .ipynb) called "train_from_checkpoint" `python main.py -train_from_checkpoint YOUR_CKPT_PATH`, if it is left as None, then the weights are randomly initialized, else they are loaded from the checkpoint path.

If instead you want to perform fine-tuning, you just have to setup the flag "fine_tune" as True when running the code `python main.py -fine_tune True`. As seen in the picture below, the last 2 layers of weights (corresponding to the last convolutional layer and its bias) are frozen (setting the flag "requires_grad" to False).

![image](https://user-images.githubusercontent.com/63954877/228523902-67781b87-ca1f-4eab-95f1-224e4e41802c.png)
## Exercise 3
- Because we used a strong data augmentation strategy (thanks to Torchio), the dataset length does not need to be equal to the real length. Indeed, in the nnUnet paper they proposed to train always with the same scheme : 1000 epoch of 250 iterations.
- Make the necessary changes so that the training epoch is always Nit iterations (i.e. Nit*batch size training volumes).
- Make sure that all training samples are equiprobably chosen, whatever the chosen Nit value.

The proposed solution consists in adding a flag called "Nit", if is None, then all the batches for each epochs are run, otherwise the model will run only the set number of iterations (eg. `python main.py -Nit 8` will run only 8 iterations/batches per epoch).

To make sure that all samples have equal probabilities of being chosen I added the flag shuffle=True to the "train_dataloader" function inside "utils.data_class"; this ensure that samples are always shuffled before making the batches and hence they have the same probability of being chosen even if not all the batches are processed in the epoch.

![image](https://user-images.githubusercontent.com/63954877/228525975-a3b8c892-7c35-4e95-ab30-f4915aa75527.png)
![image](https://user-images.githubusercontent.com/63954877/228526568-a75223e1-0818-45ca-a918-b396cd1251e8.png)
