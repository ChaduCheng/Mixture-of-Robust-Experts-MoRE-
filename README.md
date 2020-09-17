# Mixture-of-Robust-Experts-MoRE-
This repo present the main code to realize the Mixture-of-Robust-Experts structure. This structure could mix clean, adversarial and natural perturbations models to defend against corresponding types of inputs.

### The folder cifar10 is to apply our MoRE method to cifar10 dataset

The `single model training` folder in this folder is to help us to train clean, adversarial and natura perturbations experts individually. To download cifar10 dataset and train different types of experts:

    python main.py

The trained model will be stored in `checkpoint` folder. and put these trained models to folder `trained_model`. Then we could use our the following codes to achieve MoRE.

To apply MoRE to clean expert and different types of adversarial models:

        python main_desk_adv.py

### The folder Tiny ImageNet is to apply our MoRE method to Tiny ImageNet dataset
