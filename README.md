# Mixture-of-Robust-Experts-MoRE-
This repo present the main code to realize the Mixture-of-Robust-Experts structure. This structure could mix clean, adversarial and natural perturbations models to defend against corresponding types of inputs.

Folder `cifar10` and `Tiny ImageNet` are the codes that achieve MoRE in two different dataset cifar10 and Tiny ImageNet.

The `single model training` folder in this folder is to help us to train clean, adversarial and natura perturbations experts individually. To download cifar10 dataset or Tiny ImageNet dataset and train different types of experts:

    python main.py

Sometimes, there are some problems about downloading Tiny ImageNet, we can also download and unzip the following archive:


* [Tiny ImageNet Dataset](https://tiny-imagenet.herokuapp.com/)  

The trained model will be stored in `checkpoint` folder. and put these trained models to folder `trained_model`. Then we could use our the following codes to achieve MoRE.

To apply MoRE to clean expert and different types of adversarial models:

    python main_desk_adv.py

To apply MoRE to clean expert and different types of natural perturbations models:

    python main_desk_nat.py
    
To apply MoRE to all experts, which include clean expert, different types of adversarial models and natural perturbations models:

    python main_desk_all.py
    
To achieve Dynamic Image Type Classifier(DITC):

    python main_desk_DITC.py
    
To achieve baseline method Mix Mixture of Experts(MMoE):

    python main_desk_max.py
    
To achieve baseline method Average Mixture of Experts(AMoE):

    python main_desk_aver.py
    
The folder `MSD` is the code to achieve MSD method in similar experimental setting. There are also have two folders `cifar` and `tinyimagenet` that are the applying MSD to two different dataset cifar10 and Tiny Imagenet. Different `.sh` file in these two folders could achieve all experiments mentioned in our paper.
