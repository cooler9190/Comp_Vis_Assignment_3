from training_and_validation import run_and_save_results

# Import model architectures

# Baseline
# from load_CIFAR10 import train_dataloader, validation_dataloader # Import dataloaders for CIFAR-10 dataset
# from CIFAR10_lenet import LeNet5Color as ModelToTrain
# model_filename = "history_baseline.json"
# model_filename = "history_baseline_scheduler.json"
# model_filename = "history_baseline_augmented_data.json"

# Variant 1: Dropout
# from load_CIFAR10 import train_dataloader, validation_dataloader # Import dataloaders for CIFAR-10 dataset
# from CIFAR10_model1 import LeNet5Variant1 as ModelToTrain
# model_filename = "history_variant1.json"

# Variant 2: Increased number of kernels and dropout [best cifar 10]
# from load_CIFAR10 import train_dataloader, validation_dataloader # Import dataloaders for CIFAR-10 dataset
# from CIFAR10_model2 import LeNet5Variant2 as ModelToTrain
# model_filename = "history_variant2.json"

# Variant Cifar 100 model: Adapted Variant 2 to work on Cifar100 dataset.
# from load_CIFAR100 import train_dataloader, validation_dataloader # Import dataloaders for CIFAR-100 dataset
# from CIFAR100_model import Cifar100 as ModelToTrain
# model_filename = "history_variant_cifar100.json"

# Variant Pretrained: Uses params of cifar 100 model (except for the output layer due to differing feature sizes).
from load_CIFAR10 import train_dataloader, validation_dataloader # Import dataloaders for CIFAR-10 dataset
from CIFAR10_pretrained import LeNet5VariantPretrained as ModelToTrain
model_filename = "history_variant_pretrained_augmented.json"

if __name__ == "__main__":
    # run_and_save_results(ModelToTrain, model_filename, train_dataloader, validation_dataloader)
    # run_and_save_results(ModelToTrain, model_filename, train_dataloader, validation_dataloader, True) # For cifar 100.
    run_and_save_results(ModelToTrain, model_filename, train_dataloader, validation_dataloader, converge_mode=False, learning_rate=0.0005) # For variant pretrained
    # run_and_save_results(ModelToTrain, model_filename, train_dataloader, validation_dataloader, use_scheduler=True) # For baseline with scheduler