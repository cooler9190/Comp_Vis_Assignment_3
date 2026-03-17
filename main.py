from training_and_validation import run_and_save_results

# Import model architectures

# Baseline
# from CIFAR10_lenet import LeNet5Color as ModelToTrain
# model_filename = "history_baseline.json"

# Variant 1: Dropout
# from CIFAR10_model1 import LeNet5Variant1 as ModelToTrain
# model_filename = "history_variant1.json"

# Variant 2: Increased number of kernels and dropout
# from CIFAR10_model2 import LeNet5Variant2 as ModelToTrain
# model_filename = "history_variant2.json"

# Cifar 100
from CIFAR100_model import Cifar100 as ModelToTrain
model_filename = "history_variant_cifar100.json"

if __name__ == "__main__":
    # run_and_save_results(ModelToTrain, model_filename)
    run_and_save_results(ModelToTrain, model_filename, True) # For cifar 100.