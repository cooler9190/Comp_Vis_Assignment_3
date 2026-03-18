import torch
from torchsummary import summary

# Baseline
from load_CIFAR10 import test_dataloader
from CIFAR10_lenet import LeNet5Color as ModelToSummarize
load_path = "baseline_results_and_weights/history_baseline.pth"

# Variant 1: Dropout
# from load_CIFAR10 import test_dataloader
# from CIFAR10_model1 import LeNet5Variant1 as ModelToSummarize
# load_path = "variant1_results_and_weights/25_percent_dropout_probability/history_variant1.pth"

# Variant 2: Increased number of kernels and dropout [best cifar 10]
# from load_CIFAR10 import test_dataloader
# from CIFAR10_model2 import LeNet5Variant2 as ModelToSummarize
# load_path = "variant2_results_and_weights/45_percent_dropout_probability/history_variant2.pth"

# Variant Cifar 100 model: Adapted Variant 2 to work on Cifar100 dataset.
# from load_CIFAR100 import test_dataloader
# from CIFAR100_model import Cifar100 as ModelToSummarize
# load_path = "variant_cifar100_results_and_weights/history_variant_cifar100.pth"

# Variant Pretrained: Uses params of cifar 100 model (except for the output layer due to differing feature sizes).
# from load_CIFAR10 import test_dataloader
# from CIFAR10_pretrained import LeNet5VariantPretrained as ModelToSummarize
# load_path = "variant_pretrained_results_and_weights/45_percent_dropout_probability/history_variant_pretrained.pth"

def summarize_model(model_to_test):
    # Get accelerator
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = model_to_test() # Instantiate the model.
    model.load_state_dict(torch.load(load_path)) # Load params.
    model.to(device) # Move model to the appropriate device (GPU or CPU).
    model.eval() # Set model to evaluation mode.

    # Print summary, using image shape from dataloader as input size.
    images, _ = next(iter(test_dataloader))
    summary(model, input_size=images.shape[1:])
    # Show and save model graph.
summarize_model(ModelToSummarize)