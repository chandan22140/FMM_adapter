import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
# Add at top
import wandb
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Dict
import math
from typing import Optional, List

import time
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#####################################
# Hyperparameters
#####################################
EPOCHS = 1
batch_size = 64            # Adjust as your GPU allows
BASE_LR = 1e-3
WEIGHT_DECAY = 0.03
DROPOUT = 0.1
R_LORA = 4              # Example LoRA rank
LORA_ALPHA = 32           # Example LoRA alpha
LORA_DROPOUT = 0

class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha

        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class xLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        pretrained_weights=None,  # Added to accept pretrained weights
        pretrained_bias=None,     # Added to accept pretrained bias
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        if pretrained_weights is not None:
            self.weight.data = pretrained_weights
        if pretrained_bias is not None:
            self.bias.data = pretrained_bias

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self._initialize_lora_parameters()  # Only initialize LoRA parameters
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    # def reset_parameters(self):
    #     nn.Linear.reset_parameters(self)
    #     if hasattr(self, 'lora_A'):
    #         nn.init.zeros_(self.lora_A)
    #         nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def _initialize_lora_parameters(self):
        """
        Initialize only the LoRA-specific parameters (lora_A and lora_B).
        Avoid reinitializing self.weight or self.bias to preserve pretrained values.
        """
        if hasattr(self, 'lora_A'):
            # nn.init.zeros_(self.lora_A)  # A is initialized to zero
            # nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))  # B is initialized as per LoRA paper

            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.01)  # Changed from zeros

            
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class SampledxLinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        pretrained_weights=None,
        pretrained_bias=None,
        s: int = 4,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if pretrained_weights is not None:
            self.weight.data = pretrained_weights
        if pretrained_bias is not None:
            self.bias.data = pretrained_bias
        self.s = s  # Number of samples per iteration
        self.C = 0.0  # To store the sum of norms for regularization

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / r
            self.weight.requires_grad = False
        self._initialize_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def _initialize_lora_parameters(self):
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.01)

    # def _initialize_lora_parameters(self):
    #     if hasattr(self, 'lora_A'):
    #         # Initialize lora_A with kaiming and lora_B with small random values
    #         nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    #         nn.init.normal_(self.lora_B, mean=0.0, std=0.01)  # Changed from zeros
            
    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            if self.training:
                A = self.lora_A  # (r, in_features)
                B = self.lora_B  # (out_features, r)
                
                norms_A = torch.norm(A, p=2, dim=1) + 1e-8  # (r,)
                norms_B = torch.norm(B, p=2, dim=0) + 1e-8  # (r,)
                products = norms_A * norms_B
                C = products.sum()  
                self.C = C  # Store for regularization
                p = products / C

                # After probability calculation
                assert not torch.isnan(p).any(), "NaN values in probabilities"
                assert (p >= 0).all(), "Negative probabilities detected"


                sampled_indices = torch.multinomial(p, self.s, replacement=True)
                p_sampled = p[sampled_indices]

                A_sampled = A[sampled_indices, :]
                B_sampled = B[:, sampled_indices]

                A_sampled_scaled = A_sampled / p_sampled[:, None]
                B_sampled_scaled = B_sampled / p_sampled[None, :]

                sum_terms = A_sampled_scaled.T @ B_sampled_scaled.T
                scaling = self.lora_alpha / self.s
                lora_term = (self.lora_dropout(x) @ sum_terms) * scaling
            else:
                lora_term = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling

            result = F.linear(x, T(self.weight), bias=self.bias) + lora_term
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
        return result

        

        

def replace_linear_with_sampled_lora(module: nn.Module, parent_name='', skip_substring='heads.head', s=4):
    for name, child in list(module.named_children()):
        module_path = f"{parent_name}.{name}" if parent_name else name
        replace_linear_with_sampled_lora(child, parent_name=module_path, skip_substring=skip_substring, s=s)
        if isinstance(child, nn.Linear) and skip_substring not in module_path:
            pretrained_weights = child.weight.data.clone()
            pretrained_bias = child.bias.data.clone() if child.bias is not None else None
            lora_linear = SampledxLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                r=R_LORA,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                pretrained_weights=pretrained_weights,
                pretrained_bias=pretrained_bias,
                s=s
            )
            lora_linear.lora_A.requires_grad = True
            lora_linear.lora_B.requires_grad = True            
            setattr(module, name, lora_linear)

def replace_linear_with_lora(module: nn.Module, parent_name='', skip_substring='heads.head'):
    """
    Recursively replace all nn.Linear modules with LoRALayer.Linear,
    while preserving pretrained weights and biases and skipping specific submodules.

    Args:
        module: The module to recursively replace layers in.
        parent_name: Tracking the current path to correctly identify modules to skip.
        skip_substring: Substring to check in the module path to decide skipping replacement.
    """
    for name, child in list(module.named_children()):
        # Form the fully qualified name (like 'encoder.layer1.linear')
        module_path = f"{parent_name}.{name}" if parent_name else name

        # Recursively apply to child modules first
        replace_linear_with_lora(child, parent_name=module_path, skip_substring=skip_substring)

        if isinstance(child, nn.Linear) and skip_substring not in module_path:
            # Extract pretrained weights and bias
            pretrained_weights = child.weight.data.clone()
            pretrained_bias = child.bias.data.clone() if child.bias is not None else None

            # Replace the nn.Linear with LoRA-wrapped Linear
            lora_linear = xLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                r=R_LORA,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                pretrained_weights=pretrained_weights,
                pretrained_bias=pretrained_bias,
                
            )
            setattr(module, name, lora_linear)
def get_regularizer_loss(model, lambda_reg):
    regularizer_loss = 0.0
    for module in model.modules():
        if isinstance(module, SampledxLinear) and module.training:
            regularizer_loss += module.C
    # return lambda_reg * regularizer_loss
    return 0

def count_trainable_parameters(model):
    """
    Counts and returns the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
#     for n, p in model.named_parameters():
#         if 'lora_' not in n:
#             p.requires_grad = False 
#         else:
#             p.requires_grad = True

#     if bias == 'none':
#         return
#     elif bias == 'all':
#         for n, p in model.named_parameters():
#             if 'bias' in n:
#                 p.requires_grad = True
#             # else:
#             #     p.requires_grad = False

#     elif bias == 'lora_only':
#         for m in model.modules():
#             if isinstance(m, LoRALayer) and \
#                 hasattr(m, 'bias') and \
#                 m.bias is not None:
#                     m.bias.requires_grad = True
#     else:
#         raise NotImplementedError

def mark_lora_and_head_as_trainable(model: nn.Module, head_substring="heads.head", bias='none'):
    """
    Unfreeze LoRA parameters + the final classification head (by default `heads.head`).
    Everything else remains frozen.
    """
    for name, param in model.named_parameters():
        # Unfreeze LoRA parameters
        if 'lora_' in name:
            param.requires_grad = True
        # Unfreeze classification head
        elif head_substring in name:
            print("head_substring came:", name)
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Optionally allow some bias fine-tuning
    if bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True

# Implement a linear learning rate decay
def lr_lambda(current_step: int):
    """
    Linear decay from step=0 to step=total_steps. At step=0 => 1.0; at step=total_steps => 0.0
    """
    progress = float(current_step) / float(EPOCHS * len(train_loader))
    return max(0.0, 1.0 - progress)

def compare_encoder_weights_consistency_with_xlinear(encoder_before, encoder_after):
    """
    Compare the pretrained weights and biases of nn.Linear layers in the encoder of two models.

    This ensures that the nn.Linear part of xLinear in the modified encoder
    matches the original nn.Linear weights and biases from the pretrained encoder.

    Args:
        encoder_before: Encoder module before applying LoRA and unfreezing logic.
        encoder_after: Encoder module after applying LoRA and unfreezing logic.

    Returns:
        None (prints whether weights and biases are consistent or not).
    """
    print("Comparing nn.Linear weights and biases between original encoder and modified encoder...")

    for (name_before, module_before), (name_after, module_after) in zip(
        encoder_before.named_modules(), encoder_after.named_modules()
    ):
        # Compare only nn.Linear layers in encoder_before to xLinear in encoder_after
        if isinstance(module_before, nn.Linear) and isinstance(module_after, xLinear):
            if torch.equal(module_before.weight.data, module_after.weight.data):
                # print(f"[MATCH] {name_before}: Weights are identical.")
                pass
            else:
                print(f"[MISMATCH] {name_before}: Weights differ.")

            if module_before.bias is not None and module_after.bias is not None:
                if torch.equal(module_before.bias.data, module_after.bias.data):
                    # print(f"[MATCH] {name_before}: Biases are identical.")
                    pass

                else:
                    print(f"[MISMATCH] {name_before}: Biases differ.")
            elif module_before.bias is None and module_after.bias is None:
                # print(f"[MATCH] {name_before}: Both layers have no bias.")
                pass
            else:
                print(f"[MISMATCH] {name_before}: One layer has bias while the other does not.")

    print("Comparison complete.")


#####################################
# 1. Data Preparation
#####################################

# Add new hyperparameters
S_LORA = 4          # Number of samples per iteration
LAMBDA_REG = 0.01   # Regularization strength

# Initialize WandB

# Add at top
import wandb
from sklearn.metrics import f1_score

# Initialize WandB
wandb.init(project="lora-sampling", config={
    "epochs": EPOCHS,
    "batch_size": batch_size,
    "base_lr": BASE_LR,
    "weight_decay": WEIGHT_DECAY,
    "dropout": DROPOUT,
    "r_lora": R_LORA,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "s": S_LORA,
    "lambda_reg": LAMBDA_REG
})

# Set random seed for reproducibility
torch.manual_seed(17)

transform = transforms.Compose([ 
    transforms.Resize((224, 224)), 
    transforms.ToTensor()        
])
# train_dir = "/kaggle/input/tiny-imagenet/tiny-imagenet-200/tiny-imagenet-200/train"
train_dir = "/home/chandan/DL_Quantization/NOLA/vit/data/CIFAR100_ImageFolder/train"
dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
print(f"Dataset size: {len(dataset)}")
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [40000, 5000, 5000])


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
# from torchvision.utils import make_grid

# for images, _ in train_loader:
#     plt.figure(figsize=(16,8))
#     plt.axis('off')
#     plt.imshow(make_grid(images, nrow=8).permute((1, 2, 0)))
#     break




#####################################
# 2. Model Preparation
#####################################



# Load pre-trained ViT-B/16 weights from torchvision
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

# Modify the classification head for CIFAR-10 (10 classes)
num_features = model.heads.head.in_features
model.heads.head = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(num_features, 200)   
)

print(f"Number of trainable parameters(total): {count_trainable_parameters(model)}")
print(f"Number of trainable parameters(heads.head): {count_trainable_parameters(model.heads.head)}")
print(f"Number of trainable parameters(encoder): {count_trainable_parameters(model.encoder)}")
print(f"Number of trainable parameters(conv_proj): {count_trainable_parameters(model.conv_proj)}")
  
# Replace linear layers with sampled LoRA
replace_linear_with_sampled_lora(model, s=S_LORA)
mark_lora_and_head_as_trainable(model, head_substring="heads.head", bias="none")
for name, param in model.named_parameters():
    if 'lora_B' in name:
        print(f"{name} mean: {param.data.mean().item():.4f}, std: {param.data.std().item():.4f}")

print(f"Number of trainable parameters(total): {count_trainable_parameters(model)}")
print(f"Number of trainable parameters(heads.head): {count_trainable_parameters(model.heads.head)}")
print(f"Number of trainable parameters(encoder): {count_trainable_parameters(model.encoder)}")
print(f"Number of trainable parameters(conv_proj): {count_trainable_parameters(model.conv_proj)}")
print(model.heads.head)

trainable_params_list = [name for name, param in model.named_parameters() if param.requires_grad]
print(f"Trainable parameters: {len(trainable_params_list)}")
print(trainable_params_list[:10])  # print first 10 for inspection

#####################################
# 3. Optimizer & Scheduler
#####################################

# Filter only trainable (LoRA) parameters
trainable_params = filter(lambda p: p.requires_grad, model.parameters())



optimizer = torch.optim.AdamW(trainable_params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


#####################################
# 4. Training & Validation Loop with Multi-GPU Support
#####################################

# Check if multiple GPUs are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)  # Wrap the model for multi-GPU
model.to(device)

criterion = nn.CrossEntropyLoss()
    
# for epoch in range(EPOCHS):
#     model.train()
#     running_loss = 0.0

#     for step, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)  # DataParallel splits the batch across GPUs
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()  # Update learning rate each step

#         running_loss += loss.item() * images.size(0)

#         if step % 10 == 0:
#             current_lr = scheduler.get_last_lr()[0]
#             print(f"[Epoch {epoch+1}/{EPOCHS} - Step {step}] Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
#             # Inside your training loop
#             writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + step)
#             writer.add_scalar('Learning Rate', current_lr, epoch * len(train_loader) + step)

#             # Validation
#             model.eval()
#             correct = 0
#             total = 0
#             with torch.no_grad():
#                 for images, labels in val_loader:
#                     images = images.to(device)
#                     labels = labels.to(device)
#                     outputs = model(images)
#                     _, predicted = torch.max(outputs, 1)
#                     total += labels.size(0)
#                     correct += (predicted == labels).sum().item()

    
#     epoch_loss = running_loss / len(train_loader.dataset)
#     val_acc = 100.0 * correct / total
#     # Inside your validation loop
#     writer.add_scalar('Validation Accuracy', val_acc, epoch * len(train_loader) + step)

#     print(f"Epoch [{epoch+1}/{EPOCHS}], Training Loss: {epoch_loss:.4f}, Validation Acc: {val_acc:.2f}%\n")

# Training loop modifications

# Modified training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        reg_loss = get_regularizer_loss(model, LAMBDA_REG)
        total_loss = loss + reg_loss
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Accumulate metrics
        epoch_train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # Inside your training loop where you have:
        if step % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            
            # Add this block to log parameter norms
            lora_a_norms = []
            lora_b_norms = []
            
            # Handle DataParallel if used
            model_to_check = model.module if isinstance(model, nn.DataParallel) else model
            
            for name, module in model_to_check.named_modules():
                if isinstance(module, SampledxLinear):
                    lora_a_norms.append(torch.norm(module.lora_A).item())
                    lora_b_norms.append(torch.norm(module.lora_B).item())
            
            # Log metrics
            log_data = {
                "batch/train_loss": loss.item(),
                "batch/reg_loss": reg_loss,
                "batch/learning_rate": current_lr
            }
            
            if lora_a_norms:  # Only add if we found LoRA layers
                log_data.update({
                    "batch/avg_lora_a_norm": sum(lora_a_norms)/len(lora_a_norms),
                    "batch/avg_lora_b_norm": sum(lora_b_norms)/len(lora_b_norms),
                    "batch/max_lora_a_norm": max(lora_a_norms),
                    "batch/max_lora_b_norm": max(lora_b_norms)
                })
            
            wandb.log(log_data)


    # Calculate epoch training metrics
    avg_train_loss = epoch_train_loss / total_train
    train_acc = 100.0 * correct_train / total_train
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate validation metrics
    avg_val_loss = val_loss / total_val
    val_acc = 100.0 * correct_val / total_val
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Log epoch metrics
    wandb.log({
        "epoch/train_loss": avg_train_loss,
        "epoch/train_acc": train_acc,
        "epoch/val_loss": avg_val_loss,
        "epoch/val_acc": val_acc,
        "epoch/val_f1": val_f1
    })
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
    print(f"Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}\n")

# Final test evaluation
model.eval()
test_acc = 0.0
test_f1 = 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100.0 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
test_f1 = f1_score(all_labels, all_preds, average='macro')

wandb.log({
    "final/test_acc": test_acc,
    "final/test_f1": test_f1
})

wandb.finish()