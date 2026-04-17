import torch

weights_path = "weights/tracknet_weights.pt"
checkpoint = torch.load(weights_path, map_location="cpu")

# Handle different PyTorch save formats
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    keys = list(checkpoint['model_state_dict'].keys())
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    keys = list(checkpoint['state_dict'].keys())
elif isinstance(checkpoint, dict):
    keys = list(checkpoint.keys())
else:
    print("Not a dictionary format.")
    keys = []

print(f"Total layers in weights file: {len(keys)}")
print("First 15 layer names in the file:")
for k in keys[:15]:
    print(f" - {k}")