import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_model.py <model.pth>")
    sys.exit(1)

model_path = sys.argv[1]
ckpt = torch.load(model_path, map_location='cpu')

print(f"Inspecting: {model_path}")
print(f"Type: {type(ckpt)}")

if isinstance(ckpt, dict):
    print(f"Dict keys: {list(ckpt.keys())}")
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        print("Has model_state_dict")
    else:
        state = ckpt
else:
    state = ckpt
    print("Direct state dict")

# Check architecture
fc_keys = [k for k in state.keys() if 'fc' in k]
print(f"\nFC layer keys: {fc_keys}")
if fc_keys:
    print(f"FC shape: {state[fc_keys[0]].shape}")

# Check if it's wrapped in backbone
has_backbone = any('backbone' in k for k in state.keys())
print(f"Has 'backbone' prefix: {has_backbone}")

# Count layers
print(f"Total parameters: {len(state.keys())}")