import torch
import sys

data = torch.load(sys.argv[1], map_location='cpu')
print('Type:', type(data))
if isinstance(data, dict):
    print('Dict keys:', list(data.keys()))
elif isinstance(data, tuple) or isinstance(data, list):
    print('Tuple/List length:', len(data))
    for i, item in enumerate(data):
        print(f'  [{i}] type: {type(item)}')
        if isinstance(item, dict):
            print(f'    keys: {list(item.keys())}')
else:
    print('Unknown structure')
