import matplotlib.pyplot as plt

from dataset import MAESTRO_small, allocate_batch
from constants import HOP_SIZE
from evaluate import evaluate

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

example_dataset = MAESTRO_small(path='data', groups=['debug'], sequence_length=None, random_sample=True)

data = example_dataset[1]
print(f'data: {data}')
print(f'audio_shape: {data["audio"].shape}')
print(f'frame_roll_shape: {data["frame"].shape}')
print(f'onset_roll_shape: {data["onset"].shape}')

print(f'HOP_SIZE({HOP_SIZE}) x piano_roll length({data["frame"].shape[0]}): {HOP_SIZE*data["frame"].shape[0]}')


plt.figure(figsize=(10,15))
plt.subplot(311)
plt.plot(data['audio'].numpy()[:400*HOP_SIZE])
plt.autoscale(enable=True, axis='x', tight=True)
plt.subplot(312)
plt.imshow(data['frame'].numpy()[:400].T, aspect='auto', origin='upper')
plt.subplot(313)
plt.imshow(data['onset'].numpy()[:400].T, aspect='auto', origin='lower')


from model import Transcriber_ONF

cnn_unit = int(48)
fc_unit = int(256)
learning_rate = float(6e-4)
weight_decay = 0
step = 0

model = Transcriber_ONF(cnn_unit=cnn_unit, fc_unit=fc_unit)
optimizer = torch.optim.Adam(model.parameters(),
                             learning_rate,
                             weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.98)
criterion = nn.BCEWithLogitsLoss()

state = torch.load('runs/exp_211110-145857/model-10000.pt')

model.load_state_dict(state['model_state_dict'])
model.eval()

batch = allocate_batch(data, 'cpu')
#sample = evaluate(model, batch, 'cpu')

print(batch['audio'].unsqueeze(0).shape)

frame_logit, onset_logit = model(batch['audio'].unsqueeze(0))

print(frame_logit.shape)
print(onset_logit.shape)

a = torch.squeeze(frame_logit)
b = torch.squeeze(onset_logit)

a = torch.sigmoid(a)
b = torch.sigmoid(b)

plt.figure(figsize=(10,15))
plt.autoscale(enable=True, axis='x', tight=True)
plt.subplot(211)
plt.imshow(a.detach().numpy()[:400].T, aspect='auto', origin='upper')
plt.subplot(212)
plt.imshow(b.detach().numpy()[:400].T, aspect='auto', origin='lower')

plt.show()