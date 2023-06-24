from lstm import BuiltInLSTM
import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

model = BuiltInLSTM()

inputs = torch.tensor([[0.0, 0.5, 0.25, 1.0], [1.0, 0.5, 0.25, 1.0]])
labels = torch.tensor([0.0, 1.0])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)


trainer = L.Trainer(max_epochs=300)
trainer.fit(model, train_dataloaders=dataloader)

print("Company A: Observed = 0, Predicted =", model(torch.tensor([0.0, 0.5, 0.25, 1.0])).detach())
print("Company B: Observed = 0, Predicted =", model(torch.tensor([1.0, 0.5, 0.25, 1.0])).detach())
