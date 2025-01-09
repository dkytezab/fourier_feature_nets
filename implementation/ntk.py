import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

def grad(model,
         input: torch.Tensor
) -> torch.Tensor:

  params = list(model.parameters())
  model.zero_grad()
  output = model(input)
  output.backward(
      torch.ones_like(output))
  return torch.cat([param.grad.view(-1) for param in params
                    if param.requires_grad])


class NTK():
  def __init__(self,
               model,
               input_loader: torch.Tensor
  ) -> None:

    dataset = input_loader.dataset

    num_samples = dataset[:, 0][0].numpy().shape[0]
    emp_matrix = torch.zeros((num_samples, num_samples))

    grads = [grad(model, dataset[idx][0])
             for idx in range(num_samples)]

    for row in range(num_samples):
      for col in range(row, num_samples):
        emp_matrix[row, col] = \
        emp_matrix[col, row] = \
        torch.dot(grads[row], grads[col])

    np_emp_matrix = emp_matrix.numpy()

    self.empirical_matrix = np_emp_matrix
    self.spectrum = np.linalg.eigvalsh(np_emp_matrix)[::-1]
    
