"""
Compute the Hessian of a molecular PES using NequIP
"""

import pickle
from pathlib import Path

import yaml
import ase.io
import numpy as np
import torch
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from nequip.nn import GraphModuleMixin, RescaleOutput
from nequip.data import AtomicDataDict, AtomicData


def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    r'''
    Compute the gradient of `outputs` with respect to `inputs`
    ```
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    ```
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])


def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    r'''
    Compute the Hessian of `output` with respect to `inputs`
    ```
    hessian((x * y).sum(), [x, y])
    ```
    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    numel = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(numel, numel)

    row_index = 0
    for i, inp in enumerate(inputs):        #https://pytorch.org/docs/stable/dynamo/index.html
        print('index i: {}'.format(i))
        [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            print('index j: {}'.format(j))
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[row_index, row_index:].add_(row.type_as(out))  # row_index's row
            if row_index + 1 < numel:
                out[row_index + 1:, row_index].add_(row[1:].type_as(out))  # row_index's column
            del row
            row_index += 1
        del grad

    return out


@compile_mode("script")
class ExtendedHessianOutput(GraphModuleMixin, torch.nn.Module):
    r"""Wrap a model and include as an output its hessian."""

    def __init__(
        self,
        func: GraphModuleMixin,
    ):
        super().__init__()
        self.func = func

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={AtomicDataDict.TOTAL_ENERGY_KEY: Irreps("0e")},
            irreps_out=func.irreps_out,
        )
        self.irreps_out.update(
            {f: self.irreps_in[wrt] for f, wrt in [('forces', 'pos'), ('hessian', 'pos')]}
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
        data = AtomicDataDict.with_batch(data)
        batch = data[AtomicDataDict.BATCH_KEY]
        num_batch: int = int(batch.max().cpu().item()) + 1
        pos = data[AtomicDataDict.POSITIONS_KEY]
        orig_cell = data[AtomicDataDict.CELL_KEY]
        cell = orig_cell.view(-1, 3, 3).expand(num_batch, 3, 3)         # Make the cell per-batch

        # add dummy displacements to cellvecs to allow autograd
        # why not take gradient wrt cell directly?
        displacement = torch.zeros(
            (num_batch, 3, 3),
            dtype=pos.dtype,
            device=pos.device,
        ).requires_grad_(True)
        #symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
        #pos.requires_grad_(True)
        # bmm is natom in batch
        #data[AtomicDataDict.POSITIONS_KEY] = pos + torch.bmm(
        #    pos.unsqueeze(-2), symmetric_displacement[batch]
        #).squeeze(-2)
        new_cell = cell + displacement
        data |= {'_displacement': displacement, AtomicDataDict.CELL_KEY: new_cell.squeeze(0)}

        # make pos depend on cellvecs (and therefore on displacements) to allow autograd
        frac = torch.matmul(pos, torch.inverse(orig_cell)).requires_grad_(True)
        data[AtomicDataDict.POSITIONS_KEY] = torch.matmul(frac, new_cell.squeeze(0))
        
        # evaluate model and compute hessian
        data = self.func(data)
        hess = hessian(data[AtomicDataDict.TOTAL_ENERGY_KEY].sum(), [frac, displacement])
        data['hessian'] = hess

        return data


def wrap_model_extended(model: RescaleOutput) -> RescaleOutput:
    """
    Modify existing NequIP model by extracting its EnergyModel
    and wrapping in into ExtendedHessianOutput and RescaleOutput layers
    """
    
    # extract correct layer
    energy_model = next(module for module in model.modules() if hasattr(module, 'total_energy_sum'))
    
    # wrap into ExtendedHessianOutput and RescaleOutput
    kwargs = {k: getattr(model, k) for k in
              ('shift_keys', 'related_scale_keys', 'related_shift_keys', 'scale_by', 'shift_by', 'irreps_in')}
    hessian_model = RescaleOutput(model=ExtendedHessianOutput(func=energy_model),
                                  scale_keys=['hessian'], **kwargs)
    
    if hessian_model.shift_by.nelement() == 0: hessian_model.has_shift = None
    if hessian_model.scale_by.nelement() == 0: hessian_model.has_scale = None
    
    return hessian_model
    
    
def get_model(path_model: Path, path_config: Path, device='cpu'):
    """Load a nequip model that is not deployed"""
    import ase.data
    from nequip.data.transforms import TypeMapper
    from nequip.train import Trainer
    config_dict = yaml.load(Path(path_config).open('r'), Loader=yaml.Loader)
    model, metadata = Trainer.load_model_from_training_session(path_model.parent, path_model.name,
                                                               device=device, config_dictionary=config_dict)
    type_names = metadata.type_names
    
    # stolen from nequip_calculator.from_deployed_model method
    species_to_type_name = {s: s for s in ase.data.chemical_symbols}
    type_name_to_index = {n: i for i, n in enumerate(type_names)}
    chemical_symbol_to_type = {sym: type_name_to_index[species_to_type_name[sym]]
                               for sym in ase.data.chemical_symbols if sym in type_name_to_index}
    if len(chemical_symbol_to_type) != len(type_names):
        raise ValueError("The default mapping of chemical symbols as type names didn't make sense")
    return model, metadata, TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)


def check_sanity(hess: np.ndarray, atoms: ase.Atoms):
    """More things that Sander included in his script"""
    
    def _enlarge_3x3(a, nrepeats):
        return np.kron(np.eye(nrepeats, dtype=int), a)
    
    assert np.allclose(hess, hess.T)
    
    # transform fractional part of extended hessian to cartesian coordinates
    invcell_enlarged = _enlarge_3x3(np.linalg.inv(atoms.cell), len(atoms))
    transform = np.eye(3 * len(atoms) + 9)
    n = 3 * len(atoms)
    transform[:n, :n] = invcell_enlarged
    hess_cart = transform @ hess @ transform.T

    # inspect hessian, compute curvatures as experienced by each atom to make
    # sure these values make sense
    for i in range(len(atoms)):
        start = 3 * i
        stop = start + 3
        values = np.linalg.eigvalsh(hess[start:stop, start:stop])
        print(values)
        
    return hess, hess_cart