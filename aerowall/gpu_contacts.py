"""Small explicit GPU contact readback adapter for native Isaac Sim 2023.1.

Pair identity/lifecycle comes from PhysX headers; this adapter supplies valid
positive-impulse points for registered filter slots. It does not infer FOUND or
LOST from force thresholds. One instance per environment is a validation-first
implementation; full training throughput remains to be measured.
"""
import math


class GPUContactReadback:
    def __init__(self, source_path, filter_paths, name, capacity=64):
        from omni.isaac.core.prims.rigid_contact_view import RigidContactView
        self.source_path = source_path
        self.filter_paths = tuple(filter_paths)
        self.capacity = capacity
        self.view = RigidContactView(source_path, list(filter_paths), name=name,
            prepare_contact_sensors=False, disable_stablization=False,
            apply_rigid_body_api=False, max_contact_count=capacity)
        self.view.initialize()
        if self.view.num_shapes != 1 or self.view.num_filters != len(filter_paths):
            raise RuntimeError('Contact source/filter shape differs from explicit routing')

    def read(self, threshold=1e-6):
        import torch
        impulse, position, normal, separation, counts, starts = self.view.get_contact_force_data(dt=1.)
        if impulse.device.type != 'cuda' or position.device.type != 'cuda':
            raise RuntimeError('Expected native GPU contact tensors')
        if int(counts.sum()) >= self.capacity:
            raise RuntimeError('Contact buffer may be saturated')
        aggregate = self.view.get_contact_force_matrix(dt=1.)
        result = [[] for _ in self.filter_paths]
        for index in range(len(self.filter_paths)):
            count, start = int(counts[0,index]), int(starts[0,index])
            if count < 0 or start < 0 or start+count > len(impulse):
                raise RuntimeError('Invalid contact buffer interval')
            reconstructed = torch.zeros(3,device=position.device)
            for row in range(start,start+count):
                value = float(impulse[row].item())
                if not math.isfinite(value):
                    raise RuntimeError('Nonfinite contact impulse')
                reconstructed += impulse[row]*normal[row]
                if abs(value) <= threshold:
                    continue
                if not torch.isfinite(position[row]).all() or not .99 < float(normal[row].norm()) < 1.01:
                    raise RuntimeError(f'Invalid contact source={self.source_path} filter={self.filter_paths[index]} impulse={value} point={position[row].cpu().tolist()} normal={normal[row].cpu().tolist()} count={count} start={start}')
                result[index].append({'impulse':abs(value), 'raw_signed_coefficient':value,
                                      'point':tuple(position[row].cpu().tolist()),
                                      'normal':tuple((normal[row] if value>=0 else -normal[row]).cpu().tolist()),
                                      'separation':float(separation[row].item())})
            if not torch.allclose(reconstructed,aggregate[0,index],atol=1e-5,rtol=1e-4):
                raise RuntimeError(f'Point impulses disagree with pair aggregate: {self.source_path} {self.filter_paths[index]}')
        return result
