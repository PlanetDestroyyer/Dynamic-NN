"""
DynamicBrain: Continual Learning via Neuron Strain Index (NSI)

This is a standalone reference implementation of the DynamicBrain architecture.
It includes the Dual-Routing kWTA mechanism and the NSI biological health monitor.
For full training loops and visualizations, see the Jupyter Notebooks in the `notebooks/` directory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def evaluate(net, tasks_test, idx):
    net.eval()
    accs = []
    with torch.no_grad():
        for i in range(idx+1):
            Xt, yt = tasks_test[i]
            Xt, yt = Xt.to(device), yt.to(device)
            pred = net(Xt)
            mask = torch.ones_like(pred, dtype=torch.bool)
            mask[:, [i*2, i*2+1]] = False
            pred[mask] = -float('inf')
            accs.append((pred.argmax(1)==yt).float().mean().item())
    net.train()
    return accs


class DynamicBrain(nn.Module):
    def __init__(self, input_dim=3072, output_dim=10, seed_neurons=10, k_frozen=10, k_plastic=5):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.k_frozen   = k_frozen
        self.k_plastic  = k_plastic
        n = seed_neurons

        self.hidden_W = nn.Parameter(torch.randn(n, input_dim) * 0.1)
        self.hidden_b = nn.Parameter(torch.zeros(n))
        self.out_W    = nn.Parameter(torch.zeros(output_dim, n))
        self.out_b    = nn.Parameter(torch.zeros(output_dim))

        self.neuron_states  = ['PLASTIC'] * n
        self.active_neurons = n

    def forward(self, x):
        h = F.linear(x, self.hidden_W, self.hidden_b)
        
        frozen_idx  = [i for i, s in enumerate(self.neuron_states) if s == 'FROZEN']
        plastic_idx = [i for i, s in enumerate(self.neuron_states) if s == 'PLASTIC']
        
        mask = torch.zeros_like(h)
        
        if frozen_idx:
            kf = min(self.k_frozen, len(frozen_idx))
            if kf > 0:
                hf = h[:, frozen_idx]
                _, top_idx = hf.topk(kf, dim=1)
                # Scatter 1.0 into mask at the correct global indices
                global_top = torch.tensor(frozen_idx, device=h.device)[top_idx]
                mask.scatter_(1, global_top, 1.0)
                
        if plastic_idx:
            kp = min(self.k_plastic, len(plastic_idx))
            if kp > 0:
                hp = h[:, plastic_idx]
                _, top_idx = hp.topk(kp, dim=1)
                global_top = torch.tensor(plastic_idx, device=h.device)[top_idx]
                mask.scatter_(1, global_top, 1.0)
                
        h = F.relu(h * mask)
        return F.linear(h, self.out_W, self.out_b)

    def zero_frozen_grads(self):
        frozen = [i for i,s in enumerate(self.neuron_states) if s=='FROZEN']
        if frozen:
            if self.hidden_W.grad is not None:
                self.hidden_W.grad[frozen, :] = 0
                self.hidden_b.grad[frozen]    = 0
            # REMOVED out_W.grad zeroing here.
            # We WANT new classes to learn to use frozen features!
            # Past classes are already protected by active-class masking.

    def grow_and_freeze(self, freeze_indices, num_new):
        for i in freeze_indices:
            self.neuron_states[i] = 'FROZEN'
        if num_new <= 0:
            return
        self.active_neurons += num_new
        self.neuron_states.extend(['PLASTIC'] * num_new)
        with torch.no_grad():
            dev = self.out_b.device
            hW = torch.randn(num_new, self.input_dim,  device=dev) * 0.1
            hb = torch.zeros(num_new,                  device=dev)
            oW = torch.zeros(self.output_dim, num_new, device=dev)  # ZERO init prevents corrupting past classes
            self.hidden_W = nn.Parameter(torch.cat([self.hidden_W, hW], 0))
            self.hidden_b = nn.Parameter(torch.cat([self.hidden_b, hb], 0))
            self.out_W    = nn.Parameter(torch.cat([self.out_W,    oW], 1))

print("DynamicBrain defined.")


class NeuronHealthMonitor:
    # Tracks per-neuron health via Neuron Strain Index (NSI).
    #
    # Three EMAs per neuron:
    #   grad_mag[n]     = EMA of ||grad_hidden_W[n]||   (demand)
    #   out_grad_mag[n] = EMA of ||grad_out_W[:,n]||    (output demand)
    #   contribution[n] = EMA of (grad_n . delta_w_n)   (SI: is it helping?)
    #
    # Freeze conditions (after min_age batches, checked every check_interval):
    #   CONVERGED  : grad_mag[n] < tau_low              for patience checks
    #   STRUGGLING : grad_mag[n] > tau_high AND
    #                contribution[n] <= tau_contrib      for patience checks
    #
    # tau_high_mode:
    #   'percentile' -> self-calibrating, adapts as network grows
    #   'absolute'   -> fixed; calibrate from diagnostic cell
    def __init__(self, num_neurons, device='cpu',
                 alpha=0.05,
                 check_interval=20, patience=3, min_age=40,
                 tau_low=0.001,
                 tau_high_mode='percentile',
                 tau_high_pct=0.70,
                 tau_high_abs=None,
                 tau_conflict=0.5,
                 track_output_grads=True):

        self.device           = device
        self.alpha            = alpha
        self.check_interval   = check_interval
        self.patience         = patience
        self.min_age          = min_age
        self.tau_low          = tau_low
        self.tau_high_mode    = tau_high_mode
        self.tau_high_pct     = tau_high_pct
        self.tau_high_abs     = tau_high_abs
        self.tau_conflict      = tau_conflict
        self.track_output_grads = track_output_grads

        self._alloc(num_neurons)
        self.w_prev   = None
        self.batch_no = 0

        # Diagnostics log (filled during training for analysis)
        self.log = []   # list of dicts, one per check_interval

    def _alloc(self, n):
        d = self.device
        self.grad_mag      = torch.zeros(n, device=d)
        self.out_grad_mag  = torch.zeros(n, device=d)
        self.conflict      = torch.zeros(n, device=d)
        self.grad_vec      = None
        self.age           = torch.zeros(n, dtype=torch.long, device=d)
        self.conv_count    = torch.zeros(n, dtype=torch.long, device=d)
        self.str_count     = torch.zeros(n, dtype=torch.long, device=d)
        self.n             = n

    def expand(self, num_new):
        d = self.device
        def _cat(t, m=None):
            if m is None:
                return torch.cat([t, torch.zeros(num_new, device=d)])
            return torch.cat([t, torch.zeros(num_new, dtype=m, device=d)])
        self.grad_mag     = _cat(self.grad_mag)
        self.out_grad_mag = _cat(self.out_grad_mag)
        self.conflict     = _cat(self.conflict)
        if self.grad_vec is not None:
            self.grad_vec = torch.cat([self.grad_vec, torch.zeros(num_new, self.grad_vec.size(1), device=d)])
        self.age          = _cat(self.age,        torch.long)
        self.conv_count   = _cat(self.conv_count, torch.long)
        self.str_count    = _cat(self.str_count,  torch.long)
        self.n           += num_new

    def pre_step(self, net):
        # Call BEFORE optimizer.step() to snapshot weights.
        self.w_prev = net.hidden_W.data.clone()

    def post_step(self, net, hidden_grad, out_grad=None):
        # Call AFTER optimizer.step() to update EMAs.
        delta = net.hidden_W.data - self.w_prev   # actual weight movement
        a     = self.alpha

        for n in range(self.n):
            if net.neuron_states[n] == 'FROZEN':
                continue
            gn     = hidden_grad[n]
            dn     = delta[n]
            gm     = gn.norm().item()
            if self.grad_vec is None:
                self.grad_vec = torch.zeros_like(hidden_grad)
                
            self.grad_mag[n] = (1-a)*self.grad_mag[n] + a*gm
            self.grad_vec[n] = (1-a)*self.grad_vec[n] + a*gn
            
            # Conflict Index: 1.0 - (||ema_grad_vec|| / ema_grad_mag)
            # High conflict (~1.0) means gradients are oscillating/pulling in different directions
            # Low conflict (~0.0) means gradients point in a consistent direction
            vec_mag = self.grad_vec[n].norm().item()
            conf = 1.0 - (vec_mag / (self.grad_mag[n].item() + 1e-8))
            self.conflict[n] = conf

            if self.track_output_grads and out_grad is not None:
                ogm = out_grad[:, n].norm().item()
                self.out_grad_mag[n] = (1-a)*self.out_grad_mag[n] + a*ogm

            self.age[n] += 1
        self.batch_no += 1

    def _tau_high(self, plastic_ids):
        if self.tau_high_mode == 'absolute':
            return self.tau_high_abs
        # percentile mode: self-calibrating
        if len(plastic_ids) < 2:
            return self.grad_mag[plastic_ids].max().item() if plastic_ids else 1e9
        gm = self.grad_mag[plastic_ids]
        return torch.quantile(gm, self.tau_high_pct).item()

    def check_health(self, net):
        # Called every check_interval batches.
        # Returns (freeze_indices, n_sprout, log_dict).

        plastic = [i for i,s in enumerate(net.neuron_states)
                   if s == 'PLASTIC' and self.age[i] >= self.min_age]
        to_freeze, n_sprout = [], 0

        tau_h = self._tau_high(plastic)

        for n in plastic:
            gm   = self.grad_mag[n].item()
            conf = self.conflict[n].item()
            ogm  = self.out_grad_mag[n].item()

            converged   = gm < self.tau_low
            struggling  = gm > tau_h and conf >= self.tau_conflict

            if converged:
                self.conv_count[n] += 1
                self.str_count[n]   = 0
                if self.conv_count[n] >= self.patience:
                    to_freeze.append(n)          # converged: freeze, no sprout
            elif struggling:
                self.str_count[n] += 1
                self.conv_count[n]  = 0
                if self.str_count[n] >= self.patience:
                    to_freeze.append(n)          # struggling: freeze + sprout
                    n_sprout += 1
            else:
                self.conv_count[n] = 0
                self.str_count[n]  = 0

        # Diagnostic snapshot
        snap = {
            'batch': self.batch_no,
            'tau_high': tau_h,
            'tau_high_mode': self.tau_high_mode,
            'n_plastic': len(plastic),
            'n_frozen': sum(1 for s in net.neuron_states if s=='FROZEN'),
            'n_freeze_now': len(to_freeze),
            'n_sprout_now': n_sprout,
            'grad_mag': self.grad_mag.cpu().tolist(),
            'out_grad_mag': self.out_grad_mag.cpu().tolist(),
            'conflict': self.conflict.cpu().tolist(),
        }
        self.log.append(snap)
        plastic_count = net.neuron_states.count('PLASTIC')
        plastic_remaining = plastic_count - len(to_freeze)
        min_p = net.k_plastic
        if plastic_remaining + n_sprout < min_p:
            n_sprout += min_p - (plastic_remaining + n_sprout)
        return to_freeze, n_sprout, snap

print("NeuronHealthMonitor defined.")




if __name__ == "__main__":
    print("Initializing DynamicBrain with NSI architecture...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Initialize the network with optimal parameters
    net = DynamicBrain(input_dim=3072, output_dim=10, seed_neurons=10, k_frozen=5, k_plastic=5).to(device)
    
    # 2. Initialize the Neuron Health Monitor
    monitor = NeuronHealthMonitor(
        num_neurons=10, 
        device=device,
        alpha=0.05, 
        check_interval=20, 
        patience=3, 
        min_age=40,
        tau_low=0.025,
        tau_high_mode='percentile',
        tau_high_pct=0.70,
        tau_conflict=0.5,
        track_output_grads=False
    )
    
    print(f"Network initialized on {device} with {net.active_neurons} seed neurons.")
    print("See notebooks/cifar10_nsi_benchmark.ipynb for the complete continual learning training loop!")
