# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# A Dual Descriptor Network class demo: hDD (Tensor form) mixed with Transformer encoder
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by ChatGPT); Date: 2025-8-12

import math
import random
import pickle

class DDNet:
    """
    DDNet: interleaves Descriptor layers and Transformer encoder layers.
    This modified implementation uses pre-LayerNorm (pre-LN) in both descriptor layers
    and transformer layers. That is:
      - Descriptor: ln_input = LayerNorm(input_vec); x = M * ln_input; res = R * ln_input; out = res + Nk
      - Transformer: x = x + Wo(Attention(LN(x))); x = x + FFN(LN(x))
    No post-LN is applied.
    NOTE: This is still a pure-Python reference implementation (no numpy).
    """

    def __init__(self, input_dim=10, model_dims=[8,6,3], num_basis_list=[5,4,3],
                 attn_ff_hidden_factor=2, ln_eps=1e-5):
        assert len(model_dims) == len(num_basis_list), "model_dims and num_basis_list must align"
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.num_basis_list = num_basis_list
        self.num_desc_layers = len(model_dims)
        self.ln_eps = ln_eps

        # descriptor layers
        # Each descriptor layer contains:
        #  - M: projection from ln_input (in_dim) -> x (out_dim)
        #  - P: basis coefficients (out_dim x out_dim x num_basis)
        #  - periods: same shape as P, used to compute cos factors
        #  - R: residual projection from ln_input (in_dim) -> out_dim (so residual addition works)
        #  - ln_gamma_in, ln_beta_in: LayerNorm learnable params for the input vector (size in_dim)
        self.desc_layers = []
        for l, out_dim in enumerate(model_dims):
            in_dim = input_dim if l==0 else model_dims[l-1]
            # M: out_dim x in_dim
            M = [[random.uniform(-0.5, 0.5) for _ in range(in_dim)] for _ in range(out_dim)]
            # P: out_dim x out_dim x num_basis
            P = [[[random.uniform(-0.1, 0.1) for _ in range(num_basis_list[l])]
                  for _ in range(out_dim)]
                 for _ in range(out_dim)]
            periods = [[[ i*(out_dim*num_basis_list[l]) + j*num_basis_list[l] + g + 2
                         for g in range(num_basis_list[l])]
                        for j in range(out_dim)]
                       for i in range(out_dim)]
            # R: residual projection out_dim x in_dim
            R = [[ (0.0 if not (i==j and in_dim==out_dim) else 1.0) + random.uniform(-0.01, 0.01)
                   for j in range(in_dim)] for i in range(out_dim)]
            # LayerNorm params for descriptor input (size = in_dim, because we normalize input_vec)
            ln_gamma_in = [1.0]*in_dim
            ln_beta_in = [0.0]*in_dim

            self.desc_layers.append({'M': M, 'P': P, 'periods': periods, 'R': R,
                                     'ln_gamma_in': ln_gamma_in, 'ln_beta_in': ln_beta_in})

        # transformer layers interleaved (after each desc except last)
        # LayerNorm params (gamma,beta) for the two pre-LN layers per transformer block.
        self.trans_layers = []
        for l in range(self.num_desc_layers - 1):
            dim = model_dims[l]
            Wq = [[random.uniform(-0.2,0.2) for _ in range(dim)] for _ in range(dim)]
            Wk = [[random.uniform(-0.2,0.2) for _ in range(dim)] for _ in range(dim)]
            Wv = [[random.uniform(-0.2,0.2) for _ in range(dim)] for _ in range(dim)]
            Wo = [[random.uniform(-0.2,0.2) for _ in range(dim)] for _ in range(dim)]
            b_o = [0.0] * dim
            ff_hidden = max(1, int(attn_ff_hidden_factor * dim))
            W1 = [[random.uniform(-0.2,0.2) for _ in range(dim)] for _ in range(ff_hidden)]
            b1 = [0.0] * ff_hidden
            W2 = [[random.uniform(-0.2,0.2) for _ in range(ff_hidden)] for _ in range(dim)]
            b2 = [0.0] * dim
            # LayerNorm params (gamma initialized to 1, beta to 0)
            ln1_gamma = [1.0]*dim   # pre-LN before attention
            ln1_beta = [0.0]*dim
            ln2_gamma = [1.0]*dim   # pre-LN before FFN
            ln2_beta = [0.0]*dim
            self.trans_layers.append({
                'dim': dim,
                'Wq': Wq, 'Wk': Wk, 'Wv': Wv, 'Wo': Wo, 'b_o': b_o,
                'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
                'ln1_gamma': ln1_gamma, 'ln1_beta': ln1_beta,
                'ln2_gamma': ln2_gamma, 'ln2_beta': ln2_beta
            })
        self.trained = False

        # ---------- readout (decoder) from network final output back to input space ----------
        # This readout is used by auto_train: decoded = read_W * y + read_b, where
        #   y is final output vector (dim = model_dims[-1]),
        #   decoded has dimension = input_dim (so we can compute MSE against original input vectors)
        # Initialize read_W (input_dim x out_dim) and read_b (input_dim).
        out_dim = model_dims[-1]
        self.read_W = [[random.uniform(-0.1, 0.1) for _ in range(out_dim)] for _ in range(self.input_dim)]
        self.read_b = [0.0] * self.input_dim

    # ---------------------
    # basic helpers
    # ---------------------
    def _mat_vec(self, M, v):
        return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

    def _transpose(self, M):
        return [list(col) for col in zip(*M)]

    def _dot(self, u, v):
        return sum(u_i*v_i for u_i,v_i in zip(u,v))

    def _zeros(self, n):
        return [0.0]*n

    # LayerNorm forward: returns normalized vector and stats for backward.
    # We use the same function for both input-size and hidden-size layernorms.
    def _layer_norm_forward(self, x, gamma, beta, eps=None):
        if eps is None:
            eps = self.ln_eps
        N = len(x)
        mean = sum(x)/N
        var = sum((xi-mean)**2 for xi in x)/N
        inv_std = 1.0 / math.sqrt(var + eps)
        xhat = [(xi - mean) * inv_std for xi in x]
        y = [gamma_i * xhat_i + beta_i for gamma_i, xhat_i, beta_i in zip(gamma, xhat, beta)]
        # return y (normalized and scaled), plus saved stats for backward
        return y, {'xhat': xhat, 'mean': mean, 'inv_std': inv_std, 'gamma': list(gamma)}

    # LayerNorm backward: given upstream gradient dout (vector), and saved stats,
    # return dx (vector), dgamma (vector), dbeta (vector).
    def _layer_norm_backward(self, dout, stats):
        xhat = stats['xhat']
        inv_std = stats['inv_std']
        gamma = stats['gamma']
        N = len(dout)
        # dgamma = dout * xhat (elementwise)
        dgamma = [dout_i * xhat_i for dout_i, xhat_i in zip(dout, xhat)]
        dbeta = [dout_i for dout_i in dout]
        # dxhat = dout * gamma
        dxhat = [dout_i * g for dout_i, g in zip(dout, gamma)]
        sum_dxhat = sum(dxhat)
        sum_dxhat_xhat = sum(dxhat_i * xhat_i for dxhat_i, xhat_i in zip(dxhat, xhat))
        inv_N = 1.0 / N
        dx = [ (inv_N * inv_std) * (N*dxh - sum_dxhat - xh * sum_dxhat_xhat)
               for dxh, xh in zip(dxhat, xhat) ]
        return dx, dgamma, dbeta

    # ---------------------
    # Descriptor forward (single layer) with pre-LN
    # ---------------------
    def _desc_forward_layer(self, layer, seq):
        """
        Descriptor forward for a single layer using pre-LN:
          - ln_input = LayerNorm(input_vec)  # size = in_dim
          - x = M * ln_input                  # out_dim
          - Nk computed from P, x and phi     # out_dim
          - res = R * ln_input                # out_dim
          - out = res + Nk                    # residual add, no post-LN
        Returns:
            out_seq: list of output vectors (per position)
            intermediates: list of per-position intermediates for backprop
        """
        out_seq = []
        intermediates = []
        out_dim = len(layer['P'])
        # detect input dim from R's columns
        in_dim = len(layer['R'][0]) if len(layer['R'])>0 else 0
        for k, vec in enumerate(seq):
            # pre-LN on input_vec
            ln_input, ln_stats = self._layer_norm_forward(vec, layer['ln_gamma_in'], layer['ln_beta_in'])
            # linear projection to x using ln_input
            x = self._mat_vec(layer['M'], ln_input)  # length = out_dim
            Nk = [0.0]*out_dim
            phi_vals = {}
            for i in range(out_dim):
                for j in range(out_dim):
                    P_row = layer['P'][i][j]
                    for g in range(len(P_row)):
                        period = layer['periods'][i][j][g]
                        phi = math.cos(2*math.pi*k/period)
                        phi_vals[(i,j,g)] = phi
                        Nk[i] += P_row[g] * x[j] * phi
            # residual projection from ln_input
            res = self._mat_vec(layer['R'], ln_input)  # length = out_dim
            # final output is res + Nk (no post-LN)
            out_vec = [res[i] + Nk[i] for i in range(out_dim)]
            out_seq.append(out_vec)
            intermediates.append({'input_vec': vec, 'ln_input': ln_input, 'ln_stats': ln_stats,
                                  'x': x, 'phi_vals': phi_vals, 'Nk': Nk, 'res': res, 'out_vec': out_vec})
        return out_seq, intermediates

    # ---------------------
    # transformer forward with pre-LN
    # ---------------------
    def _softmax(self, arr):
        m = max(arr)
        exps = [math.exp(a - m) for a in arr]
        s = sum(exps)
        if s == 0.0:
            return [1.0/len(arr)]*len(arr)
        return [e/s for e in exps]

    def _transformer_forward_layer(self, layer, seq):
        """
        Transformer forward for a single interleaved transformer block using pre-LN:
          - For each position p:
              ln_seq = LayerNorm(seq[p])              # ln1: pre-attention
          - Compute Q,K,V from ln_seq
          - attn_out per pos computed from Q/K/V
          - attn_linear = Wo * attn_out + b
          - attn_res = seq + attn_linear             # residual add
          - ln_attn_res = LayerNorm(attn_res)        # ln2: pre-FFN
          - FFN applied to ln_attn_res -> fo
          - final = attn_res + fo
        Returns:
            new_seq: list of output vectors (per position)
            intermediates: dictionary with saved tensors for backprop
        """
        dim = layer['dim']
        L = len(seq)
        # compute ln_seq for each position (pre-attention LN)
        ln_seqs = []
        ln1_stats = []
        for p in range(L):
            ln_seq_p, stats1 = self._layer_norm_forward(seq[p], layer['ln1_gamma'], layer['ln1_beta'])
            ln_seqs.append(ln_seq_p)
            ln1_stats.append(stats1)
        # linear projections (from ln_seq)
        Qs = [self._mat_vec(layer['Wq'], x) for x in ln_seqs]
        Ks = [self._mat_vec(layer['Wk'], x) for x in ln_seqs]
        Vs = [self._mat_vec(layer['Wv'], x) for x in ln_seqs]
        sqrt_d = math.sqrt(dim) if dim>0 else 1.0
        attn_weights = []
        attn_outputs = []
        score_matrices = []
        for p in range(L):
            scores = [ (self._dot(Qs[p], Ks[q]) / sqrt_d) for q in range(L) ]
            ws = self._softmax(scores)
            attn_weights.append(ws)
            O_p = [0.0]*dim
            for q in range(L):
                a = ws[q]
                Vq = Vs[q]
                for i in range(dim):
                    O_p[i] += a * Vq[i]
            attn_outputs.append(O_p)
            score_matrices.append(scores)
        # linear output of attention
        attn_linear = []
        for p in range(L):
            Ao = self._mat_vec(layer['Wo'], attn_outputs[p])
            Ao = [Ao[i] + layer['b_o'][i] for i in range(dim)]
            attn_linear.append(Ao)
        # residual add: attn_res = seq + attn_linear
        attn_res = []
        for p in range(L):
            res = [seq[p][i] + attn_linear[p][i] for i in range(dim)]
            attn_res.append(res)
        # pre-FFN LayerNorm on attn_res (ln2)
        ln2_stats = []
        ln_attn_res = []
        for p in range(L):
            y2, stats2 = self._layer_norm_forward(attn_res[p], layer['ln2_gamma'], layer['ln2_beta'])
            ln_attn_res.append(y2)
            ln2_stats.append(stats2)
        # Feed-forward network applied to ln_attn_res
        FFN_h_pre = []
        FFN_h = []
        FFN_out = []
        final_vecs = []
        for p in range(L):
            pre = self._mat_vec(layer['W1'], ln_attn_res[p])
            pre = [pre[i] + layer['b1'][i] for i in range(len(pre))]
            h = [x if x>0 else 0.0 for x in pre]  # ReLU
            fo = self._mat_vec(layer['W2'], h)
            fo = [fo[i] + layer['b2'][i] for i in range(len(fo))]
            # residual connection for FFN: attn_res + fo
            final = [attn_res[p][i] + fo[i] for i in range(dim)]
            FFN_h_pre.append(pre); FFN_h.append(h); FFN_out.append(fo)
            final_vecs.append(final)
        intermediates = {
            'ln1_stats': ln1_stats, 'ln_seqs': ln_seqs,
            'Q': Qs, 'K': Ks, 'V': Vs,
            'scores': score_matrices, 'attn_weights': attn_weights,
            'attn_outputs': attn_outputs,
            'attn_linear': attn_linear, 'attn_res': attn_res,
            'ln2_stats': ln2_stats, 'ln_attn_res': ln_attn_res,
            'FFN_pre': FFN_h_pre, 'FFN_h': FFN_h, 'FFN_out': FFN_out,
            'seq_in': seq
        }
        return final_vecs, intermediates

    # overall describe (interleaved forward)
    def describe(self, seq):
        current = seq
        for idx in range(self.num_desc_layers):
            desc_layer = self.desc_layers[idx]
            current, _ = self._desc_forward_layer(desc_layer, current)
            if idx < len(self.trans_layers):
                trans_layer = self.trans_layers[idx]
                current, _ = self._transformer_forward_layer(trans_layer, current)
        return current

    # ---------- End-to-end gradient training with complete pre-LN backprop ----------
    def grad_train(self, seqs, t_list, max_iters=200, learning_rate=0.5, decay_rate=0.995,
                   print_every=10, tol=1e-9, continued=False):
        """
        End-to-end gradient descent for the pre-LN model.
        """
        if not continued:
            self.__init__(self.input_dim, self.model_dims, self.num_basis_list, ln_eps=self.ln_eps)
        history = []
        D_prev = float('inf')
        total_positions = sum(len(s) for s in seqs)
        if total_positions == 0:
            return history

        for it in range(max_iters):
            # initialize descriptor grads
            grad_desc_P = []
            grad_desc_M = []
            grad_desc_R = []
            grad_desc_ln_gamma_in = []
            grad_desc_ln_beta_in = []
            for l_idx, layer in enumerate(self.desc_layers):
                out_dim = len(layer['P'])
                num_basis = len(layer['P'][0][0]) if len(layer['P'])>0 and len(layer['P'][0])>0 else 0
                in_dim = len(layer['R'][0]) if len(layer['R'])>0 else 0
                # grad shapes: same as parameters
                grad_P = [[[0.0]*num_basis for _ in range(out_dim)] for _ in range(out_dim)]
                grad_M = [[0.0]*in_dim for _ in range(out_dim)]
                grad_R = [[0.0]*in_dim for _ in range(out_dim)]
                grad_desc_P.append(grad_P); grad_desc_M.append(grad_M); grad_desc_R.append(grad_R)
                grad_desc_ln_gamma_in.append([0.0]*in_dim); grad_desc_ln_beta_in.append([0.0]*in_dim)

            # initialize transformer grads
            grad_trans = []
            for l in self.trans_layers:
                dim = l['dim']
                def zero_mat(r,c): return [[0.0]*c for _ in range(r)]
                grad_trans.append({
                    'Wq': zero_mat(dim, dim), 'Wk': zero_mat(dim, dim), 'Wv': zero_mat(dim, dim),
                    'Wo': zero_mat(dim, dim), 'b_o': [0.0]*dim,
                    'W1': zero_mat(len(l['W1']), dim), 'b1':[0.0]*len(l['b1']),
                    'W2': zero_mat(len(l['W2']), len(l['W2'][0])), 'b2':[0.0]*len(l['b2']),
                    'ln1_gamma':[0.0]*dim, 'ln1_beta':[0.0]*dim,
                    'ln2_gamma':[0.0]*dim, 'ln2_beta':[0.0]*dim
                })
            # forward pass per sequence, storing intermediates
            all_blocks = []  # per-seq list of blocks (('desc',idx,interm) or ('trans',idx,interm))
            outputs_per_seq = []
            for seq in seqs:
                blocks = []
                current = seq
                for idx in range(self.num_desc_layers):
                    desc_layer = self.desc_layers[idx]
                    current, desc_inter = self._desc_forward_layer(desc_layer, current)
                    blocks.append(('desc', idx, desc_inter))
                    if idx < len(self.trans_layers):
                        trans_layer = self.trans_layers[idx]
                        current, trans_inter = self._transformer_forward_layer(trans_layer, current)
                        blocks.append(('trans', idx, trans_inter))
                outputs_per_seq.append(current)
                all_blocks.append(blocks)

            # backprop per sequence: create dY initial from loss (MSE)
            for s_idx, seq in enumerate(seqs):
                blocks = all_blocks[s_idx]
                outputs = outputs_per_seq[s_idx]
                L = len(seq)
                # initialize dY list for each position: dY[pos] is gradient wrt final block output
                dY = []
                target = t_list[s_idx]
                out_dim = len(outputs[0]) if outputs else 0
                for k in range(L):
                    y = outputs[k]
                    # gradient of MSE: dL/dy = 2*(y - t) / total_positions
                    dY.append([2*(y_i - target_i)/total_positions for y_i, target_i in zip(y, target)])
                # process blocks in reverse
                for btype, idx, interm in reversed(blocks):
                    if btype == 'desc':
                        # descriptor backprop (position-wise), with pre-LN
                        desc = self.desc_layers[idx]
                        out_dim = len(desc['P'])
                        in_dim = len(desc['R'][0]) if len(desc['R'])>0 else 0
                        dX_prev = []
                        for k in range(L):
                            # incoming gradient w.r.t descriptor layer output (out_vec = res + Nk)
                            d_out = dY[k]
                            # split gradient: both res and Nk receive d_out
                            d_res = list(d_out)
                            d_Nk = list(d_out)
                            # grad R: d_res contributes to R (R * ln_input = res)
                            ln_input = interm[k]['ln_input']
                            for i in range(out_dim):
                                for j in range(in_dim):
                                    grad_desc_R[idx][i][j] += d_res[i] * ln_input[j]
                            # contribution to ln_input from R: R^T * d_res
                            R_T = self._transpose(desc['R'])
                            d_ln_from_R = self._mat_vec(R_T, d_res)
                            # Now backprop through Nk computation (same as original logic), to obtain d_x
                            x = interm[k]['x']  # x = M * ln_input, length = out_dim
                            phi_vals = interm[k]['phi_vals']
                            # grad P: dP[i][j][g] += d_Nk[i] * x[j] * phi
                            for i in range(out_dim):
                                for j in range(out_dim):
                                    for g in range(len(desc['P'][i][j])):
                                        grad_desc_P[idx][i][j][g] += d_Nk[i] * x[j] * phi_vals[(i,j,g)]
                            # d_x[j] = sum_i d_Nk[i] * sum_g P[i][j][g] * phi
                            d_x = [0.0]*len(x)
                            for j in range(len(x)):
                                s = 0.0
                                for i in range(out_dim):
                                    for g in range(len(desc['P'][i][j])):
                                        s += d_Nk[i] * desc['P'][i][j][g] * phi_vals[(i,j,g)]
                                d_x[j] = s
                            # grad M: dM[j][d] += d_x[j] * ln_input[d]
                            for j in range(len(d_x)):
                                for d in range(in_dim):
                                    grad_desc_M[idx][j][d] += d_x[j] * ln_input[d]
                            # propagate to ln_input from M-path: M^T * d_x
                            M_T = self._transpose(desc['M'])
                            d_ln_from_M = self._mat_vec(M_T, d_x)
                            # total gradient wrt ln_input is sum of contributions from R and from M-path
                            d_ln_input = [d_ln_from_R[j] + d_ln_from_M[j] for j in range(in_dim)]
                            # now backprop through pre-LN: ln_input = LayerNorm(input_vec)
                            dx_input_vec, dgamma_vec, dbeta_vec = self._layer_norm_backward(d_ln_input, interm[k]['ln_stats'])
                            # accumulate ln param grads (these are per-layer, per-input-dim)
                            for j in range(in_dim):
                                grad_desc_ln_gamma_in[idx][j] += dgamma_vec[j]
                                grad_desc_ln_beta_in[idx][j] += dbeta_vec[j]
                            dX_prev.append(dx_input_vec)
                        # set dY = dX_prev for next (previous) block
                        dY = dX_prev

                    else:  # transformer block backprop (sequence-coupled), pre-LN logic
                        trans = self.trans_layers[idx]
                        dim = trans['dim']
                        L = len(interm['seq_in'])
                        # extract intermediates
                        ln1_stats = interm['ln1_stats']          # stats for LN before attention (per pos)
                        ln_seqs = interm['ln_seqs']              # ln_seq inputs used for Q/K/V (per pos)
                        Qs = interm['Q']; Ks = interm['K']; Vs = interm['V']
                        attn_w = interm['attn_weights']; attn_outs = interm['attn_outputs']
                        attn_linear = interm['attn_linear']; attn_res = interm['attn_res']
                        ln2_stats = interm['ln2_stats']          # stats for LN before FFN (per pos)
                        ln_attn_res = interm['ln_attn_res']      # inputs to FFN
                        FFN_pre = interm['FFN_pre']; FFN_h = interm['FFN_h']; FFN_out = interm['FFN_out']
                        seq_in = interm['seq_in']
                        sqrt_d = math.sqrt(dim) if dim>0 else 1.0

                        # accumulators for grads for this transformer layer (per sequence)
                        gWq = [[0.0]*dim for _ in range(dim)]
                        gWk = [[0.0]*dim for _ in range(dim)]
                        gWv = [[0.0]*dim for _ in range(dim)]
                        gWo = [[0.0]*dim for _ in range(dim)]
                        gb_o = [0.0]*dim
                        gW1 = [[0.0]*dim for _ in range(len(trans['W1']))]
                        gb1 = [0.0]*len(trans['b1'])
                        gW2 = [[0.0]*len(trans['W2'][0]) for _ in range(len(trans['W2']))]
                        gb2 = [0.0]*len(trans['b2'])
                        # LayerNorm grads
                        gln1_gamma = [0.0]*dim; gln1_beta = [0.0]*dim
                        gln2_gamma = [0.0]*dim; gln2_beta = [0.0]*dim

                        # gradients wrt inputs to transformer (per position) -- these are gradients w.r.t seq elements
                        d_inputs = [ [0.0]*len(seq_in[0]) for _ in range(L) ]

                        # ---------- Backprop flow for pre-LN transformer ----------
                        # dY is gradient wrt final output: final = attn_res + fo
                        # So residual path: d_attn_res_from_final = dY
                        # and FFN path: d_fo = dY

                        # Step A: Backprop through FFN to obtain gradient wrt ln_attn_res (FFN input)
                        # ff input = ln_attn_res
                        d_ln_attn_res = [ [0.0]*dim for _ in range(L) ]
                        for p in range(L):
                            # gradient arriving to FFN output (fo) is dY[p]
                            dfo = dY[p]
                            h = FFN_h[p]
                            # gW2, gb2
                            for i in range(len(dfo)):
                                for j in range(len(h)):
                                    gW2[i][j] += dfo[i] * h[j]
                                gb2[i] += dfo[i]
                            # compute dh = W2^T * dfo
                            dh = [0.0]*len(h)
                            for j in range(len(h)):
                                s = 0.0
                                for i in range(len(dfo)):
                                    s += trans['W2'][i][j] * dfo[i]
                                dh[j] = s
                            # backprop through ReLU: dpre = dh * (pre > 0)
                            pre = FFN_pre[p]
                            dpre = [dh_j if pre_j>0 else 0.0 for dh_j, pre_j in zip(dh, pre)]
                            # gW1, gb1
                            for i in range(len(dpre)):
                                for j in range(dim):
                                    gW1[i][j] += dpre[i] * ln_attn_res[p][j]
                                gb1[i] += dpre[i]
                            # contribution to ln_attn_res from FFN: d_ln_attn_res += W1^T * dpre
                            for j in range(dim):
                                s = 0.0
                                for i in range(len(dpre)):
                                    s += trans['W1'][i][j] * dpre[i]
                                d_ln_attn_res[p][j] += s

                        # Step B: Backprop through pre-FFN LayerNorm (ln2): obtain gradient wrt attn_res and ln2 params
                        d_attn_res_from_ln = [ [0.0]*dim for _ in range(L) ]
                        for p in range(L):
                            stats2 = ln2_stats[p]
                            dout2 = d_ln_attn_res[p]
                            dx2, dgamma2_vec, dbeta2_vec = self._layer_norm_backward(dout2, stats2)
                            for i in range(dim):
                                gln2_gamma[i] += dgamma2_vec[i]
                                gln2_beta[i] += dbeta2_vec[i]
                            d_attn_res_from_ln[p] = dx2

                        # Step C: Combine residual and ln-derived gradients at attn_res:
                        # total d_attn_res = dY (from final residual) + d_attn_res_from_ln
                        d_attn_res = [ [0.0]*dim for _ in range(L) ]
                        for p in range(L):
                            for i in range(dim):
                                d_attn_res[p][i] = dY[p][i] + d_attn_res_from_ln[p][i]

                        # Step D: attn_res = seq + attn_linear
                        # => d_seq (from residual path) += d_attn_res ; d_attn_linear = d_attn_res
                        d_attn_linear = [ [0.0]*dim for _ in range(L) ]
                        for p in range(L):
                            for i in range(dim):
                                d_inputs[p][i] += d_attn_res[p][i]   # accumulate gradient back to seq input (residual path)
                                d_attn_linear[p][i] += d_attn_res[p][i]

                        # Step E: attn_linear = Wo * attn_out + b_o
                        # backprop to Wo, b_o, and get dattno = Wo^T * d_attn_linear
                        dattno = [ [0.0]*dim for _ in range(L) ]
                        Wo_T = self._transpose(trans['Wo'])
                        for p in range(L):
                            for i in range(dim):
                                for j in range(dim):
                                    gWo[i][j] += d_attn_linear[p][i] * attn_outs[p][j]
                                gb_o[i] += d_attn_linear[p][i]
                            # compute Wo^T * d_attn_linear[p]
                            for j in range(dim):
                                s = 0.0
                                for i in range(dim):
                                    s += Wo_T[j][i] * d_attn_linear[p][i]
                                dattno[p][j] = s

                        # Step F: Attention backprop: convert dattno into gradients on V and attention weights
                        dV = [ [0.0]*dim for _ in range(L) ]  # gradient wrt V_q
                        da_matrix = [ [0.0]*L for _ in range(L) ]  # da[p][q]
                        for p in range(L):
                            for q in range(L):
                                a = attn_w[p][q]
                                # dV[q] += a * dattno[p]
                                for i in range(dim):
                                    dV[q][i] += a * dattno[p][i]
                                # da_pq = dot(dattno[p], Vq)
                                s = 0.0
                                Vq = Vs[q]
                                for i in range(dim):
                                    s += dattno[p][i] * Vq[i]
                                da_matrix[p][q] = s

                        # Step G: Vq = Wv * ln_seq[q] => gWv += outer(dV[q], ln_seq[q]); and d_ln_seq_from_V += Wv^T * dV[q]
                        Wv_T = self._transpose(trans['Wv'])
                        # prepare container for accumulating d_ln_seq from all Q/K/V paths
                        d_ln_seq = [ [0.0]*dim for _ in range(L) ]
                        for q in range(L):
                            ln_q = ln_seqs[q]
                            for i in range(dim):
                                for j in range(len(ln_q)):
                                    gWv[i][j] += dV[q][i] * ln_q[j]
                            # propagate to ln_seq from V path
                            for j in range(len(ln_q)):
                                s = 0.0
                                for i in range(dim):
                                    s += Wv_T[j][i] * dV[q][i]
                                d_ln_seq[q][j] += s

                        # Step H: Softmax backprop: convert da_matrix -> dscores using softmax jacobian for each p
                        dscores = [ [0.0]*L for _ in range(L) ]  # dscores[p][r]
                        for p in range(L):
                            a = attn_w[p]
                            da = da_matrix[p]
                            dot_ad = 0.0
                            for q in range(L):
                                dot_ad += a[q] * da[q]
                            for r in range(L):
                                dscores[p][r] = a[r] * (da[r] - dot_ad)

                        # Step I: scores[p][q] = dot(Qp, Kq) / sqrt_d
                        # => dscores contributes to dQ and dK
                        dQ = [ [0.0]*dim for _ in range(L) ]
                        dK = [ [0.0]*dim for _ in range(L) ]
                        for p in range(L):
                            for q in range(L):
                                coeff = dscores[p][q] / sqrt_d
                                # accumulate into dQ[p] using Kq
                                for i in range(dim):
                                    dQ[p][i] += coeff * Ks[q][i]
                                # accumulate into dK[q] using Qp
                                for i in range(dim):
                                    dK[q][i] += coeff * Qs[p][i]

                        # Step J: Qp = Wq * ln_seq[p]; Kq = Wk * ln_seq[q]
                        # => gWq += outer(dQ[p], ln_seq[p]), gWk += outer(dK[q], ln_seq[q])
                        Wq_T = self._transpose(trans['Wq'])
                        Wk_T = self._transpose(trans['Wk'])
                        for p in range(L):
                            ln_p = ln_seqs[p]
                            # gWq
                            for i in range(dim):
                                for j in range(len(ln_p)):
                                    gWq[i][j] += dQ[p][i] * ln_p[j]
                            # propagate to ln_seq from Q: Wq^T * dQ[p]
                            for j in range(len(ln_p)):
                                s = 0.0
                                for i in range(dim):
                                    s += Wq_T[j][i] * dQ[p][i]
                                d_ln_seq[p][j] += s
                        for q in range(L):
                            ln_q = ln_seqs[q]
                            for i in range(dim):
                                for j in range(len(ln_q)):
                                    gWk[i][j] += dK[q][i] * ln_q[j]
                            # propagate to ln_seq from K: Wk^T * dK[q]
                            for j in range(len(ln_q)):
                                s = 0.0
                                for i in range(dim):
                                    s += Wk_T[j][i] * dK[q][i]
                                d_ln_seq[q][j] += s

                        # Now we have accumulated d_ln_seq from V/Q/K paths in d_ln_seq.
                        # Step K: Backprop through pre-attention LayerNorm (ln1) to obtain gradient wrt seq (previous layer outputs)
                        for p in range(L):
                            stats1 = ln1_stats[p]
                            dout1 = d_ln_seq[p]
                            dx1, dgamma1_vec, dbeta1_vec = self._layer_norm_backward(dout1, stats1)
                            for i in range(dim):
                                gln1_gamma[i] += dgamma1_vec[i]
                                gln1_beta[i] += dbeta1_vec[i]
                            # dx1 is gradient with respect to seq[p] (previous layer output)
                            # add to d_inputs (which already contains residual contributions)
                            for j in range(len(dx1)):
                                d_inputs[p][j] += dx1[j]

                        # Step L: accumulate transformer grads into global grad_trans[idx]
                        gtrans = grad_trans[idx]
                        # sum into gtrans structures
                        for i in range(dim):
                            for j in range(dim):
                                gtrans['Wq'][i][j] += gWq[i][j]
                                gtrans['Wk'][i][j] += gWk[i][j]
                                gtrans['Wv'][i][j] += gWv[i][j]
                                gtrans['Wo'][i][j] += gWo[i][j]
                            gtrans['b_o'][i] += gb_o[i]
                        for i in range(len(gW1)):
                            for j in range(len(gW1[0])):
                                gtrans['W1'][i][j] += gW1[i][j]
                            gtrans['b1'][i] += gb1[i]
                        for i in range(len(gW2)):
                            for j in range(len(gW2[0])):
                                gtrans['W2'][i][j] += gW2[i][j]
                            gtrans['b2'][i] += gb2[i]
                        # accumulate layernorm grads
                        for i in range(dim):
                            gtrans['ln1_gamma'][i] += gln1_gamma[i]
                            gtrans['ln1_beta'][i] += gln1_beta[i]
                            gtrans['ln2_gamma'][i] += gln2_gamma[i]
                            gtrans['ln2_beta'][i] += gln2_beta[i]

                        # set dY for next (previous) block to d_inputs
                        dY = d_inputs
                # end reversed blocks loop for this sequence
            # end per-sequence backprop accumulation

            # ---------------- Update parameters ----------------
            # Update descriptor parameters
            for idx in range(len(self.desc_layers)):
                desc = self.desc_layers[idx]
                for i in range(len(desc['P'])):
                    for j in range(len(desc['P'][i])):
                        for g in range(len(desc['P'][i][j])):
                            desc['P'][i][j][g] -= learning_rate * grad_desc_P[idx][i][j][g]
                for i in range(len(desc['M'])):
                    for j in range(len(desc['M'][i])):
                        desc['M'][i][j] -= learning_rate * grad_desc_M[idx][i][j]
                # update residual projection R
                for i in range(len(desc['R'])):
                    for j in range(len(desc['R'][0])):
                        desc['R'][i][j] -= learning_rate * grad_desc_R[idx][i][j]
                # update LayerNorm params for descriptor input
                for j in range(len(desc['ln_gamma_in'])):
                    desc['ln_gamma_in'][j] -= learning_rate * grad_desc_ln_gamma_in[idx][j]
                    desc['ln_beta_in'][j] -= learning_rate * grad_desc_ln_beta_in[idx][j]

            # Update transformer parameters
            for idx in range(len(self.trans_layers)):
                trans = self.trans_layers[idx]
                g = grad_trans[idx]
                for i in range(len(trans['Wq'])):
                    for j in range(len(trans['Wq'][0])):
                        trans['Wq'][i][j] -= learning_rate * g['Wq'][i][j]
                for i in range(len(trans['Wk'])):
                    for j in range(len(trans['Wk'][0])):
                        trans['Wk'][i][j] -= learning_rate * g['Wk'][i][j]
                for i in range(len(trans['Wv'])):
                    for j in range(len(trans['Wv'][0])):
                        trans['Wv'][i][j] -= learning_rate * g['Wv'][i][j]
                for i in range(len(trans['Wo'])):
                    for j in range(len(trans['Wo'][0])):
                        trans['Wo'][i][j] -= learning_rate * g['Wo'][i][j]
                for i in range(len(trans['b_o'])):
                    trans['b_o'][i] -= learning_rate * g['b_o'][i]
                for i in range(len(trans['W1'])):
                    for j in range(len(trans['W1'][0])):
                        trans['W1'][i][j] -= learning_rate * g['W1'][i][j]
                for i in range(len(trans['b1'])):
                    trans['b1'][i] -= learning_rate * g['b1'][i]
                for i in range(len(trans['W2'])):
                    for j in range(len(trans['W2'][0])):
                        trans['W2'][i][j] -= learning_rate * g['W2'][i][j]
                for i in range(len(trans['b2'])):
                    trans['b2'][i] -= learning_rate * g['b2'][i]
                # update layernorm parameters (pre-LN)
                for i in range(len(trans['ln1_gamma'])):
                    trans['ln1_gamma'][i] -= learning_rate * g['ln1_gamma'][i]
                    trans['ln1_beta'][i] -= learning_rate * g['ln1_beta'][i]
                    trans['ln2_gamma'][i] -= learning_rate * g['ln2_gamma'][i]
                    trans['ln2_beta'][i] -= learning_rate * g['ln2_beta'][i]

            # compute loss for reporting (training set)
            total_loss = 0.0
            total_count = 0
            for seq, t in zip(seqs, t_list):
                outs = self.describe(seq)
                for vec in outs:
                    err = [vec_i - t_i for vec_i, t_i in zip(vec, t)]
                    total_loss += sum(e*e for e in err)
                    total_count += 1
            current_D = total_loss / total_count if total_count>0 else 0.0
            history.append(current_D)
            if it % print_every == 0 or it == max_iters-1:
                print(f"DDNet GD Iter {it:3d}: D = {current_D:.6e}, LR = {learning_rate:.6f}")
            if current_D >= D_prev - tol:
                print(f"Converged after {it+1} iterations.")
                break
            D_prev = current_D
            learning_rate *= decay_rate
        self.trained = True
        return history

    # predict average
    def predict_t(self, seq):
        outputs = self.describe(seq)
        if not outputs:
            return [0.0]*self.model_dims[-1]
        t_pred = [0.0]*len(outputs[0])
        for vec in outputs:
            for i in range(len(vec)):
                t_pred[i] += vec[i]
        return [x/len(outputs) for x in t_pred]

    def auto_train(self, seqs, mode='gap', max_iters=200, learning_rate=0.5, decay_rate=0.995,
               print_every=10, tol=1e-9, continued=False):
        """
        Auto-training using input sequence vectors as targets.

        Parameters
        ----------
        seqs : list of sequences, each sequence is list of input vectors (length=input_dim)
        mode : 'gap' or 'reg'
            'gap' - predict current vector at position k (target = seq[k])
            'reg' - predict next vector at position k (target = seq[k+1]); the last position of each sequence has no target and is skipped
        The method performs full end-to-end gradient descent similar to grad_train,
        but constructs per-position targets from the input sequences and computes
        loss between a decoded network output (read_W * y + read_b) and the input-target vectors.
        """
        assert mode in ('gap', 'reg'), "mode must be 'gap' or 'reg'"
        if not continued:
            # reinitialize model (keeps same architecture but re-randomizes params)
            self.__init__(self.input_dim, self.model_dims, self.num_basis_list, ln_eps=self.ln_eps)
        history = []
        D_prev = float('inf')

        # compute total number of target positions for normalization (gap: sum len(seq); reg: sum (len(seq)-1))
        total_targets = 0
        for seq in seqs:
            if mode == 'gap':
                total_targets += len(seq)
            else:  # reg
                total_targets += max(0, len(seq)-1)
        if total_targets == 0:
            return history

        for it in range(max_iters):
            # initialize descriptor grads
            grad_desc_P = []
            grad_desc_M = []
            grad_desc_R = []
            grad_desc_ln_gamma_in = []
            grad_desc_ln_beta_in = []
            for l_idx, layer in enumerate(self.desc_layers):
                out_dim_layer = len(layer['P'])
                num_basis = len(layer['P'][0][0]) if len(layer['P'])>0 and len(layer['P'][0])>0 else 0
                in_dim = len(layer['R'][0]) if len(layer['R'])>0 else 0
                # grad shapes: same as parameters
                grad_P = [[[0.0]*num_basis for _ in range(out_dim_layer)] for _ in range(out_dim_layer)]
                grad_M = [[0.0]*in_dim for _ in range(out_dim_layer)]
                grad_R = [[0.0]*in_dim for _ in range(out_dim_layer)]
                grad_desc_P.append(grad_P); grad_desc_M.append(grad_M); grad_desc_R.append(grad_R)
                grad_desc_ln_gamma_in.append([0.0]*in_dim); grad_desc_ln_beta_in.append([0.0]*in_dim)

            # initialize transformer grads
            grad_trans = []
            for l in self.trans_layers:
                dim = l['dim']
                def zero_mat(r,c): return [[0.0]*c for _ in range(r)]
                grad_trans.append({
                    'Wq': zero_mat(dim, dim), 'Wk': zero_mat(dim, dim), 'Wv': zero_mat(dim, dim),
                    'Wo': zero_mat(dim, dim), 'b_o': [0.0]*dim,
                    'W1': zero_mat(len(l['W1']), dim), 'b1':[0.0]*len(l['b1']),
                    'W2': zero_mat(len(l['W2']), len(l['W2'][0])), 'b2':[0.0]*len(l['b2']),
                    'ln1_gamma':[0.0]*dim, 'ln1_beta':[0.0]*dim,
                    'ln2_gamma':[0.0]*dim, 'ln2_beta':[0.0]*dim
                })

            # ---------- FORWARD PASS (store intermediates) ----------
            all_blocks = []
            outputs_per_seq = []
            for seq in seqs:
                blocks = []
                current = seq
                for idx in range(self.num_desc_layers):
                    desc_layer = self.desc_layers[idx]
                    current, desc_inter = self._desc_forward_layer(desc_layer, current)
                    blocks.append(('desc', idx, desc_inter))
                    if idx < len(self.trans_layers):
                        trans_layer = self.trans_layers[idx]
                        current, trans_inter = self._transformer_forward_layer(trans_layer, current)
                        blocks.append(('trans', idx, trans_inter))
                outputs_per_seq.append(current)
                all_blocks.append(blocks)

            # ---------- Determine actual network output dimension (fix for IndexError) ----------
            any_out_dim = None
            for out_seq in outputs_per_seq:
                if len(out_seq) > 0:
                    any_out_dim = len(out_seq[0])
                    break
            if any_out_dim is None:
                # nothing to train (all sequences empty)
                return history
            out_dim = any_out_dim

            # initialize readout grads using actual out_dim
            grad_read_W = [[0.0]*out_dim for _ in range(self.input_dim)]
            grad_read_b = [0.0]*self.input_dim

            # ---------- BACKPROP per sequence ----------
            for s_idx, seq in enumerate(seqs):
                blocks = all_blocks[s_idx]
                outputs = outputs_per_seq[s_idx]
                L = len(seq)
                # initialize dY list for each position (gradient wrt final model output y)
                dY = []
                for k in range(L):
                    has_target = (mode == 'gap') or (mode == 'reg' and k < L-1)
                    if not has_target:
                        dY.append([0.0]*out_dim)
                        continue
                    # determine target vector (in input space)
                    if mode == 'gap':
                        target_vec = seq[k]
                    else:
                        target_vec = seq[k+1]
                    y = outputs[k]
                    # safety: if y length differs from out_dim, use local_out
                    local_out = min(len(y), out_dim)
                    # decoded = read_W * y + read_b
                    decoded = [0.0]*self.input_dim
                    for i in range(self.input_dim):
                        s = 0.0
                        row = self.read_W[i]
                        for j in range(local_out):
                            s += row[j] * y[j]
                        decoded[i] = s + self.read_b[i]
                    # error and readout grads (normalized by total_targets)
                    err = [decoded_i - t_i for decoded_i, t_i in zip(decoded, target_vec)]
                    for i in range(self.input_dim):
                        coef = 2.0 * err[i] / total_targets
                        grad_read_b[i] += coef
                        for j in range(local_out):
                            grad_read_W[i][j] += coef * y[j]
                    # chain to network output space: dL/dy = 2 * W_read^T * err / total_targets
                    d_y = [0.0]*out_dim
                    for j in range(local_out):
                        s = 0.0
                        for i in range(self.input_dim):
                            s += self.read_W[i][j] * err[i]
                        d_y[j] = 2.0 * s / total_targets
                    dY.append(d_y)

                # reverse through stored blocks (same structure as grad_train)
                for btype, idx, interm in reversed(blocks):
                    if btype == 'desc':
                        desc = self.desc_layers[idx]
                        out_dim_desc = len(desc['P'])
                        in_dim = len(desc['R'][0]) if len(desc['R'])>0 else 0
                        dX_prev = []
                        for k in range(L):
                            d_out = dY[k]
                            # split gradient: res + Nk
                            d_res = list(d_out)
                            d_Nk = list(d_out)
                            ln_input = interm[k]['ln_input']
                            # grad R
                            for i in range(out_dim_desc):
                                for j in range(in_dim):
                                    grad_desc_R[idx][i][j] += d_res[i] * ln_input[j]
                            R_T = self._transpose(desc['R'])
                            d_ln_from_R = self._mat_vec(R_T, d_res)
                            # Nk backprop -> dP and d_x
                            x = interm[k]['x']
                            phi_vals = interm[k]['phi_vals']
                            for i in range(out_dim_desc):
                                for j in range(out_dim_desc):
                                    for g in range(len(desc['P'][i][j])):
                                        grad_desc_P[idx][i][j][g] += d_Nk[i] * x[j] * phi_vals[(i,j,g)]
                            d_x = [0.0]*len(x)
                            for j in range(len(x)):
                                s = 0.0
                                for i in range(out_dim_desc):
                                    for g in range(len(desc['P'][i][j])):
                                        s += d_Nk[i] * desc['P'][i][j][g] * phi_vals[(i,j,g)]
                                d_x[j] = s
                            # grad M
                            for j in range(len(d_x)):
                                for d in range(in_dim):
                                    grad_desc_M[idx][j][d] += d_x[j] * ln_input[d]
                            M_T = self._transpose(desc['M'])
                            d_ln_from_M = self._mat_vec(M_T, d_x)
                            # total wrt ln_input
                            d_ln_input = [d_ln_from_R[j] + d_ln_from_M[j] for j in range(in_dim)]
                            dx_input_vec, dgamma_vec, dbeta_vec = self._layer_norm_backward(d_ln_input, interm[k]['ln_stats'])
                            for j in range(in_dim):
                                grad_desc_ln_gamma_in[idx][j] += dgamma_vec[j]
                                grad_desc_ln_beta_in[idx][j] += dbeta_vec[j]
                            dX_prev.append(dx_input_vec)
                        dY = dX_prev

                    else:
                        # transformer backprop (pre-LN) - same as grad_train implementation
                        trans = self.trans_layers[idx]
                        dim = trans['dim']
                        Lblock = len(interm['seq_in'])
                        ln1_stats = interm['ln1_stats']
                        ln_seqs = interm['ln_seqs']
                        Qs = interm['Q']; Ks = interm['K']; Vs = interm['V']
                        attn_w = interm['attn_weights']; attn_outs = interm['attn_outputs']
                        attn_linear = interm['attn_linear']; attn_res = interm['attn_res']
                        ln2_stats = interm['ln2_stats']; ln_attn_res = interm['ln_attn_res']
                        FFN_pre = interm['FFN_pre']; FFN_h = interm['FFN_h']; FFN_out = interm['FFN_out']
                        seq_in = interm['seq_in']
                        sqrt_d = math.sqrt(dim) if dim>0 else 1.0

                        # per-block accumulators
                        gWq = [[0.0]*dim for _ in range(dim)]
                        gWk = [[0.0]*dim for _ in range(dim)]
                        gWv = [[0.0]*dim for _ in range(dim)]
                        gWo = [[0.0]*dim for _ in range(dim)]
                        gb_o = [0.0]*dim
                        gW1 = [[0.0]*dim for _ in range(len(trans['W1']))]
                        gb1 = [0.0]*len(trans['b1'])
                        gW2 = [[0.0]*len(trans['W2'][0]) for _ in range(len(trans['W2']))]
                        gb2 = [0.0]*len(trans['b2'])
                        gln1_gamma = [0.0]*dim; gln1_beta = [0.0]*dim
                        gln2_gamma = [0.0]*dim; gln2_beta = [0.0]*dim

                        d_inputs = [ [0.0]*len(seq_in[0]) for _ in range(Lblock) ]

                        # Step A: FFN backprop -> d_ln_attn_res
                        d_ln_attn_res = [ [0.0]*dim for _ in range(Lblock) ]
                        for p in range(Lblock):
                            dfo = dY[p]
                            h = FFN_h[p]
                            # gW2, gb2
                            for i in range(len(dfo)):
                                for j in range(len(h)):
                                    gW2[i][j] += dfo[i] * h[j]
                                gb2[i] += dfo[i]
                            # dh = W2^T * dfo
                            dh = [0.0]*len(h)
                            for j in range(len(h)):
                                s = 0.0
                                for i in range(len(dfo)):
                                    s += trans['W2'][i][j] * dfo[i]
                                dh[j] = s
                            pre = FFN_pre[p]
                            dpre = [dh_j if pre_j>0 else 0.0 for dh_j, pre_j in zip(dh, pre)]
                            for i in range(len(dpre)):
                                for j in range(dim):
                                    gW1[i][j] += dpre[i] * ln_attn_res[p][j]
                                gb1[i] += dpre[i]
                            for j in range(dim):
                                s = 0.0
                                for i in range(len(dpre)):
                                    s += trans['W1'][i][j] * dpre[i]
                                d_ln_attn_res[p][j] += s

                        # Step B: pre-FFN LayerNorm backward -> d_attn_res_from_ln
                        d_attn_res_from_ln = [ [0.0]*dim for _ in range(Lblock) ]
                        for p in range(Lblock):
                            stats2 = ln2_stats[p]
                            dout2 = d_ln_attn_res[p]
                            dx2, dgamma2_vec, dbeta2_vec = self._layer_norm_backward(dout2, stats2)
                            for i in range(dim):
                                gln2_gamma[i] += dgamma2_vec[i]
                                gln2_beta[i] += dbeta2_vec[i]
                            d_attn_res_from_ln[p] = dx2

                        # Step C: combine residual and ln-derived grads
                        d_attn_res = [ [0.0]*dim for _ in range(Lblock) ]
                        for p in range(Lblock):
                            for i in range(dim):
                                d_attn_res[p][i] = dY[p][i] + d_attn_res_from_ln[p][i]

                        # Step D: attn_res = seq + attn_linear -> residual to seq, attn_linear grad
                        d_attn_linear = [ [0.0]*dim for _ in range(Lblock) ]
                        for p in range(Lblock):
                            for i in range(dim):
                                d_inputs[p][i] += d_attn_res[p][i]
                                d_attn_linear[p][i] += d_attn_res[p][i]

                        # Step E: attn_linear = Wo * attn_out + b_o
                        dattno = [ [0.0]*dim for _ in range(Lblock) ]
                        Wo_T = self._transpose(trans['Wo'])
                        for p in range(Lblock):
                            for i in range(dim):
                                for j in range(dim):
                                    gWo[i][j] += d_attn_linear[p][i] * attn_outs[p][j]
                                gb_o[i] += d_attn_linear[p][i]
                            for j in range(dim):
                                s = 0.0
                                for i in range(dim):
                                    s += Wo_T[j][i] * d_attn_linear[p][i]
                                dattno[p][j] = s

                        # Step F: attention backprop -> dV and da_matrix
                        dV = [ [0.0]*dim for _ in range(Lblock) ]
                        da_matrix = [ [0.0]*Lblock for _ in range(Lblock) ]
                        for p in range(Lblock):
                            for q in range(Lblock):
                                a = attn_w[p][q]
                                for i in range(dim):
                                    dV[q][i] += a * dattno[p][i]
                                s = 0.0
                                Vq = Vs[q]
                                for i in range(dim):
                                    s += dattno[p][i] * Vq[i]
                                da_matrix[p][q] = s

                        # Step G: Vq = Wv * ln_seq[q]
                        Wv_T = self._transpose(trans['Wv'])
                        d_ln_seq = [ [0.0]*dim for _ in range(Lblock) ]
                        for q in range(Lblock):
                            ln_q = ln_seqs[q]
                            for i in range(dim):
                                for j in range(len(ln_q)):
                                    gWv[i][j] += dV[q][i] * ln_q[j]
                            for j in range(len(ln_q)):
                                s = 0.0
                                for i in range(dim):
                                    s += Wv_T[j][i] * dV[q][i]
                                d_ln_seq[q][j] += s

                        # Step H: softmax jacobian -> dscores
                        dscores = [ [0.0]*Lblock for _ in range(Lblock) ]
                        for p in range(Lblock):
                            a = attn_w[p]
                            da = da_matrix[p]
                            dot_ad = 0.0
                            for q in range(Lblock):
                                dot_ad += a[q] * da[q]
                            for r in range(Lblock):
                                dscores[p][r] = a[r] * (da[r] - dot_ad)

                        # Step I: dscores -> dQ and dK
                        dQ = [ [0.0]*dim for _ in range(Lblock) ]
                        dK = [ [0.0]*dim for _ in range(Lblock) ]
                        for p in range(Lblock):
                            for q in range(Lblock):
                                coeff = dscores[p][q] / sqrt_d
                                for i in range(dim):
                                    dQ[p][i] += coeff * Ks[q][i]
                                for i in range(dim):
                                    dK[q][i] += coeff * Qs[p][i]

                        # Step J: Q = Wq * ln_seq, K = Wk * ln_seq
                        Wq_T = self._transpose(trans['Wq'])
                        Wk_T = self._transpose(trans['Wk'])
                        for p in range(Lblock):
                            ln_p = ln_seqs[p]
                            for i in range(dim):
                                for j in range(len(ln_p)):
                                    gWq[i][j] += dQ[p][i] * ln_p[j]
                            for j in range(len(ln_p)):
                                s = 0.0
                                for i in range(dim):
                                    s += Wq_T[j][i] * dQ[p][i]
                                d_ln_seq[p][j] += s
                        for q in range(Lblock):
                            ln_q = ln_seqs[q]
                            for i in range(dim):
                                for j in range(len(ln_q)):
                                    gWk[i][j] += dK[q][i] * ln_q[j]
                            for j in range(len(ln_q)):
                                s = 0.0
                                for i in range(dim):
                                    s += Wk_T[j][i] * dK[q][i]
                                d_ln_seq[q][j] += s

                        # Step K: pre-attention LayerNorm backward -> gradients wrt seq (previous layer outputs)
                        for p in range(Lblock):
                            stats1 = ln1_stats[p]
                            dout1 = d_ln_seq[p]
                            dx1, dgamma1_vec, dbeta1_vec = self._layer_norm_backward(dout1, stats1)
                            for i in range(dim):
                                gln1_gamma[i] += dgamma1_vec[i]
                                gln1_beta[i] += dbeta1_vec[i]
                            for j in range(len(dx1)):
                                d_inputs[p][j] += dx1[j]

                        # Step L: accumulate this transformer's grads into global grad_trans
                        gtrans = grad_trans[idx]
                        for i in range(dim):
                            for j in range(dim):
                                gtrans['Wq'][i][j] += gWq[i][j]
                                gtrans['Wk'][i][j] += gWk[i][j]
                                gtrans['Wv'][i][j] += gWv[i][j]
                                gtrans['Wo'][i][j] += gWo[i][j]
                            gtrans['b_o'][i] += gb_o[i]
                        for i in range(len(gW1)):
                            for j in range(len(gW1[0])):
                                gtrans['W1'][i][j] += gW1[i][j]
                            gtrans['b1'][i] += gb1[i]
                        for i in range(len(gW2)):
                            for j in range(len(gW2[0])):
                                gtrans['W2'][i][j] += gW2[i][j]
                            gtrans['b2'][i] += gb2[i]
                        for i in range(dim):
                            gtrans['ln1_gamma'][i] += gln1_gamma[i]
                            gtrans['ln1_beta'][i] += gln1_beta[i]
                            gtrans['ln2_gamma'][i] += gln2_gamma[i]
                            gtrans['ln2_beta'][i] += gln2_beta[i]

                        # set dY for previous block
                        dY = d_inputs
                # end reversed blocks for this sequence
            # end per-sequence backprop accumulation

            # ---------------- Update parameters ----------------
            # Update descriptor parameters
            for idx in range(len(self.desc_layers)):
                desc = self.desc_layers[idx]
                for i in range(len(desc['P'])):
                    for j in range(len(desc['P'][i])):
                        for g in range(len(desc['P'][i][j])):
                            desc['P'][i][j][g] -= learning_rate * grad_desc_P[idx][i][j][g]
                for i in range(len(desc['M'])):
                    for j in range(len(desc['M'][0])):
                        desc['M'][i][j] -= learning_rate * grad_desc_M[idx][i][j]
                # update residual projection R
                for i in range(len(desc['R'])):
                    for j in range(len(desc['R'][0])):
                        desc['R'][i][j] -= learning_rate * grad_desc_R[idx][i][j]
                # update LayerNorm params for descriptor input
                for j in range(len(desc['ln_gamma_in'])):
                    desc['ln_gamma_in'][j] -= learning_rate * grad_desc_ln_gamma_in[idx][j]
                    desc['ln_beta_in'][j] -= learning_rate * grad_desc_ln_beta_in[idx][j]

            # Update transformer parameters
            for idx in range(len(self.trans_layers)):
                trans = self.trans_layers[idx]
                g = grad_trans[idx]
                for i in range(len(trans['Wq'])):
                    for j in range(len(trans['Wq'][0])):
                        trans['Wq'][i][j] -= learning_rate * g['Wq'][i][j]
                for i in range(len(trans['Wk'])):
                    for j in range(len(trans['Wk'][0])):
                        trans['Wk'][i][j] -= learning_rate * g['Wk'][i][j]
                for i in range(len(trans['Wv'])):
                    for j in range(len(trans['Wv'][0])):
                        trans['Wv'][i][j] -= learning_rate * g['Wv'][i][j]
                for i in range(len(trans['Wo'])):
                    for j in range(len(trans['Wo'][0])):
                        trans['Wo'][i][j] -= learning_rate * g['Wo'][i][j]
                for i in range(len(trans['b_o'])):
                    trans['b_o'][i] -= learning_rate * g['b_o'][i]
                for i in range(len(trans['W1'])):
                    for j in range(len(trans['W1'][0])):
                        trans['W1'][i][j] -= learning_rate * g['W1'][i][j]
                for i in range(len(trans['b1'])):
                    trans['b1'][i] -= learning_rate * g['b1'][i]
                for i in range(len(trans['W2'])):
                    for j in range(len(trans['W2'][0])):
                        trans['W2'][i][j] -= learning_rate * g['W2'][i][j]
                for i in range(len(trans['b2'])):
                    trans['b2'][i] -= learning_rate * g['b2'][i]
                # update layernorm parameters (pre-LN)
                for i in range(len(trans['ln1_gamma'])):
                    trans['ln1_gamma'][i] -= learning_rate * g['ln1_gamma'][i]
                    trans['ln1_beta'][i] -= learning_rate * g['ln1_beta'][i]
                    trans['ln2_gamma'][i] -= learning_rate * g['ln2_gamma'][i]
                    trans['ln2_beta'][i] -= learning_rate * g['ln2_beta'][i]

            # Update readout parameters
            for i in range(self.input_dim):
                for j in range(out_dim):
                    # guard: in case of unexpected shape differences
                    if j < len(self.read_W[i]):
                        self.read_W[i][j] -= learning_rate * grad_read_W[i][j]
                self.read_b[i] -= learning_rate * grad_read_b[i]

            # compute loss for reporting (decoded vs targets)
            total_loss = 0.0
            total_count = 0
            for seq in seqs:
                outs = self.describe(seq)
                L = len(seq)
                for k in range(L):
                    has_target = (mode == 'gap') or (mode == 'reg' and k < L-1)
                    if not has_target:
                        continue
                    if mode == 'gap':
                        target_vec = seq[k]
                    else:
                        target_vec = seq[k+1]
                    y = outs[k]
                    local_out = min(len(y), out_dim)
                    decoded = [0.0]*self.input_dim
                    for i in range(self.input_dim):
                        s = 0.0
                        row = self.read_W[i]
                        for j in range(local_out):
                            s += row[j] * y[j]
                        decoded[i] = s + self.read_b[i]
                    err = [decoded_i - t_i for decoded_i, t_i in zip(decoded, target_vec)]
                    total_loss += sum(e*e for e in err)
                    total_count += 1
            current_D = total_loss / total_count if total_count>0 else 0.0
            history.append(current_D)
            if it % print_every == 0 or it == max_iters-1:
                print(f"DDNet AUTO Iter {it:3d}: D = {current_D:.6e}, LR = {learning_rate:.6f}, mode={mode}")
            if current_D >= D_prev - tol:
                print(f"Auto-train converged after {it+1} iterations.")
                break
            D_prev = current_D
            learning_rate *= decay_rate

        self.trained = True
        return history

    def generate(self, L, tau=0.0):
        """
        Generate a sequence of vectors using the trained DDNet model.

        Parameters
        ----------
        L : int
            Number of vectors to generate.
        tau : float
            Temperature / noise scale. tau == 0 -> deterministic. tau > 0 -> add Gaussian noise N(0, tau^2) to each output dimension.

        Returns
        -------
        generated : list of list of float
            Generated sequence of length L, each item is an input-dim vector.
        """
        if L <= 0:
            return []

        # Warning if model not trained (optional)
        # (we do not enforce training, model can still generate with random weights)
        if not getattr(self, 'trained', False):
            # optional: print("Warning: model not marked as trained; outputs may be random.")
            pass

        # initial context: start from a zero vector (could be replaced by other seed strategy)
        context = [ [0.0] * self.input_dim ]

        generated = []
        for step in range(L):
            # forward the current context through the model
            outs = self.describe(context)
            if len(outs) == 0:
                # fallback: if describe returned empty, just produce noise or zeros
                y = [0.0] * (self.model_dims[-1] if hasattr(self, 'model_dims') else 1)
            else:
                y = outs[-1]

            # robust decode: decoded = read_W * y + read_b
            out_dim_y = len(y)
            decoded = []
            for i in range(self.input_dim):
                row = self.read_W[i] if i < len(self.read_W) else [0.0]*out_dim_y
                s = 0.0
                # multiply up to min length to avoid index error if shapes inconsistent
                use_len = min(len(row), out_dim_y)
                for j in range(use_len):
                    s += row[j] * y[j]
                # if row longer than y, remaining contributions ignored; if row shorter, ignored too
                s += self.read_b[i] if i < len(self.read_b) else 0.0
                decoded.append(s)

            # apply stochasticity: add Gaussian noise scaled by tau
            if tau and tau > 0.0:
                decoded = [ d + random.gauss(0.0, tau) for d in decoded ]

            # append to generated list and extend context for next step
            generated.append(decoded)
            context.append(decoded)

        return generated

    def show(self, what=None, first_num=5):
        print("DDNet Status")
        print("="*60)
        print(f"Input dim: {self.input_dim}")
        print(f"Descriptor layer dims: {self.model_dims}")
        print(f"Basis per descriptor: {self.num_basis_list}")
        print(f"Number of descriptor layers: {self.num_desc_layers}")
        print(f"Number of transformer layers: {len(self.trans_layers)}")
        print(f"Trained: {self.trained}")
        print("-"*60)
        if what is None or what=='desc':
            print("Descriptor layers sample (M shapes, R shapes and P shapes). Note: ln params are for input (in_dim).")
            for i, d in enumerate(self.desc_layers):
                M = d['M']; P = d['P']; R = d['R']
                print(f" Desc {i}: M {len(M)}x{len(M[0])}, R {len(R)}x{len(R[0])}, P {len(P)}x{len(P[0])}x{len(P[0][0])}")
                in_dim = len(R[0]) if len(R)>0 else 0
                print("  ln_input gamma sample:", [round(x,4) for x in d['ln_gamma_in'][:min(first_num,in_dim)]])
                print("  M sample row 0:", [round(x,4) for x in M[0][:first_num]])
        if what is None or what=='trans':
            print("\nTransformer layers sample (Wq shapes etc):")
            for i, t in enumerate(self.trans_layers):
                print(f" Trans {i}: dim {t['dim']}")
                print("  Wq sample row 0:", [round(x,4) for x in t['Wq'][0][:first_num]])
        print("="*60)

    def count_parameters(self):
        total = 0
        print("Parameter count for DDNet:")
        for i, d in enumerate(self.desc_layers):
            M = d['M']; P = d['P']; R = d['R']
            m_params = len(M)*len(M[0])
            r_params = len(R)*len(R[0])
            p_params = len(P)*len(P[0])*len(P[0][0])
            ln_in = len(d['ln_gamma_in'])
            print(f" Desc {i}: M {len(M)}x{len(M[0])} => {m_params}; R => {r_params}; P => {p_params}; ln_input_params => {2*ln_in}")
            total += m_params + p_params + r_params + 2*ln_in
        for i, t in enumerate(self.trans_layers):
            dim = t['dim']
            Wq = dim*dim; Wk = dim*dim; Wv = dim*dim; Wo = dim*dim
            b_o = dim
            W1 = len(t['W1']) * len(t['W1'][0])
            b1 = len(t['b1'])
            W2 = len(t['W2']) * len(t['W2'][0])
            b2 = len(t['b2'])
            # layernorm params (two pre-LNs)
            ln = 4 * dim
            t_total = Wq+Wk+Wv+Wo+b_o+W1+b1+W2+b2+ln
            print(f" Trans {i}: approx params = {t_total}")
            total += t_total
        # include readout params
        readout_params = self.input_dim * self.model_dims[-1] + self.input_dim
        print(f" Readout params (read_W and read_b): {readout_params}")
        total += readout_params
        print(f"Total parameters (approx): {total}")
        return total

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj

# ---------------------------
# --- Example usage & quick test ---
# ---------------------------
if __name__=="__main__":

    random.seed(12)
    input_dim = 10
    model_dims = [8,6,3]
    num_basis_list = [5,4,3]
    num_seqs = 10
    min_len, max_len = 100, 200

    seqs = []
    t_list = []
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        seq = [[random.uniform(-1,1) for _ in range(input_dim)] for __ in range(L)]
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(model_dims[-1])])

    dd = DDNet(input_dim=input_dim, model_dims=model_dims, num_basis_list=num_basis_list)
    dd.show()
    dd.count_parameters()

    print("\nTraining DDNet (few iters, may be slow because pure-Python)...")
    hist = dd.grad_train(seqs, t_list, max_iters=50, tol=1e-38, learning_rate=0.1, decay_rate=0.99, print_every=5)
    print("History:", hist)

    # correlation
    def pearson_corr(xs, ys):
        n = len(xs)
        if n==0: return 0.0
        mean_x = sum(xs)/n
        mean_y = sum(ys)/n
        num = sum((a-mean_x)*(b-mean_y) for a,b in zip(xs,ys))
        den_x = math.sqrt(sum((a-mean_x)**2 for a in xs))
        den_y = math.sqrt(sum((b-mean_y)**2 for b in ys))
        if den_x*den_y == 0:
            return 0.0
        return num/(den_x*den_y)

    preds = [dd.predict_t(s) for s in seqs]
    print("\nPer-output-dimension correlations:")
    corrs = []
    for dim in range(model_dims[-1]):
        actual = [t[dim] for t in t_list]
        predicted = [p[dim] for p in preds]
        c = pearson_corr(actual, predicted)
        corrs.append(c)
        print(f"  dim {dim}: corr = {c:.4f}")
    if corrs:
        print("Average correlation:", sum(corrs)/len(corrs))

        print("\nAuto-training (gap mode, few iters)...")
    # Example: use auto_train to predict input vectors (gap filling)
    hist2 = dd.auto_train(seqs[:3], mode='gap', max_iters=10, learning_rate=0.05, decay_rate=0.99, print_every=2)
    print("Auto-train history:", hist2)

    print("\nPredictions (first 3 sequences):")
    for i, seq in enumerate(seqs[:3]):
        pred = dd.predict_t(seq)
        print(f" Seq {i+1} target: {[f'{x:.4f}' for x in t_list[i]]}")
        print(f"            pred: {[f'{x:.4f}' for x in pred]}")
        
    dd.save("ddnet_model.pkl")
    loaded = DDNet.load("ddnet_model.pkl")
    print("Loaded model prediction on seq0:", [f"{x:.4f}" for x in loaded.predict_t(seqs[0])])

    # Example: training and generation with DDNet

    # 1. Create random training sequences
    input_dim = 5
    num_sequences = 3
    seq_length = 8
    train_data = []
    for _ in range(num_sequences):
        seq = []
        for _ in range(seq_length):
            vec = [random.uniform(-1, 1) for _ in range(input_dim)]
            seq.append(vec)
        train_data.append(seq)

    # 2. Initialize DDNet
    # model_dims: list of dimensions after each descriptor/transformer block
    # num_basis_list: number of basis functions per descriptor layer
    model_dims = [8, 8]
    num_basis_list = [3, 3]
    dd = DDNet(input_dim, model_dims, num_basis_list)

    # 3. Train using auto_train in regression mode
    print("Starting auto_train...")
    history = dd.auto_train(train_data, mode='reg', max_iters=20, learning_rate=0.05, decay_rate=0.99, print_every=5)
    print("Training history:", history)

    # 4. Generate sequence after training
    print("\nGenerating sequence...")
    generated_seq = dd.generate(L=10, tau=0.2)

    # 5. Print generated vectors
    print("Generated sequence (length = 10):")
    for idx, vec in enumerate(generated_seq):
        print(f"Step {idx+1}: {vec}")
