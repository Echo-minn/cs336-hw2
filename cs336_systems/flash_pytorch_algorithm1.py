"""
Pure PyTorch implementation of FlashAttention-2 Algorithm 1 forward pass.
"""

import torch
import math


@torch.compile
def _flash_attention_forward_compiled(Q, K, V, B_q, B_k, scale):
    """Compiled version of FlashAttention forward pass core computation."""
    batch_size, N_q, d = Q.shape
    _, N_k, _ = K.shape
    
    # Number of tiles
    T_q = math.ceil(N_q / B_q)
    T_k = math.ceil(N_k / B_k)
    
    # Initialize output O and logsumexp L
    O = torch.zeros_like(Q)
    L = torch.zeros((batch_size, N_q), device=Q.device, dtype=Q.dtype)
    
    # Split Q into T_q tiles Q_1, ..., Q_{T_q} of size B_q × d
    for i in range(T_q):
        # Load Q_i from global memory
        q_start = i * B_q
        q_end = min((i + 1) * B_q, N_q)
        Q_i = Q[:, q_start:q_end, :]  # [batch_size, B_q, d]
        
        # Initialize O_i^(0) = 0, l_i^(0) = 0, m_i^(0) = -∞
        O_i = torch.zeros_like(Q_i)  # [batch_size, B_q, d]
        l_i = torch.zeros((batch_size, q_end - q_start), device=Q.device, dtype=Q.dtype)  # [batch_size, B_q]
        m_i = torch.full((batch_size, q_end - q_start), -torch.inf, device=Q.device, dtype=Q.dtype)  # [batch_size, B_q]
        
        # Split K, V into T_k tiles K^(1), ..., K^(T_k) and V^(1), ..., V^(T_k) of size B_k × d
        for j in range(T_k):
            # Load K^(j), V^(j) from global memory
            k_start = j * B_k
            k_end = min((j + 1) * B_k, N_k)
            K_j = K[:, k_start:k_end, :]  # [batch_size, B_k, d]
            V_j = V[:, k_start:k_end, :]  # [batch_size, B_k, d]
            
            # Compute tile of pre-softmax attention scores S_i^(j) = Q_i K^(j)^T / √d
            S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale  # [batch_size, B_q, B_k]
            
            # Compute m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
            m_ij = torch.max(S_ij, dim=-1)[0]  # [batch_size, B_q]
            m_i_new = torch.maximum(m_i, m_ij)  # [batch_size, B_q]
            
            # Compute P̃_i^(j) = exp(S_i^(j) - m_i^(j))
            # unsqueeze(-1)在最后一个维度增加一个维度
            # m_i_new shape: [batch_size, B_q] -> [batch_size, B_q, 1]
            # 这样可以进行广播操作,使m_i_new的shape与S_ij的shape匹配
            P_tilde_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))  # [batch_size, B_q, B_k]
            
            # Compute l_i^(j) = exp(m_i^(j-1) - m_i^(j)) l_i^(j-1) + rowsum(P̃_i^(j))
            exp_diff = torch.exp(m_i - m_i_new)  # [batch_size, B_q]
            rowsum_P_tilde = torch.sum(P_tilde_ij, dim=-1)  # [batch_size, B_q]
            l_i_new = exp_diff * l_i + rowsum_P_tilde  # [batch_size, B_q]
            
            # Compute O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) O_i^(j-1) + P̃_i^(j) V^(j)
            diag_term = exp_diff.unsqueeze(-1)  # [batch_size, B_q, 1]
            O_i_new = diag_term * O_i + torch.matmul(P_tilde_ij, V_j)  # [batch_size, B_q, d]
            
            # Update for next iteration
            O_i = O_i_new
            l_i = l_i_new
            m_i = m_i_new
        
        # Compute O_i = diag(l_i^(T_k))^(-1) O_i^(T_k)
        O_i = O_i / l_i.unsqueeze(-1)  # [batch_size, B_q, d]
        
        # Compute L_i = m_i^(T_k) + log(l_i^(T_k))
        L_i = m_i + torch.log(l_i)  # [batch_size, B_q]
        
        # Write O_i to global memory as the i-th tile of O
        O[:, q_start:q_end, :] = O_i
        
        # Write L_i to global memory as the i-th tile of L
        L[:, q_start:q_end] = L_i
    
    return O, L


@torch.compile
def _flash_attention_backward_compiled(Q, K, V, O, L, dO, B_q, B_k, scale):
    """Compiled version of FlashAttention backward pass core computation."""
    batch_size, N_q, d = Q.shape
    _, N_k, _ = K.shape
    
    # Number of tiles
    T_q = math.ceil(N_q / B_q)
    T_k = math.ceil(N_k / B_k)
    
    # Compute D = rowsum(dO ∘ O) where ∘ is element-wise multiplication
    D = torch.sum(dO * O, dim=-1)  # [batch_size, N_q]
    
    # Initialize gradients
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    
    # Split K, V into T_k tiles K^(1), ..., K^(T_k) and V^(1), ..., V^(T_k) of size B_k × d
    for j in range(T_k):
        # Load K^(j), V^(j) from global memory
        k_start = j * B_k
        k_end = min((j + 1) * B_k, N_k)
        K_j = K[:, k_start:k_end, :]  # [batch_size, B_k, d]
        V_j = V[:, k_start:k_end, :]  # [batch_size, B_k, d]
        
        # Initialize dK^(j) = dV^(j) = 0
        dK_j = torch.zeros_like(K_j)  # [batch_size, B_k, d]
        dV_j = torch.zeros_like(V_j)  # [batch_size, B_k, d]
        
        # Split Q, O, dO into T_q tiles
        for i in range(T_q):
            # Load Q_i, O_i, dO_i, dQ_i from global memory
            q_start = i * B_q
            q_end = min((i + 1) * B_q, N_q)
            Q_i = Q[:, q_start:q_end, :]      # [batch_size, B_q, d]
            O_i = O[:, q_start:q_end, :]      # [batch_size, B_q, d]
            dO_i = dO[:, q_start:q_end, :]    # [batch_size, B_q, d]
            L_i = L[:, q_start:q_end]         # [batch_size, B_q]
            D_i = D[:, q_start:q_end]         # [batch_size, B_q]
            
            # Compute tile of attention scores S_i^(j) = Q_i (K^(j))^T / √d
            S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale  # [batch_size, B_q, B_k]
            
            # Compute attention probabilities P_i^(j) = exp(S_i^(j) - L_i)
            P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))  # [batch_size, B_q, B_k]
            
            # Compute dV^(j) += (P_i^(j))^T dO_i
            dV_j += torch.matmul(P_ij.transpose(-2, -1), dO_i)  # [batch_size, B_k, d]
            
            # Compute dP_i^(j) = dO_i V_j^T
            dP_ij = torch.matmul(dO_i, V_j.transpose(-2, -1))  # [batch_size, B_q, B_k]
            
            # Compute dS_i^(j) = P_i^(j) ∘ (dP_i^(j) - D_i) / √d
            dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1)) * scale  # [batch_size, B_q, B_k]
            
            # Load dQ_i from global memory, then update dQ_i += dS_i^(j) K^(j) and write back to global memory
            # Must be atomic for correctness!
            dQ_i_update = torch.matmul(dS_ij, K_j)  # [batch_size, B_q, d]
            dQ[:, q_start:q_end, :] += dQ_i_update
            
            # Compute dK^(j) += (dS_i^(j))^T Q_i
            dK_j += torch.matmul(dS_ij.transpose(-2, -1), Q_i)  # [batch_size, B_k, d]
        
        # Write dK^(j) and dV^(j) to global memory as the j-th tiles of dK and dV
        dK[:, k_start:k_end, :] = dK_j
        dV[:, k_start:k_end, :] = dV_j
    
    return dQ, dK, dV


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention-2 forward pass implementing Algorithm 1.
        
        Args:
            ctx: PyTorch autograd context
            Q: Query tensor of shape [batch_size, N_q, d]
            K: Key tensor of shape [batch_size, N_k, d] 
            V: Value tensor of shape [batch_size, N_k, d]
            is_causal: Whether to apply causal masking (ignored for now)
            
        Returns:
            O: Output tensor of shape [batch_size, N_q, d]
        """
        batch_size, N_q, d = Q.shape
        _, N_k, _ = K.shape
        
        # Choose tile sizes - at least 16x16 as required
        B_q = min(32, N_q)  # Query tile size
        B_k = min(32, N_k)  # Key tile size
        
        # Scale factor
        scale = 1.0 / math.sqrt(d)
        
        # Use compiled forward function
        O, L = _flash_attention_forward_compiled(Q, K, V, B_q, B_k, scale)
        
        # Save tensors for backward pass
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.B_q = B_q
        ctx.B_k = B_k
        ctx.scale = scale
        
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        FlashAttention-2 backward pass implementing Algorithm 2.
        
        Args:
            ctx: PyTorch autograd context
            grad_output (dO): Gradient w.r.t. output of shape [batch_size, N_q, d]
            
        Returns:
            Tuple of gradients: (dQ, dK, dV, None) for (Q, K, V, is_causal)
        """
        # Retrieve saved tensors from forward pass
        L, Q, K, V, O = ctx.saved_tensors
        dO = grad_output
        
        # Retrieve saved parameters from forward pass
        B_q = ctx.B_q
        B_k = ctx.B_k
        scale = ctx.scale
        
        # Use compiled backward function
        dQ, dK, dV = _flash_attention_backward_compiled(Q, K, V, O, L, dO, B_q, B_k, scale)
        
        # Return dQ, dK, dV, None (for is_causal parameter which doesn't need gradients)
        return dQ, dK, dV, None
