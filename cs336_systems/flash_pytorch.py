'''
Pure PyTorch (no Triton) autograd.Function that implements the FlashAttention-2 forward pass.
'''

import torch
import math


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention-2 forward pass implemented in pure PyTorch.
        
        Args:
            ctx: PyTorch autograd context
            Q: Query tensor of shape [batch_size, n_queries, d_head]
            K: Key tensor of shape [batch_size, n_keys, d_head] 
            V: Value tensor of shape [batch_size, n_keys, d_head]
            is_causal: Whether to apply causal masking (ignored for now)
            
        Returns:
            O: Output tensor of shape [batch_size, n_queries, d_head]
        """
        batch_size, n_queries, d_head = Q.shape
        _, n_keys, _ = K.shape
        
        # Choose tile sizes - making them at least 16x16 as required
        Br = min(32, n_queries)  # Query block size 
        Bc = min(32, n_keys)     # Key block size
        
        scale = 1.0 / math.sqrt(d_head)
        
        # Initialize output and logsumexp
        O = torch.zeros_like(Q)
        L = torch.full((batch_size, n_queries), -torch.inf, device=Q.device, dtype=Q.dtype)
        
        # Process in tiles
        num_query_blocks = math.ceil(n_queries / Br)
        num_key_blocks = math.ceil(n_keys / Bc)
        
        for i in range(num_query_blocks):
            # Query block indices
            q_start = i * Br
            q_end = min((i + 1) * Br, n_queries)
            
            # Extract query block, load into cache
            Q_i = Q[:, q_start:q_end, :]  # [batch_size, Br, d_head]
            
            # Initialize block output and statistics
            O_i = torch.zeros_like(Q_i)
            m_i = torch.full((batch_size, q_end - q_start), -torch.inf, device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros((batch_size, q_end - q_start), device=Q.device, dtype=Q.dtype)
            
            for j in range(num_key_blocks):
                # Key/Value block indices  
                k_start = j * Bc
                k_end = min((j + 1) * Bc, n_keys)
                
                # Extract key and value blocks
                K_j = K[:, k_start:k_end, :]  # [batch_size, Bc, d_head]
                V_j = V[:, k_start:k_end, :]  # [batch_size, Bc, d_head]
                
                # Compute attention scores for this block
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale  # [batch_size, Br, Bc]
                
                # Compute max for this block
                m_ij = torch.max(S_ij, dim=-1)[0]  # [batch_size, Br]
                
                # Update global max
                m_i_new = torch.maximum(m_i, m_ij)
                
                # Correct scaling factors for numerical stability
                alpha = torch.exp(m_i - m_i_new)  # Scale factor for previous sum
                beta = torch.exp(m_ij - m_i_new)  # Scale factor for current block
                
                # Compute softmax probabilities for current block
                P_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))  # [batch_size, Br, Bc]
                
                # Sum of current block probabilities  
                l_ij = torch.sum(P_ij, dim=-1)  # [batch_size, Br]
                
                # Update normalizer
                l_i_new = alpha * l_i + l_ij
                
                # Update output: properly weight old and new contributions
                O_i_new = (alpha.unsqueeze(-1) * O_i * l_i.unsqueeze(-1) + torch.matmul(P_ij, V_j)) / l_i_new.unsqueeze(-1)
                
                # Update state
                O_i = O_i_new
                m_i = m_i_new  
                l_i = l_i_new
            
            # Store block results
            O[:, q_start:q_end, :] = O_i
            L[:, q_start:q_end] = m_i + torch.log(l_i)  # logsumexp = max + log(sum_exp)
        
        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, V, O, L)
        
        return O
    
    @staticmethod
    @torch.compile
    def backward(ctx, grad_output):
        """
        FlashAttention-2 backward pass implemented in pure PyTorch.
        
        Args:
            ctx: PyTorch autograd context
            grad_output: Gradient w.r.t. output O, shape [batch_size, n_queries, d_head]
            
        Returns:
            Tuple of gradients: (dQ, dK, dV, None) where None is for is_causal parameter
        """
        # Recover saved tensors
        Q, K, V, O, L = ctx.saved_tensors
        dO = grad_output
        
        batch_size, n_queries, d_head = Q.shape
        _, n_keys, _ = K.shape
        
        scale = 1.0 / math.sqrt(d_head)
        
        # Compute D vector: D_i = sum(dO_i * O_i, dim=-1)
        D = torch.sum(dO * O, dim=-1)  # [batch_size, n_queries]
        
        # Initialize gradient tensors
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        # Use same tile sizes as forward pass
        Br = min(32, n_queries)  # Query block size
        Bc = min(32, n_keys)     # Key block size
        
        num_query_blocks = math.ceil(n_queries / Br)
        num_key_blocks = math.ceil(n_keys / Bc)
        
        for i in range(num_query_blocks):
            # Query block indices
            q_start = i * Br
            q_end = min((i + 1) * Br, n_queries)
            
            # Extract query block and related tensors
            Q_i = Q[:, q_start:q_end, :]      # [batch_size, Br, d_head]
            O_i = O[:, q_start:q_end, :]      # [batch_size, Br, d_head]
            dO_i = dO[:, q_start:q_end, :]    # [batch_size, Br, d_head]
            L_i = L[:, q_start:q_end]         # [batch_size, Br]
            D_i = D[:, q_start:q_end]         # [batch_size, Br]
            
            # Initialize gradients for this query block
            dQ_i = torch.zeros_like(Q_i)
            
            for j in range(num_key_blocks):
                # Key/Value block indices
                k_start = j * Bc
                k_end = min((j + 1) * Bc, n_keys)
                
                # Extract key and value blocks
                K_j = K[:, k_start:k_end, :]  # [batch_size, Bc, d_head]
                V_j = V[:, k_start:k_end, :]  # [batch_size, Bc, d_head]
                
                # Recompute attention scores for this block (Equation 13)
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale  # [batch_size, Br, Bc]
                
                # Recompute attention probabilities (Equation 14)
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))  # [batch_size, Br, Bc]
                
                # Compute dV for this block (Equation 15): dV = P⊤dO
                dV_j = torch.matmul(P_ij.transpose(-2, -1), dO_i)  # [batch_size, Bc, d_head]
                dV[:, k_start:k_end, :] += dV_j
                
                # Compute dP (Equation 16): dP = dOV⊤
                dP_ij = torch.matmul(dO_i, V_j.transpose(-2, -1))  # [batch_size, Br, Bc]
                
                # Compute dS (Equation 17): dS_ij = P_ij ◦ (dP_ij - D_i)
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))  # [batch_size, Br, Bc]
                
                # Compute dQ for this block (Equation 18): dQ = dS K / √d
                dQ_ij = torch.matmul(dS_ij, K_j) * scale  # [batch_size, Br, d_head]
                dQ_i += dQ_ij
                
                # Compute dK for this block (Equation 19): dK = dS⊤ Q / √d
                dK_j = torch.matmul(dS_ij.transpose(-2, -1), Q_i) * scale  # [batch_size, Bc, d_head]
                dK[:, k_start:k_end, :] += dK_j
            
            # Store query gradients
            dQ[:, q_start:q_end, :] = dQ_i
        
        return dQ, dK, dV, None
