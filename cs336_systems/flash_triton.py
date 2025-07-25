'''
Triton implementation of FlashAttention-2 forward pass following Algorithm 1.
'''

import torch
import triton
import triton.language as tl
import math


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q_i from global memory
    Q_i = tl.load(Q_block_ptr)

    # Initialize O_i^(0) = 0, l_i^(0) = 0, m_i^(0) = -∞
    O_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    m_i = tl.full([Q_TILE_SIZE], -float("inf"), dtype=tl.float32)

    # Number of key tiles
    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    
    # For causal masking, we need query and key indices
    if is_causal:
        q_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE

    # Loop over key tiles j = 1, ..., T_k
    for j in range(num_key_tiles):
        # Load K^(j), V^(j) from global memory
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)
        
        # Compute tile of pre-softmax attention scores S_i^(j) = Q_i (K^(j))^T / √d
        S_ij = tl.dot(Q_i, K_j, allow_tf32=False) * scale
        
        # Apply causal masking if enabled
        if is_causal:
            k_indices = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            # Create mask: q_idx >= k_idx for causal attention
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            # Add large negative value to masked positions
            S_ij = tl.where(causal_mask, S_ij, -1e6)
        
        # Compute m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
        m_ij = tl.max(S_ij, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Compute P̃_i^(j) = exp(S_i^(j) - m_i^(j))
        P_ij = tl.exp(S_ij - m_i_new[:, None])
        
        # Compute l_i^(j) = exp(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(P̃_i^(j))
        alpha = tl.exp(m_i - m_i_new)
        l_ij = tl.sum(P_ij, axis=1)
        l_i_new = alpha * l_i + l_ij
        
        # Compute O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) O_i^(j-1) + P̃_i^(j) V^(j)
        # Cast P_ij to V_j dtype before multiplication
        P_ij = P_ij.to(V_j.dtype)
        O_i_new = alpha[:, None] * O_i + tl.dot(P_ij, V_j, allow_tf32=False)
        
        # Update state
        O_i = O_i_new
        l_i = l_i_new
        m_i = m_i_new
        
        # Advance block pointers for next iteration
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # Compute final O_i = diag(l_i^(T_k))^(-1) O_i^(T_k)
    O_i = O_i / l_i[:, None]
    
    # Compute L_i = m_i^(T_k) + log(l_i^(T_k))
    L_i = m_i + tl.log(l_i)
    
    # Cast O_i to appropriate dtype before writing to global memory
    O_i = O_i.to(O_block_ptr.type.element_ty)
    
    # Write O_i to global memory as the i-th tile of O
    tl.store(O_block_ptr, O_i)
    
    # Write L_i to global memory as the i-th tile of L
    tl.store(L_block_ptr, L_i)


@triton.jit
def flash_bwd_preprocess_kernel(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    N_QUERIES,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    """
    Compute D = rowsum(dO * O) for backward pass.
    """
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Create block pointers
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    # Load O and dO
    O_i = tl.load(O_block_ptr)
    dO_i = tl.load(dO_block_ptr)
    
    # Compute D = rowsum(dO * O)
    D_i = tl.sum(dO_i * O_i, axis=1)
    
    # Store D
    tl.store(D_block_ptr, D_i)


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, D_ptr,
    dO_ptr, dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Compute dQ gradient for FlashAttention-2 backward pass.
    """
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Set up block pointers for this query tile
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    # Set up K, V block pointers
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    # Load query tile data
    Q_i = tl.load(Q_block_ptr)
    dO_i = tl.load(dO_block_ptr)
    L_i = tl.load(L_block_ptr)
    D_i = tl.load(D_block_ptr)
    
    # Initialize dQ for this query tile
    dQ_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    
    # For causal masking
    if is_causal:
        q_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
    
    # Loop over key tiles
    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(num_key_tiles):
        # Load K and V tiles
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)
        
        # Recompute attention scores S_ij
        S_ij = tl.dot(Q_i, K_j, allow_tf32=False) * scale
        
        # Apply causal masking if enabled
        if is_causal:
            k_indices = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            S_ij = tl.where(causal_mask, S_ij, -1e6)
        
        # Recompute attention probabilities P_ij = exp(S_ij - L_i)
        P_ij = tl.exp(S_ij - L_i[:, None])
        
        # Compute dP_ij = dO_i @ V_j^T
        dP_ij = tl.dot(dO_i, tl.trans(V_j), allow_tf32=False)
        
        # Compute dS_ij = P_ij * (dP_ij - D_i)
        dS_ij = P_ij * (dP_ij - D_i[:, None])
        
        # Compute dQ contribution: dQ_ij = dS_ij @ K_j^T * scale
        dQ_ij = tl.dot(dS_ij, tl.trans(K_j), allow_tf32=False) * scale
        dQ_i += dQ_ij
        
        # Advance block pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    # Store dQ for this query tile
    dQ_i = dQ_i.to(dQ_block_ptr.type.element_ty)
    tl.store(dQ_block_ptr, dQ_i)


@triton.jit
def flash_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, D_ptr,
    dO_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dob, stride_doq, stride_dod,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Compute dK and dV gradients for FlashAttention-2 backward pass.
    """
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Set up block pointers for this key tile
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    # Set up Q, O, dO, L, D block pointers 
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    # Load key/value tile data
    K_j = tl.load(K_block_ptr)
    V_j = tl.load(V_block_ptr)
    
    # Initialize gradients for this key tile
    dK_j = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)
    dV_j = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)
    
    # For causal masking
    if is_causal:
        k_indices = tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE
    
    # Loop over query tiles
    num_query_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    for i in range(num_query_tiles):
        # Load Q, dO, L, D tiles
        Q_i = tl.load(Q_block_ptr)
        dO_i = tl.load(dO_block_ptr)
        L_i = tl.load(L_block_ptr)
        D_i = tl.load(D_block_ptr)
        
        # Recompute attention scores S_ij
        S_ij = tl.dot(Q_i, tl.trans(K_j), allow_tf32=False) * scale
        
        # Apply causal masking if enabled
        if is_causal:
            q_indices = tl.arange(0, Q_TILE_SIZE) + i * Q_TILE_SIZE
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            S_ij = tl.where(causal_mask, S_ij, -1e6)
        
        # Recompute attention probabilities P_ij = exp(S_ij - L_i)
        P_ij = tl.exp(S_ij - L_i[:, None])
        
        # Compute dV contribution: dV_j += P_ij^T @ dO_i
        dV_ij = tl.dot(tl.trans(P_ij), dO_i, allow_tf32=False)
        dV_j += dV_ij
        
        # Compute dP_ij = dO_i @ V_j^T
        dP_ij = tl.dot(dO_i, tl.trans(V_j), allow_tf32=False)
        
        # Compute dS_ij = P_ij * (dP_ij - D_i)
        dS_ij = P_ij * (dP_ij - D_i[:, None])
        
        # Compute dK contribution: dK_j += dS_ij^T @ Q_i * scale
        dK_ij = tl.dot(tl.trans(dS_ij), Q_i, allow_tf32=False) * scale
        dK_j += dK_ij
        
        # Advance block pointers for query tiles
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))
    
    # Store gradients for this key tile
    dK_j = dK_j.to(dK_block_ptr.type.element_ty)
    dV_j = dV_j.to(dV_block_ptr.type.element_ty)
    tl.store(dK_block_ptr, dK_j)
    tl.store(dV_block_ptr, dV_j)


class FlashAttentionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention-2 forward pass implemented using Triton kernels.
        
        Args:
            ctx: PyTorch autograd context
            Q: Query tensor of shape [batch_size, n_queries, d_head]
            K: Key tensor of shape [batch_size, n_keys, d_head] 
            V: Value tensor of shape [batch_size, n_keys, d_head]
            is_causal: Whether to apply causal masking
            
        Returns:
            O: Output tensor of shape [batch_size, n_queries, d_head]
        """
        # Ensure inputs are contiguous and on CUDA
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        
        batch_size, n_queries, d_head = Q.shape
        _, n_keys, _ = K.shape
        
        # Choose tile sizes
        Q_TILE_SIZE = min(64, n_queries)
        K_TILE_SIZE = min(64, n_keys)
        
        # Ensure tile sizes are at least 16 and powers of 2 for better performance
        Q_TILE_SIZE = max(16, 1 << (Q_TILE_SIZE - 1).bit_length())
        K_TILE_SIZE = max(16, 1 << (K_TILE_SIZE - 1).bit_length())
        
        scale = 1.0 / math.sqrt(d_head)
        
        # Initialize output tensors
        O = torch.empty_like(Q)
        L = torch.empty((batch_size, n_queries), device=Q.device, dtype=Q.dtype)
        
        # Calculate grid size: (num_query_tiles, batch_size)
        num_query_tiles = (n_queries + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        grid = (num_query_tiles, batch_size)
        
        # Get strides
        stride_qb, stride_qq, stride_qd = Q.stride()
        stride_kb, stride_kk, stride_kd = K.stride()
        stride_vb, stride_vk, stride_vd = V.stride()
        stride_ob, stride_oq, stride_od = O.stride()
        stride_lb, stride_lq = L.stride()
        
        # Launch Triton kernel
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            n_queries, n_keys,
            scale,
            D=d_head,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        
        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        FlashAttention-2 backward pass implemented using Triton kernels.
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        dO = grad_output.contiguous()
        
        batch_size, n_queries, d_head = Q.shape
        _, n_keys, _ = K.shape
        
        # Choose same tile sizes as forward pass
        Q_TILE_SIZE = min(64, n_queries)
        K_TILE_SIZE = min(64, n_keys)
        Q_TILE_SIZE = max(16, 1 << (Q_TILE_SIZE - 1).bit_length())
        K_TILE_SIZE = max(16, 1 << (K_TILE_SIZE - 1).bit_length())
        
        scale = 1.0 / math.sqrt(d_head)
        
        # Initialize gradient tensors
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        # Create D tensor for preprocessing
        D = torch.empty((batch_size, n_queries), device=Q.device, dtype=Q.dtype)
        
        # Step 1: Compute D = rowsum(dO * O) using preprocessing kernel
        num_query_tiles = (n_queries + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        grid_preprocess = (num_query_tiles, batch_size)
        
        # Get strides for preprocessing kernel
        stride_ob, stride_oq, stride_od = O.stride()
        stride_dob, stride_doq, stride_dod = dO.stride()
        stride_db, stride_dq = D.stride()
        
        # grid_preprocess = (num_query_tiles, batch_size) 定义了kernel的并行执行网格
        # 每个tile对应一个线程块,总共有num_query_tiles * batch_size个线程块并行执行
        flash_bwd_preprocess_kernel[grid_preprocess](
            O, dO, D,
            stride_ob, stride_oq, stride_od,
            stride_dob, stride_doq, stride_dod,
            stride_db, stride_dq,
            n_queries,
            D=d_head,
            Q_TILE_SIZE=Q_TILE_SIZE,
        )
        
        # Step 2: Compute dQ gradients
        grid_dq = (num_query_tiles, batch_size)
        
        # Get all strides
        stride_qb, stride_qq, stride_qd = Q.stride()
        stride_kb, stride_kk, stride_kd = K.stride()
        stride_vb, stride_vk, stride_vd = V.stride()
        stride_lb, stride_lq = L.stride()
        stride_dqb, stride_dqq, stride_dqd = dQ.stride()
        stride_dkb, stride_dkk, stride_dkd = dK.stride()
        stride_dvb, stride_dvk, stride_dvd = dV.stride()
        
        flash_bwd_dq_kernel[grid_dq](
            Q, K, V, O, L, D,
            dO, dQ,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            stride_db, stride_dq,
            stride_dob, stride_doq, stride_dod,
            stride_dqb, stride_dqq, stride_dqd,
            n_queries, n_keys,
            scale,
            D=d_head,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        
        # Step 3: Compute dK and dV gradients
        num_key_tiles = (n_keys + K_TILE_SIZE - 1) // K_TILE_SIZE
        grid_dkv = (num_key_tiles, batch_size)
        
        flash_bwd_dkv_kernel[grid_dkv](
            Q, K, V, O, L, D,
            dO, dK, dV,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            stride_db, stride_dq,
            stride_dob, stride_doq, stride_dod,
            stride_dkb, stride_dkk, stride_dkd,
            stride_dvb, stride_dvk, stride_dvd,
            n_queries, n_keys,
            scale,
            D=d_head,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        
        return dQ, dK, dV, None
