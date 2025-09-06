import einops
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce, repeat


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        # Precompute frequencies
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2).float() / d_k))
        t = torch.arange(max_seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_seq_len, d_k/2]
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        # x: [*, seq_len, d_k]
        # token_positions: [*, seq_len]
        # Get cos/sin for each token position
        if token_positions:
            cos = self.cos_cached[token_positions]  # [..., seq_len, d_k//2]
            sin = self.sin_cached[token_positions]  # [..., seq_len, d_k//2]
        else:
            cos = self.cos_cached[None, : x.shape[-2], :]  # [..., seq_len, d_k//2]
            sin = self.sin_cached[None, : x.shape[-2], :]  # [..., seq_len, d_k//2]
        # Split x into even and odd dims
        x_even = x[..., ::2]  # [..., seq_len, d_k//2]
        x_odd = x[..., 1::2]  # [..., seq_len, d_k//2]
        # Apply rotation
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        # Interleave even/odd back to original shape
        x_rot = torch.stack(
            (x_rot_even, x_rot_odd), dim=-1
        )  # [..., seq_len, d_k//2, 2]
        x_rot = x_rot.flatten(-2)  # [..., seq_len, d_k]
        return x_rot


class RotaryPositionalEmbeddingMine(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k

        # self.position_to_rotary_matrix = []
        # for position in range(max_seq_len):
        #     self.position_to_rotary_matrix.append(
        #         self._create_rotary_matrix_for_position_i(position)
        #     )
        # self.position_to_rotary_matrix = torch.stack(
        #     self.position_to_rotary_matrix, dim=0
        # )  # [max_seq_len, d_k / 2, 2, 2]

    def _create_rotary_matrix_for_position_i(self, position: int) -> torch.Tensor:
        """
        2 x d_k matrix for a given position
        """
        R_i_list: list[torch.Tensor] = []
        for k in range(1, int(self.d_k / 2) + 1):
            exponent = self.theta ** ((2 * k - 1) / self.d_k)
            theta_i_k = position / exponent
            R_i_k = torch.tensor(
                [
                    torch.cos(torch.tensor(theta_i_k)),
                    -torch.sin(torch.tensor(theta_i_k)),
                    torch.sin(torch.tensor(theta_i_k)),
                    torch.cos(torch.tensor(theta_i_k)),
                ]
            ).view(2, 2)
            R_i_list.append(R_i_k)
        return torch.stack(R_i_list, dim=0)  # [d_k / 2, 2, 2]

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension.
        """
        seq_len = x.shape[-2]
        position_to_rotary_matrix = []
        for position in range(seq_len):
            position_to_rotary_matrix.append(
                self._create_rotary_matrix_for_position_i(position)
            )
        position_to_rotary_matrix = torch.stack(
            position_to_rotary_matrix, dim=0
        )  # [max_seq_len, d_k / 2, 2, 2]
        one_hot = F.one_hot(token_positions, num_classes=seq_len)  # [..., s, position]
        # print(position_to_rotary_matrix.shape)
        # print(one_hot.shape)
        # choose the cos sin tensor to use for each position
        cos_sin = einsum(
            one_hot.float(),
            position_to_rotary_matrix,
            "... s position, position d_k_2 two two_again -> ... s d_k_2 two two_again",
        )
        x_reshaped = rearrange(
            x, "... s (d_k_2 two one) -> ... s d_k_2 two one", two=2, one=1
        )  # [..., s, d_k/2, 2 1]

        x_rotated = einsum(
            cos_sin,
            x_reshaped,
            "... s d_k_2 t_1 two, ... s d_k_2 two one -> ... s d_k_2 t_1 one",
        )

        x_rotated = rearrange(
            x_rotated, "... s d_k_2 two one -> ... s (d_k_2 two one)", two=2, one=1
        )
        return x_rotated
