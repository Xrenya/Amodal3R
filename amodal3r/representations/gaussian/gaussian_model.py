import torch
import numpy as np
from plyfile import PlyData, PlyElement
from .general_utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation


class Gaussian:
    def __init__(
            self, 
            aabb : list,
            sh_degree : int = 0,
            mininum_kernel_size : float = 0.0,
            scaling_bias : float = 0.01,
            opacity_bias : float = 0.1,
            scaling_activation : str = "exp",
            device='cuda'
        ):
        self.init_params = {
            'aabb': aabb,
            'sh_degree': sh_degree,
            'mininum_kernel_size': mininum_kernel_size,
            'scaling_bias': scaling_bias,
            'opacity_bias': opacity_bias,
            'scaling_activation': scaling_activation,
        }
        
        self.sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size 
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
        self.scale_bias = self.inverse_scaling_activation(torch.tensor(self.scaling_bias)).to(self.device)
        self.rots_bias = torch.zeros((4), device=self.device)
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(torch.tensor(self.opacity_bias)).to(self.device)

    # ==================== PROPERTIES ====================
    
    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size ** 2
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        """Returns normalized quaternions in (w, x, y, z) format."""
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])
    
    @property
    def get_rotation_xyzw(self):
        """Returns normalized quaternions in (x, y, z, w) format."""
        quat_wxyz = self.get_rotation
        return torch.cat([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], dim=-1)
    
    @property
    def get_xyz(self):
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]
    
    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=2) if self._features_rest is not None else self._features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity + self.opacity_bias)
    
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :])

    # ==================== QUATERNION/MATRIX CONVERSIONS (NO EXTERNAL DEPENDENCY) ====================
    
    @staticmethod
    def _quaternion_wxyz_to_matrix_np(quaternions):
        """
        Convert quaternions (w, x, y, z) to rotation matrices.
        Works with numpy arrays.
        
        Args:
            quaternions: array of shape (N, 4) in (w, x, y, z) format
        
        Returns:
            matrices: array of shape (N, 3, 3)
        """
        # Normalize
        quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
        
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        
        N = quaternions.shape[0]
        R = np.zeros((N, 3, 3), dtype=quaternions.dtype)
        
        R[:, 0, 0] = 1 - 2*y*y - 2*z*z
        R[:, 0, 1] = 2*x*y - 2*w*z
        R[:, 0, 2] = 2*x*z + 2*w*y
        R[:, 1, 0] = 2*x*y + 2*w*z
        R[:, 1, 1] = 1 - 2*x*x - 2*z*z
        R[:, 1, 2] = 2*y*z - 2*w*x
        R[:, 2, 0] = 2*x*z - 2*w*y
        R[:, 2, 1] = 2*y*z + 2*w*x
        R[:, 2, 2] = 1 - 2*x*x - 2*y*y
        
        return R
    
    @staticmethod
    def _matrix_to_quaternion_wxyz_np(matrices):
        """
        Convert rotation matrices to quaternions (w, x, y, z).
        Works with numpy arrays.
        
        Args:
            matrices: array of shape (N, 3, 3)
        
        Returns:
            quaternions: array of shape (N, 4) in (w, x, y, z) format
        """
        N = matrices.shape[0]
        m = matrices
        trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
        
        quaternions = np.zeros((N, 4), dtype=matrices.dtype)
        
        # Case 1: trace > 0
        mask1 = trace > 0
        if np.any(mask1):
            s = np.sqrt(trace[mask1] + 1.0) * 2
            quaternions[mask1, 0] = 0.25 * s
            quaternions[mask1, 1] = (m[mask1, 2, 1] - m[mask1, 1, 2]) / s
            quaternions[mask1, 2] = (m[mask1, 0, 2] - m[mask1, 2, 0]) / s
            quaternions[mask1, 3] = (m[mask1, 1, 0] - m[mask1, 0, 1]) / s
        
        # Case 2: m[0,0] is largest diagonal
        mask2 = (~mask1) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
        if np.any(mask2):
            s = np.sqrt(1.0 + m[mask2, 0, 0] - m[mask2, 1, 1] - m[mask2, 2, 2]) * 2
            quaternions[mask2, 0] = (m[mask2, 2, 1] - m[mask2, 1, 2]) / s
            quaternions[mask2, 1] = 0.25 * s
            quaternions[mask2, 2] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s
            quaternions[mask2, 3] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s
        
        # Case 3: m[1,1] is largest diagonal
        mask3 = (~mask1) & (~mask2) & (m[:, 1, 1] > m[:, 2, 2])
        if np.any(mask3):
            s = np.sqrt(1.0 + m[mask3, 1, 1] - m[mask3, 0, 0] - m[mask3, 2, 2]) * 2
            quaternions[mask3, 0] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s
            quaternions[mask3, 1] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s
            quaternions[mask3, 2] = 0.25 * s
            quaternions[mask3, 3] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s
        
        # Case 4: m[2,2] is largest diagonal
        mask4 = (~mask1) & (~mask2) & (~mask3)
        if np.any(mask4):
            s = np.sqrt(1.0 + m[mask4, 2, 2] - m[mask4, 0, 0] - m[mask4, 1, 1]) * 2
            quaternions[mask4, 0] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s
            quaternions[mask4, 1] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s
            quaternions[mask4, 2] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s
            quaternions[mask4, 3] = 0.25 * s
        
        # Normalize
        quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
        
        return quaternions

    # ==================== RODRIGUES CONVERSION ====================
    
    @property
    def get_rodrigues(self):
        """Get rotation as Rodrigues vectors (axis * angle)."""
        return self.quaternion_wxyz_to_rodrigues(self.get_rotation)
    
    @staticmethod
    def quaternion_wxyz_to_rodrigues(quaternions_wxyz):
        """Convert quaternions (w, x, y, z) to Rodrigues vectors."""
        quaternions_wxyz = torch.nn.functional.normalize(quaternions_wxyz, p=2, dim=-1)
        
        w = quaternions_wxyz[..., 0]
        x = quaternions_wxyz[..., 1]
        y = quaternions_wxyz[..., 2]
        z = quaternions_wxyz[..., 3]
        
        angle = 2 * torch.acos(torch.clamp(w, -1.0 + 1e-7, 1.0 - 1e-7))
        sin_half_angle = torch.sqrt(torch.clamp(1.0 - w * w, min=1e-10))
        
        axis = torch.stack([x, y, z], dim=-1)
        
        non_zero_mask = sin_half_angle > 1e-8
        axis[non_zero_mask] = axis[non_zero_mask] / sin_half_angle[non_zero_mask].unsqueeze(-1)
        axis[~non_zero_mask] = torch.tensor([1.0, 0.0, 0.0], device=axis.device, dtype=axis.dtype)
        
        rodrigues = axis * angle.unsqueeze(-1)
        return rodrigues
    
    @staticmethod
    def rodrigues_to_quaternion_wxyz(rodrigues):
        """Convert Rodrigues vectors to quaternions (w, x, y, z)."""
        angle = torch.norm(rodrigues, dim=-1, keepdim=True)
        angle_safe = torch.clamp(angle, min=1e-10)
        
        axis = rodrigues / angle_safe
        
        half_angle = angle / 2
        w = torch.cos(half_angle)
        xyz = axis * torch.sin(half_angle)
        
        quaternions = torch.cat([w, xyz], dim=-1)
        
        zero_mask = angle.squeeze(-1) < 1e-8
        quaternions[zero_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=rodrigues.device, dtype=rodrigues.dtype)
        
        return torch.nn.functional.normalize(quaternions, dim=-1)

    # ==================== TORCH MATRIX CONVERSIONS ====================
    
    def _quaternion_wxyz_to_matrix(self, quaternions):
        """Convert quaternions (w,x,y,z) to rotation matrices (torch)."""
        quaternions = torch.nn.functional.normalize(quaternions, dim=-1)
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        
        N = quaternions.shape[0]
        R = torch.zeros(N, 3, 3, device=quaternions.device, dtype=quaternions.dtype)
        
        R[:, 0, 0] = 1 - 2*y*y - 2*z*z
        R[:, 0, 1] = 2*x*y - 2*w*z
        R[:, 0, 2] = 2*x*z + 2*w*y
        R[:, 1, 0] = 2*x*y + 2*w*z
        R[:, 1, 1] = 1 - 2*x*x - 2*z*z
        R[:, 1, 2] = 2*y*z - 2*w*x
        R[:, 2, 0] = 2*x*z - 2*w*y
        R[:, 2, 1] = 2*y*z + 2*w*x
        R[:, 2, 2] = 1 - 2*x*x - 2*y*y
        
        return R
    
    def _matrix_to_quaternion_wxyz(self, matrices):
        """Convert rotation matrices to quaternions (w,x,y,z) (torch)."""
        batch_size = matrices.shape[0]
        m = matrices
        trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
        
        quaternions = torch.zeros(batch_size, 4, device=matrices.device, dtype=matrices.dtype)
        
        mask1 = trace > 0
        if mask1.any():
            s = torch.sqrt(trace[mask1] + 1.0) * 2
            quaternions[mask1, 0] = 0.25 * s
            quaternions[mask1, 1] = (m[mask1, 2, 1] - m[mask1, 1, 2]) / s
            quaternions[mask1, 2] = (m[mask1, 0, 2] - m[mask1, 2, 0]) / s
            quaternions[mask1, 3] = (m[mask1, 1, 0] - m[mask1, 0, 1]) / s
        
        mask2 = (~mask1) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
        if mask2.any():
            s = torch.sqrt(1.0 + m[mask2, 0, 0] - m[mask2, 1, 1] - m[mask2, 2, 2]) * 2
            quaternions[mask2, 0] = (m[mask2, 2, 1] - m[mask2, 1, 2]) / s
            quaternions[mask2, 1] = 0.25 * s
            quaternions[mask2, 2] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s
            quaternions[mask2, 3] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s
        
        mask3 = (~mask1) & (~mask2) & (m[:, 1, 1] > m[:, 2, 2])
        if mask3.any():
            s = torch.sqrt(1.0 + m[mask3, 1, 1] - m[mask3, 0, 0] - m[mask3, 2, 2]) * 2
            quaternions[mask3, 0] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s
            quaternions[mask3, 1] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s
            quaternions[mask3, 2] = 0.25 * s
            quaternions[mask3, 3] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s
        
        mask4 = (~mask1) & (~mask2) & (~mask3)
        if mask4.any():
            s = torch.sqrt(1.0 + m[mask4, 2, 2] - m[mask4, 0, 0] - m[mask4, 1, 1]) * 2
            quaternions[mask4, 0] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s
            quaternions[mask4, 1] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s
            quaternions[mask4, 2] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s
            quaternions[mask4, 3] = 0.25 * s
        
        return torch.nn.functional.normalize(quaternions, dim=-1)

    # ==================== SETTERS ====================
    
    def from_scaling(self, scales):
        inner = torch.clamp(torch.square(scales) - self.mininum_kernel_size ** 2, min=1e-12)
        scales_adjusted = torch.sqrt(inner)
        self._scaling = self.inverse_scaling_activation(scales_adjusted) - self.scale_bias
        
    def from_rotation(self, rots):
        """Set rotation from quaternions (w, x, y, z)."""
        self._rotation = rots - self.rots_bias[None, :]
    
    def from_rodrigues(self, rodrigues):
        """Set rotation from Rodrigues vectors."""
        quat_wxyz = self.rodrigues_to_quaternion_wxyz(rodrigues)
        self._rotation = quat_wxyz - self.rots_bias[None, :]
    
    def from_xyz(self, xyz):
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        
    def from_features(self, features):
        self._features_dc = features
        
    def from_opacity(self, opacities):
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias

    # ==================== BOUNDING BOX ====================
    
    def get_bounding_box(self):
        xyz = self.get_xyz
        return xyz.min(dim=0)[0], xyz.max(dim=0)[0]
    
    def get_current_size(self):
        min_xyz, max_xyz = self.get_bounding_box()
        return max_xyz - min_xyz
    
    def get_center(self):
        min_xyz, max_xyz = self.get_bounding_box()
        return (min_xyz + max_xyz) / 2

    # ==================== TRANSFORMATIONS ====================
    
    def translate(self, offset):
        if not isinstance(offset, torch.Tensor):
            offset = torch.tensor(offset, dtype=torch.float32, device=self.device)
        xyz = self.get_xyz + offset
        self.from_xyz(xyz)
    
    def center_at_origin(self):
        self.translate(-self.get_center())
    
    def scale_uniform(self, scale_factor, center=None):
        """Uniform scaling - rotation unchanged."""
        if center is None:
            center = self.get_center()
        elif not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=torch.float32, device=self.device)
        
        xyz = self.get_xyz
        scaled_xyz = (xyz - center) * scale_factor + center
        
        current_scales = self.get_scaling
        scaled_scales = current_scales * abs(scale_factor)
        
        self.from_xyz(scaled_xyz)
        self.from_scaling(scaled_scales)
        self.mininum_kernel_size *= abs(scale_factor)

    def scale_nonuniform(self, scale_factors, center=None):
        """Non-uniform scaling - rotation changes via covariance transformation."""
        if not isinstance(scale_factors, torch.Tensor):
            scale_factors = torch.tensor(scale_factors, dtype=torch.float32, device=self.device)
        
        if center is None:
            center = self.get_center()
        elif not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=torch.float32, device=self.device)
        
        xyz = self.get_xyz
        scaled_xyz = (xyz - center) * scale_factors + center
        self.from_xyz(scaled_xyz)
        
        quats = self.get_rotation
        scales = self.get_scaling
        R = self._quaternion_wxyz_to_matrix(quats)
        
        D_sq = torch.diag_embed(scales ** 2)
        cov = R @ D_sq @ R.transpose(-1, -2)
        
        S = torch.diag_embed(scale_factors.expand(cov.shape[0], -1))
        new_cov = S @ cov @ S.transpose(-1, -2)
        new_cov = (new_cov + new_cov.transpose(-1, -2)) / 2
        new_cov = new_cov + torch.eye(3, device=self.device) * 1e-8
        
        eigenvalues, eigenvectors = torch.linalg.eigh(new_cov)
        new_scales = torch.sqrt(torch.clamp(eigenvalues, min=1e-10))
        new_R = eigenvectors
        
        det = torch.linalg.det(new_R)
        flip_mask = det < 0
        if flip_mask.any():
            new_R[flip_mask, :, -1] *= -1
        
        new_quats = self._matrix_to_quaternion_wxyz(new_R)
        
        self.from_scaling(new_scales)
        self.from_rotation(new_quats)
        self.mininum_kernel_size *= scale_factors.mean().item()

    def scale(self, scale_factors, center=None):
        """Scale the model (auto-selects uniform or non-uniform)."""
        if isinstance(scale_factors, (int, float)):
            self.scale_uniform(scale_factors, center)
        else:
            if not isinstance(scale_factors, torch.Tensor):
                scale_factors = torch.tensor(scale_factors, dtype=torch.float32, device=self.device)
            
            if torch.allclose(scale_factors[0], scale_factors[1], atol=1e-6) and \
               torch.allclose(scale_factors[1], scale_factors[2], atol=1e-6):
                self.scale_uniform(scale_factors[0].item(), center)
            else:
                self.scale_nonuniform(scale_factors, center)

    def scale_to_target_size(self, target_size, mode='fit', center_at_origin=True):
        """Scale to target size."""
        if not isinstance(target_size, torch.Tensor):
            target_size = torch.tensor(target_size, dtype=torch.float32, device=self.device)
        
        current_size = torch.clamp(self.get_current_size(), min=1e-8)
        scale_ratios = target_size / current_size
        
        print(f"Current size: {current_size.cpu().numpy()}")
        print(f"Target size:  {target_size.cpu().numpy()}")
        
        if mode == 'fit':
            scale_factor = scale_ratios.min().item()
            print(f"Uniform scale (fit): {scale_factor:.4f}")
            self.scale_uniform(scale_factor)
        elif mode == 'fill':
            scale_factor = scale_ratios.max().item()
            print(f"Uniform scale (fill): {scale_factor:.4f}")
            self.scale_uniform(scale_factor)
        elif mode == 'exact':
            print(f"Non-uniform scale: {scale_ratios.cpu().numpy()}")
            self.scale_nonuniform(scale_ratios)
        
        if center_at_origin:
            self.center_at_origin()
        
        print(f"Final size:   {self.get_current_size().cpu().numpy()}")

    # ==================== PLY I/O (FIXED) ====================
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        if self._features_rest is not None:
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
        
    def save_ply(self, path, transform=None):
        """
        Save Gaussian model to PLY file.
        
        Args:
            path: output file path
            transform: 3x3 rotation matrix to transform coordinates before saving.
                      Use [[1, 0, 0], [0, 0, -1], [0, 1, 0]] for Y-up to Z-up.
                      Default None means no transformation.
        """
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        if self._features_rest is not None:
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        else:
            f_rest = None
            
        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()
        
        if transform is not None:
            transform = np.array(transform, dtype=np.float32)
            # Transform positions: xyz' = xyz @ T^T
            xyz = np.matmul(xyz, transform.T)
            # Transform rotations: R' = T @ R
            rot_matrices = self._quaternion_wxyz_to_matrix_np(rotation)
            # Apply: result[i] = transform @ rot_matrices[i]
            rot_matrices = np.matmul(transform, rot_matrices)
            rotation = self._matrix_to_quaternion_wxyz_np(rot_matrices)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        if f_rest is not None:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
            
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"Saved {xyz.shape[0]} Gaussians to {path}")

    def load_ply(self, path, transform=None):
        """
        Load Gaussian model from PLY file.
        
        Args:
            path: input file path
            transform: 3x3 rotation matrix that was used when saving.
                      This will be inverted to recover original coordinates.
                      Must match the transform used in save_ply.
                      Default None means no transformation.
        """
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        features_extra = None
        if self.sh_degree > 0:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            if len(extra_f_names) > 0:
                extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
                features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
                for idx, attr_name in enumerate(extra_f_names):
                    features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
                features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        if transform is not None:
            transform = np.array(transform, dtype=np.float32)
            # Inverse transform for positions: xyz = xyz' @ T 
            # (since xyz' = xyz @ T^T, and T @ T^T = I for orthogonal T)
            xyz = np.matmul(xyz, transform)
            
            # Inverse transform for rotations: R = T^T @ R'
            # (since R' = T @ R, we need T^T @ R' = T^T @ T @ R = R)
            rot_matrices = self._quaternion_wxyz_to_matrix_np(rots)
            # Apply: result[i] = T^T @ rot_matrices[i]
            rot_matrices = np.matmul(transform.T, rot_matrices)
            rots = self._matrix_to_quaternion_wxyz_np(rot_matrices)
            
        xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        features_dc = torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        if features_extra is not None:
            features_extra = torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float, device=self.device))
        scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device))
        rots = torch.tensor(rots, dtype=torch.float, device=self.device)
        
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        self._features_dc = features_dc
        self._features_rest = features_extra
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias
        
        inner = torch.clamp(torch.square(scales) - self.mininum_kernel_size ** 2, min=1e-12)
        self._scaling = self.inverse_scaling_activation(torch.sqrt(inner)) - self.scale_bias
        self._rotation = rots - self.rots_bias[None, :]
        
        print(f"Loaded {xyz.shape[0]} Gaussians from {path}")

    # ==================== EXTRACT PARAMETERS ====================
    
    def get_all_parameters(self, rotation_format='rodrigues'):
        """Extract all Gaussian parameters."""
        result = {
            'xyz': self.get_xyz,
            'scales': self.get_scaling,
            'opacity': self.get_opacity,
            'features_dc': self._features_dc,
            'features_rest': self._features_rest,
        }
        
        if rotation_format == 'rodrigues':
            result['rotation'] = self.get_rodrigues
        elif rotation_format == 'quaternion_wxyz':
            result['rotation'] = self.get_rotation
        elif rotation_format == 'quaternion_xyzw':
            result['rotation'] = self.get_rotation_xyzw
        elif rotation_format == 'matrix':
            result['rotation'] = self._quaternion_wxyz_to_matrix(self.get_rotation)
        
        return result
