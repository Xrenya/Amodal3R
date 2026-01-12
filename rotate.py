import torch
import torch.nn.functional as F


# ==================== ROTATION UTILITIES ====================

def euler_to_rotation_matrix(yaw, pitch, roll, degrees=True, device='cuda'):
    """
    Convert Euler angles (yaw, pitch, roll) to rotation matrix.
    Convention: ZYX (yaw around Z, pitch around Y, roll around X)
    
    Args:
        yaw: rotation around Z axis
        pitch: rotation around Y axis  
        roll: rotation around X axis
        degrees: if True, angles are in degrees
    
    Returns:
        R: rotation matrix (3, 3)
    """
    if degrees:
        yaw = torch.deg2rad(torch.tensor(yaw, dtype=torch.float32, device=device))
        pitch = torch.deg2rad(torch.tensor(pitch, dtype=torch.float32, device=device))
        roll = torch.deg2rad(torch.tensor(roll, dtype=torch.float32, device=device))
    else:
        yaw = torch.tensor(yaw, dtype=torch.float32, device=device)
        pitch = torch.tensor(pitch, dtype=torch.float32, device=device)
        roll = torch.tensor(roll, dtype=torch.float32, device=device)
    
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    
    # ZYX convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = torch.tensor([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ], device=device, dtype=torch.float32)
    
    return R


def axis_angle_to_rotation_matrix(axis, angle, degrees=True, device='cuda'):
    """
    Convert axis-angle representation to rotation matrix.
    
    Args:
        axis: rotation axis [x, y, z], will be normalized
        angle: rotation angle
        degrees: if True, angle is in degrees
    
    Returns:
        R: rotation matrix (3, 3)
    """
    if not isinstance(axis, torch.Tensor):
        axis = torch.tensor(axis, dtype=torch.float32, device=device)
    axis = F.normalize(axis.float(), dim=-1)
    
    if degrees:
        angle = torch.deg2rad(torch.tensor(angle, dtype=torch.float32, device=device))
    else:
        angle = torch.tensor(angle, dtype=torch.float32, device=device)
    
    x, y, z = axis[0], axis[1], axis[2]
    c, s = torch.cos(angle), torch.sin(angle)
    t = 1 - c
    
    R = torch.tensor([
        [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ], device=device, dtype=torch.float32)
    
    return R


def rodrigues_to_matrix(rodrigues):
    """
    Convert Rodrigues vectors to rotation matrices.
    
    Args:
        rodrigues: tensor of shape (N, 3) or (3,)
    
    Returns:
        R: rotation matrices of shape (N, 3, 3) or (3, 3)
    """
    if rodrigues.dim() == 1:
        rodrigues = rodrigues.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    N = rodrigues.shape[0]
    device = rodrigues.device
    dtype = rodrigues.dtype
    
    angle = torch.norm(rodrigues, dim=-1, keepdim=True)  # (N, 1)
    angle_safe = torch.clamp(angle, min=1e-8)
    
    axis = rodrigues / angle_safe  # (N, 3)
    
    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    # where K is the skew-symmetric matrix of axis
    
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    
    # Skew-symmetric matrices K
    K = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    K[:, 0, 1] = -z
    K[:, 0, 2] = y
    K[:, 1, 0] = z
    K[:, 1, 2] = -x
    K[:, 2, 0] = -y
    K[:, 2, 1] = x
    
    angle = angle.squeeze(-1)  # (N,)
    
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(N, -1, -1)
    sin_angle = torch.sin(angle).view(N, 1, 1)
    cos_angle = torch.cos(angle).view(N, 1, 1)
    
    R = I + sin_angle * K + (1 - cos_angle) * (K @ K)
    
    # Handle zero rotation case
    zero_mask = angle < 1e-8
    if zero_mask.any():
        R[zero_mask] = torch.eye(3, device=device, dtype=dtype)
    
    if squeeze_output:
        R = R.squeeze(0)
    
    return R


def matrix_to_rodrigues(R):
    """
    Convert rotation matrices to Rodrigues vectors.
    
    Args:
        R: rotation matrices of shape (N, 3, 3) or (3, 3)
    
    Returns:
        rodrigues: tensor of shape (N, 3) or (3,)
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    N = R.shape[0]
    device = R.device
    dtype = R.dtype
    
    # Compute angle from trace: trace(R) = 1 + 2*cos(θ)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)  # (N,)
    
    # Compute axis from skew-symmetric part: (R - R^T) / (2*sin(θ))
    sin_angle = torch.sin(angle)
    sin_angle_safe = torch.clamp(sin_angle.abs(), min=1e-8)
    
    axis = torch.zeros(N, 3, device=device, dtype=dtype)
    axis[:, 0] = (R[:, 2, 1] - R[:, 1, 2]) / (2 * sin_angle_safe)
    axis[:, 1] = (R[:, 0, 2] - R[:, 2, 0]) / (2 * sin_angle_safe)
    axis[:, 2] = (R[:, 1, 0] - R[:, 0, 1]) / (2 * sin_angle_safe)
    
    # Normalize axis
    axis = F.normalize(axis, dim=-1)
    
    # Handle small angles (near identity)
    small_angle_mask = angle.abs() < 1e-6
    if small_angle_mask.any():
        axis[small_angle_mask] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    
    # Handle angles near π (axis is in null space of R + I)
    near_pi_mask = (angle - torch.pi).abs() < 1e-6
    if near_pi_mask.any():
        # Use eigenvector method for stability
        for i in torch.where(near_pi_mask)[0]:
            # Find the column of R + I with largest norm
            RpI = R[i] + torch.eye(3, device=device, dtype=dtype)
            norms = torch.norm(RpI, dim=0)
            max_col = torch.argmax(norms)
            axis[i] = F.normalize(RpI[:, max_col], dim=0)
    
    rodrigues = axis * angle.unsqueeze(-1)
    
    if squeeze_output:
        rodrigues = rodrigues.squeeze(0)
    
    return rodrigues


# ==================== MAIN ROTATION FUNCTION ====================

def rotate_gaussians(data, rotation_matrix, center=None, device='cuda'):
    """
    Rotate 3D Gaussians in a scene.
    
    This function properly rotates:
    1. Positions (means) around the center
    2. Gaussian orientations (rotations) by composing with the scene rotation
    
    Args:
        data: dict with keys 'means', 'scales', 'rotations', 'rgbs', 'opacities'
              - means: (N, 3) positions
              - scales: (N, 3) log of scaling factors
              - rotations: (N, 3) Rodrigues vectors
              - rgbs: (N, 3) colors
              - opacities: (N, 1) inverse sigmoid opacities
        rotation_matrix: (3, 3) rotation matrix for the scene
        center: (3,) rotation center, defaults to centroid of means
    
    Returns:
        rotated_data: dict with same structure, with rotated values
    """
    means = data['params']['means'].to(device)
    scales = data['params']['scales'].to(device)
    rotations = data['params']['rotations'].to(device)
    rgbs = data['params']['rgbs'].to(device)
    opacities = data['params']['opacities'].to(device)
    
    R = rotation_matrix.to(device).float()
    
    N = means.shape[0]
    
    # 1. Rotate positions around center
    if center is None:
        center = means.mean(dim=0)
    elif not isinstance(center, torch.Tensor):
        center = torch.tensor(center, dtype=torch.float32, device=device)
    
    centered_means = means - center
    rotated_means = (R @ centered_means.T).T + center
    
    # 2. Rotate Gaussian orientations
    # Convert Rodrigues to rotation matrices
    R_gaussians = rodrigues_to_matrix(rotations)  # (N, 3, 3)
    
    # Compose rotations: R_new = R_scene @ R_gaussian
    R_new = R.unsqueeze(0) @ R_gaussians  # (N, 3, 3)
    
    # Convert back to Rodrigues
    rotated_rodrigues = matrix_to_rodrigues(R_new)  # (N, 3)
    
    # 3. Scales, colors, and opacities remain unchanged
    rotated_data = {
        "params": {
            "means": rotated_means,
            "scales": scales.clone(),
            "rotations": rotated_rodrigues,
            "rgbs": rgbs.clone(),
            "opacities": opacities.clone(),
        }
    }
    
    return rotated_data


def rotate_gaussians_euler(data, yaw=0, pitch=0, roll=0, degrees=True, center=None, device='cuda'):
    """
    Rotate 3D Gaussians using Euler angles.
    
    Args:
        data: dict with Gaussian parameters
        yaw: rotation around Z axis (up)
        pitch: rotation around Y axis
        roll: rotation around X axis
        degrees: if True, angles are in degrees
        center: rotation center
    
    Returns:
        rotated_data: dict with rotated parameters
    """
    R = euler_to_rotation_matrix(yaw, pitch, roll, degrees=degrees, device=device)
    return rotate_gaussians(data, R, center=center, device=device)


def rotate_gaussians_axis_angle(data, axis, angle, degrees=True, center=None, device='cuda'):
    """
    Rotate 3D Gaussians around an axis by an angle.
    
    Args:
        data: dict with Gaussian parameters
        axis: rotation axis [x, y, z]
        angle: rotation angle
        degrees: if True, angle is in degrees
        center: rotation center
    
    Returns:
        rotated_data: dict with rotated parameters
    """
    R = axis_angle_to_rotation_matrix(axis, angle, degrees=degrees, device=device)
    return rotate_gaussians(data, R, center=center, device=device)


# ==================== CONVENIENCE FUNCTIONS ====================

def rotate_gaussians_x(data, angle, degrees=True, center=None, device='cuda'):
    """Rotate around X axis."""
    return rotate_gaussians_axis_angle(data, [1, 0, 0], angle, degrees, center, device)


def rotate_gaussians_y(data, angle, degrees=True, center=None, device='cuda'):
    """Rotate around Y axis."""
    return rotate_gaussians_axis_angle(data, [0, 1, 0], angle, degrees, center, device)


def rotate_gaussians_z(data, angle, degrees=True, center=None, device='cuda'):
    """Rotate around Z axis."""
    return rotate_gaussians_axis_angle(data, [0, 0, 1], angle, degrees, center, device)


# ==================== COMBINED TRANSFORM ====================

def transform_gaussians(data, rotation_matrix=None, translation=None, scale=None, center=None, device='cuda'):
    """
    Apply full transformation (scale, rotate, translate) to 3D Gaussians.
    
    Order of operations: Scale -> Rotate -> Translate
    
    Args:
        data: dict with Gaussian parameters
        rotation_matrix: (3, 3) rotation matrix
        translation: (3,) translation vector
        scale: float or (3,) scale factors
        center: center for rotation/scaling, defaults to centroid
    
    Returns:
        transformed_data: dict with transformed parameters
    """
    means = data['params']['means'].to(device).float()
    scales = data['params']['scales'].to(device).float()
    rotations = data['params']['rotations'].to(device).float()
    rgbs = data['params']['rgbs'].to(device).float()
    opacities = data['params']['opacities'].to(device).float()
    
    N = means.shape[0]
    
    if center is None:
        center = means.mean(dim=0)
    elif not isinstance(center, torch.Tensor):
        center = torch.tensor(center, dtype=torch.float32, device=device)
    
    # Center the points
    new_means = means - center
    new_scales = scales.clone()
    new_rotations = rotations.clone()
    
    # 1. Apply scaling
    if scale is not None:
        if isinstance(scale, (int, float)):
            scale_factors = torch.tensor([scale, scale, scale], dtype=torch.float32, device=device)
        else:
            scale_factors = torch.tensor(scale, dtype=torch.float32, device=device)
        
        # Scale positions
        new_means = new_means * scale_factors
        
        # Scale Gaussian sizes (scales are in log space)
        # log(s * k) = log(s) + log(k)
        new_scales = new_scales + torch.log(scale_factors).unsqueeze(0)
        
        # For non-uniform scaling, need to adjust rotations via covariance
        if not torch.allclose(scale_factors[0], scale_factors[1], atol=1e-6) or \
           not torch.allclose(scale_factors[1], scale_factors[2], atol=1e-6):
            # Get current scales (actual, not log)
            actual_scales = torch.exp(scales)
            
            # Convert rotations to matrices
            R_gaussians = rodrigues_to_matrix(rotations)
            
            # Build covariance: Σ = R @ D² @ R^T
            D_sq = torch.diag_embed(actual_scales ** 2)
            cov = R_gaussians @ D_sq @ R_gaussians.transpose(-1, -2)
            
            # Transform covariance: Σ' = S @ Σ @ S^T
            S = torch.diag_embed(scale_factors.expand(N, -1))
            new_cov = S @ cov @ S.transpose(-1, -2)
            new_cov = (new_cov + new_cov.transpose(-1, -2)) / 2
            new_cov = new_cov + torch.eye(3, device=device) * 1e-8
            
            # Eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(new_cov)
            actual_new_scales = torch.sqrt(torch.clamp(eigenvalues, min=1e-10))
            new_R_gaussians = eigenvectors
            
            # Fix determinant
            det = torch.linalg.det(new_R_gaussians)
            flip_mask = det < 0
            if flip_mask.any():
                new_R_gaussians[flip_mask, :, -1] *= -1
            
            new_scales = torch.log(actual_new_scales)
            new_rotations = matrix_to_rodrigues(new_R_gaussians)
    
    # 2. Apply rotation
    if rotation_matrix is not None:
        R = rotation_matrix.to(device).float()
        
        # Rotate positions
        new_means = (R @ new_means.T).T
        
        # Rotate Gaussian orientations
        R_gaussians = rodrigues_to_matrix(new_rotations)
        R_new = R.unsqueeze(0) @ R_gaussians
        new_rotations = matrix_to_rodrigues(R_new)
    
    # Move back from center
    new_means = new_means + center
    
    # 3. Apply translation
    if translation is not None:
        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(translation, dtype=torch.float32, device=device)
        new_means = new_means + translation
    
    transformed_data = {
        "params": {
            "means": new_means,
            "scales": new_scales,
            "rotations": new_rotations,
            "rgbs": rgbs.clone(),
            "opacities": opacities.clone(),
        }
    }
    
    return transformed_data
