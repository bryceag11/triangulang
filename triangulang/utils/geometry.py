import torch


def triangulate_centroid(masks, extrinsics, intrinsics, mask_threshold=0.5):
    """Multi-view triangulation for 3D centroid estimation.

    Casts rays from each camera through mask centroids and finds the 3D point
    that minimizes squared distances to all rays (least-squares triangulation).

    Args:
        masks: [N, H, W] predicted mask logits for N views
        extrinsics: [N, 4, 4] camera-to-world transformation matrices
        intrinsics: [N, 3, 3] camera intrinsic matrices

    Returns:
        centroid: [3] triangulated 3D centroid in world frame
        valid: bool, whether triangulation succeeded
    """
    device = masks.device
    N = masks.shape[0]

    # Collect valid rays
    ray_origins = []
    ray_dirs = []

    for i in range(N):
        mask = masks[i]
        mask_binary = (torch.sigmoid(mask) > mask_threshold).float()

        # Skip if no valid mask
        if mask_binary.sum() < 10:
            continue

        # Compute 2D mask centroid (weighted by mask confidence)
        H, W = mask.shape
        y_coords = torch.arange(H, device=device).float().view(-1, 1).expand(H, W)
        x_coords = torch.arange(W, device=device).float().view(1, -1).expand(H, W)

        mask_sum = mask_binary.sum()
        u = (x_coords * mask_binary).sum() / mask_sum  # x centroid
        v = (y_coords * mask_binary).sum() / mask_sum  # y centroid

        # Compute ray in camera frame
        K_inv = torch.inverse(intrinsics[i])  # [3, 3]
        pixel_homo = torch.tensor([u, v, 1.0], device=device)  # [3]
        ray_cam = K_inv @ pixel_homo  # [3] direction in camera frame
        ray_cam = ray_cam / ray_cam.norm()  # normalize

        # Transform to world frame
        # extrinsics is camera-to-world: T_cw
        R = extrinsics[i, :3, :3]  # [3, 3] rotation
        t = extrinsics[i, :3, 3]   # [3] translation (camera position in world)

        ray_world = R @ ray_cam  # direction in world frame
        ray_world = ray_world / ray_world.norm()  # normalize
        origin_world = t  # camera origin in world frame

        ray_origins.append(origin_world)
        ray_dirs.append(ray_world)

    # Need at least 2 rays for triangulation
    if len(ray_origins) < 2:
        return torch.zeros(3, device=device), False

    # Stack rays
    origins = torch.stack(ray_origins)  # [M, 3]
    dirs = torch.stack(ray_dirs)        # [M, 3]
    M = origins.shape[0]

    # Least-squares triangulation:
    # Find point c that minimizes sum of squared distances to rays
    # For ray r_i(t) = o_i + t * d_i, distance to point c is:
    # ||(c - o_i) - ((c - o_i) . d_i) * d_i||
    #
    # Closed form: c = (sum_i (I - d_i d_i^T))^{-1} (sum_i (I - d_i d_i^T) o_i)

    I = torch.eye(3, device=device)
    A = torch.zeros(3, 3, device=device)
    b = torch.zeros(3, device=device)

    for i in range(M):
        d = dirs[i]
        o = origins[i]
        P = I - torch.outer(d, d)  # projection matrix orthogonal to ray
        A = A + P
        b = b + P @ o

    # Solve Ac = b (use lstsq to avoid SIGABRT on singular matrices)
    try:
        result = torch.linalg.lstsq(A.unsqueeze(0), b.unsqueeze(0).unsqueeze(-1))
        centroid = result.solution.squeeze()
    except RuntimeError:
        # Singular matrix (rays are parallel)
        # Fall back to midpoint of closest approach between first two rays
        o1, d1 = origins[0], dirs[0]
        o2, d2 = origins[1], dirs[1]
        # Solve for t1, t2 that minimize ||(o1 + t1*d1) - (o2 + t2*d2)||
        # This is a 2x2 linear system
        w0 = o1 - o2
        a = d1.dot(d1)
        b_val = d1.dot(d2)
        c = d2.dot(d2)
        d_val = d1.dot(w0)
        e = d2.dot(w0)
        denom = a * c - b_val * b_val
        if abs(denom) < 1e-8:
            # Rays are parallel, use midpoint of origins
            centroid = (o1 + o2) / 2
        else:
            t1 = (b_val * e - c * d_val) / denom
            t2 = (a * e - b_val * d_val) / denom
            p1 = o1 + t1 * d1
            p2 = o2 + t2 * d2
            centroid = (p1 + p2) / 2

    return centroid, True
