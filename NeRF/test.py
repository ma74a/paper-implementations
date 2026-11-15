import torch



def sample_points_along_ray(
    ray_origin,
    ray_direction,
    near,
    far,
    num_samples
):
    t_values = torch.linspace(
        start=near,
        end=far,
        steps=num_samples
    )
    points = (
        ray_origin.unsqueeze(0) +
        t_values.unsqueeze(-1) * ray_direction.unsqueeze(0)
    )
    
    return points
    
    
    
if __name__ == "__main__":
    origin = torch.tensor([1.0, 1.0, 1.0])
    direction = torch.tensor([0.0, 1.0, 0.0])

    near = 1.0
    far = 5.0
    num_samples = 3

    points = sample_points_along_ray(
        ray_origin=origin,
        ray_direction=direction,
        near=near,
        far=far,
        num_samples=num_samples,
    )

    print(points)