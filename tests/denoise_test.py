import os
import numpy as np
import sapien

# --- enable ray tracing + OIDN denoiser ---
sapien.render.set_camera_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(32)
sapien.render.set_ray_tracing_path_depth(4)
sapien.render.set_ray_tracing_denoiser("oidn")
print("denoiser set to:", sapien.render.get_ray_tracing_denoiser())

# --- minimal scene with geometry + lighting ---
scene = sapien.Scene()
scene.set_ambient_light([0.3, 0.3, 0.3])
scene.add_directional_light([0, 0.5, -1], [3.0, 3.0, 3.0])

builder = scene.create_actor_builder()
builder.add_box_visual(half_size=[0.5, 0.5, 0.5])
builder.build_kinematic(name="box")

cam = scene.add_camera("cam", width=256, height=256, fovy=1.0, near=0.1, far=100.0)

# look-at pose: columns = forward(+x), left(+y), up(+z)
cam_pos = np.array([-2.5, -2.5, 2.0])
forward = -cam_pos / np.linalg.norm(cam_pos)
left = np.cross([0, 0, 1], forward); left /= np.linalg.norm(left)
up = np.cross(forward, left)
mat = np.eye(4)
mat[:3, 0], mat[:3, 1], mat[:3, 2] = forward, left, up
mat[:3, 3] = cam_pos
cam.entity.set_pose(sapien.Pose(mat))

scene.update_render()
cam.take_picture()
rgba = cam.get_picture("Color")  # HxWx4 float
print("picture shape:", rgba.shape, "dtype:", rgba.dtype)
print("finite:", np.isfinite(rgba).all(), "| min/max:", float(rgba.min()), float(rgba.max()))
print("nonzero pixels:", int((rgba[..., :3].sum(-1) > 0.01).sum()), "/", rgba.shape[0]*rgba.shape[1])

# --- prove the OIDN CUDA device module actually loaded ---
with open(f"/proc/{os.getpid()}/maps") as f:
    oidn = sorted(set(l.split()[-1] for l in f if "OpenImageDenoise" in l))
print("OIDN libs mapped:")
for p in oidn:
    print("  ", os.path.basename(p))
assert any("device_cuda" in p for p in oidn), "CUDA denoise device module NOT loaded!"
print("\nRESULT: OIDN CUDA denoise device loaded and ray-traced render succeeded.")
