import torch
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    HardPhongShader, FoVPerspectiveCameras, look_at_view_transform
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mesh = load_objs_as_meshes(["data/cow.obj"], device=device)

# Thiết lập máy ảnh
R, T = look_at_view_transform(dist=3, elev=10, azim=0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Cấu hình rasterization: tạo ảnh 256x256, không làm mờ, mỗi pixel chỉ lấy tam giác gần nhất
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Tạo renderer: kết hợp rasterizer (chiếu 3D thành 2D) và shader (tính màu, nhưng ở đây không dùng ánh sáng)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(device=device, lights=None)
)

# Render depth map: chiếu mesh lên ảnh 2D, lấy thông tin độ sâu
fragments = renderer.rasterizer(mesh)
# fragments.zbuf: tensor (1, 256, 256, 1) chứa độ sâu mỗi pixel
depth_map = fragments.zbuf[0, ..., 0].cpu()  # Lấy ma trận (256,256), độ sâu của tam giác gần nhất

# Tính pháp tuyến (normal) của các mặt tam giác
verts = mesh.verts_packed()  # Lấy tất cả đỉnh của mesh, kích thước (V,3)
faces = mesh.faces_packed()   # Lấy tất cả mặt tam giác, kích thước (F,3)

v0 = verts[faces[:,0]]        # Tọa độ đỉnh đầu tiên của mỗi mặt
v1 = verts[faces[:,1]]        # Tọa độ đỉnh thứ hai
v2 = verts[faces[:,2]]        # Tọa độ đỉnh thứ ba
# Tính pháp tuyến: tích chéo (v1-v0) và (v2-v0), chuẩn hóa để độ dài = 1
face_normals = torch.nn.functional.normalize(torch.cross(v1-v0, v2-v0), dim=1) # (F,3)

# Tạo normal map: pháp tuyến cho mỗi pixel
pix_to_face = fragments.pix_to_face[0, ..., 0]    # (256,256), chỉ số mặt tam giác cho mỗi pixel
bary_coords = fragments.bary_coords[0, ..., 0, :]  # (256,256,3), tọa độ barycentric trong tam giác

# Khởi tạo normal map rỗng, kích thước (256,256,3) để lưu pháp tuyến (x,y,z)
H, W = pix_to_face.shape  # 256,256
normal_map = torch.zeros((H, W, 3), device=device)

# Gán pháp tuyến nội suy cho các pixel hợp lệ
valid = pix_to_face >= 0            # Mask các pixel có mặt tam giác (không phải nền)
f = pix_to_face[valid]              # Chỉ số mặt tam giác của các pixel hợp lệ
bc = bary_coords[valid]             # Tọa độ barycentric của các pixel hợp lệ
vnorms = face_normals[f]            # Pháp tuyến của các mặt tương ứng
# Nội suy pháp tuyến: dùng tọa độ barycentric để tính pháp tuyến tại pixel
normal_map[valid] = bc[:,0:1]*vnorms[:,0:1] + bc[:,1:2]*vnorms[:,1:2] + bc[:,2:3]*vnorms[:,2:3]

# Chuẩn hóa normal map để mỗi pháp tuyến có độ dài = 1
normal_map = torch.nn.functional.normalize(normal_map, dim=2).cpu().numpy()
# Chuyển pháp tuyến từ [-1,1] sang [0,1] để hiển thị dưới dạng màu RGB
normal_map_vis = (normal_map + 1)/2.0

# Hiển thị kết quả
plt.figure(figsize=(10,5))  
plt.subplot(1,2,1)          
plt.title("Depth Map")      
plt.imshow(depth_map, cmap="gray")  
plt.axis("off")            

plt.subplot(1,2,2)          
plt.title("Normal Map")     
plt.imshow(normal_map_vis)   
plt.axis("off")             
plt.show()                  