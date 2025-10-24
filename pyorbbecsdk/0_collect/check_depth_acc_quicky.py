# Author: Ziyuan Qin
# 用法示例：
# python orbbec_data_collection/check_depth_acc.py --base_dir /home/qin/奥比数据/一汽红旗/
# 如果要退出一个点云，在可视化界面里按 Q 或 Esc 键。

import os, glob, argparse, re
import numpy as np
import cv2
import open3d as o3d

def load_K(path):
    if path.endswith('.npy'):
        K = np.load(path)
    else:
        K = np.loadtxt(path)
    if K.shape != (3,3):
        raise ValueError(f"Invalid K shape in {path}: {K.shape}")
    return K


def show_pointcloud(geom, title):
    """
    显示点云，支持按键操作
    返回: 'next' - 下一帧, 'random' - 随机帧, 'quit' - 退出
    """
    print(f"\n[显示] {title}")
    print("  • Enter/Space: 下一帧（按序）")
    print("  • N: 下一帧（随机）")
    print("  • Q / Esc: 退出程序\n")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=1280, height=800)
    vis.add_geometry(geom)

    # 用于存储用户操作
    user_action = {'action': 'next'}

    def on_enter(vis):
        user_action['action'] = 'next'
        vis.close()
        return False

    def on_n(vis):
        user_action['action'] = 'random'
        vis.close()
        return False

    def on_q(vis):
        user_action['action'] = 'quit'
        vis.close()
        return False

    # 注册按键回调
    vis.register_key_callback(32, on_enter)   # Space
    vis.register_key_callback(257, on_enter)  # Enter
    vis.register_key_callback(78, on_n)       # N
    vis.register_key_callback(110, on_n)      # n
    vis.register_key_callback(81, on_q)       # Q
    vis.register_key_callback(113, on_q)      # q
    vis.register_key_callback(256, on_q)      # Esc

    vis.run()
    vis.destroy_window()

    return user_action['action']



def parse_ts(fname):
    # 文件名形如 1234567.890000.png
    base = os.path.splitext(os.path.basename(fname))[0]
    try:
        return float(base)
    except:
        return None

def parse_frame(fname):
    base = os.path.basename(fname)
    match = re.search(r'frame_(\d*)', base)
    return int(match.group(1)) if match else None

def find_pairs(color_dir, depth_dir, max_dt=0.050):
    cf = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    df = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    color_ts = [(f, parse_ts(f)) for f in cf]
    depth_ts = [(f, parse_ts(f)) for f in df]
    color_ts = [(f,t) for f,t in color_ts if t is not None]
    depth_ts = [(f,t) for f,t in depth_ts if t is not None]
    pairs = []
    j = 0
    for f_c, tc in color_ts:
        # 最近邻匹配
        best = None
        best_dt = 1e9
        while j < len(depth_ts):
            td = depth_ts[j][1]
            dt = abs(td - tc)/1000.0 if td > 1e6 else abs(td - tc)  # 兼容 ms 或 s 时间戳
            if dt < best_dt:
                best_dt = dt; best = depth_ts[j]
                if j+1 < len(depth_ts) and abs(depth_ts[j+1][1]-tc) < abs(td-tc):
                    j += 1
                    continue
            break
        if best is not None and best_dt <= max_dt:
            pairs.append((f_c, best[0], best_dt))
    return pairs

def find_frames(base_dir):
    color_files = sorted(glob.glob(os.path.join(base_dir, "frame_*.jpg")))
    frames = []
    for color in color_files:
        frame_num = parse_frame(color)
        if frame_num is None:
            continue
        depth = color.replace('.jpg', '_depth.npy')
        # intr_color = color.replace('.jpg', '_color_intrinsics.npy')
        intr_color = color.replace('.jpg', '_intrinsics.npy')
        intr_depth = color.replace('.jpg', '_depth_intrinsics.npy')
        if os.path.exists(depth) and os.path.exists(intr_color) and os.path.exists(intr_depth):
            frames.append((color, depth, intr_color, intr_depth, frame_num))
    frames.sort(key=lambda x: x[4])  # sort by frame number
    return frames

def backproject_aligned(depth_path, color_path, K, z_min=0.1, z_max=5.0):
    if depth_path.endswith('.npy'):
        depth = np.load(depth_path)  # assume uint16 mm
    else:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    color = cv2.imread(color_path, cv2.IMREAD_COLOR)      # BGR
    if depth is None or color is None:
        raise RuntimeError("Fail to read images.")
    if depth.max() > 20:
        scale_mm_to_m = 0.001
    else:
        scale_mm_to_m = 1.0
    h, w = depth.shape
    if color.shape[0] != h or color.shape[1] != w:
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32) * scale_mm_to_m
    mask = (z > z_min) & (z < z_max)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack([x[mask], y[mask], z[mask]], axis=1)
    cols = (color.reshape(-1,3)[mask.reshape(-1)] / 255.0)[:, ::-1]  # BGR->RGB
    return pts, cols


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_dir", default= '/home/houbenzhou/文档/xwechat_files/dxc3568zhu_051d/msg/file/2025-10/Training_set_orbbec_data', help="Directory containing frame_*.jpg, frame_*_depth.npy, etc.")
    ap.add_argument("--voxel", type=float, default=0.0, help="点云体素下采样（米）")
    ap.add_argument("--zmax", type=float, default=3.0, help="可视化最大深度范围（米）")
    ap.add_argument("--save_ply", type=str, default=None, help="保存点云为 PLY 文件路径（可选）")
    args = ap.parse_args()

    frames = find_frames(args.base_dir)
    if not frames:
        raise RuntimeError("找不到匹配的帧文件。")

    np.random.seed(42)

    # 从第一帧开始
    idx = 0

    # Outer loop: keep selecting frames until user quits
    while True:
        color_png, depth_png, intr_color, intr_depth, frame_num = frames[idx]
        print(f"[FRAME] 索引={idx}/{len(frames)-1}, 图片={frame_num}\n  color={os.path.basename(color_png)}\n  depth={os.path.basename(depth_png)}")

        Kd = load_K(intr_color)
        pts, cols = backproject_aligned(depth_png, color_png, Kd, z_max=args.zmax)

        # build point cloud once per frame (apply voxel downsample if requested)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if cols is not None:
            pcd.colors = o3d.utility.Vector3dVector(cols)
        if args.voxel > 0:
            pcd = pcd.voxel_down_sample(args.voxel)

        # 显示当前帧的点云，并获取用户操作
        action = show_pointcloud(pcd, title=f"图片 {frame_num} - 点云查看")

        if args.save_ply:
            o3d.io.write_point_cloud(args.save_ply, pcd)
            print(f"Point cloud saved to {args.save_ply}")

        # 根据用户在点云窗口中的按键操作进行处理
        if action == 'next':
            # 下一帧（按序）
            idx = (idx + 1) % len(frames)
            print(f"→ 切换到下一帧")
        elif action == 'random':
            # 随机选择下一帧
            idx = np.random.randint(0, len(frames))
            print(f"→ 随机跳转")
        elif action == 'quit':
            print("退出程序。")
            return


if __name__ == "__main__":
    main()
