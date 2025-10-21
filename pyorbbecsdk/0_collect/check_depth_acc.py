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


def pick_points(geom, title, min_points=1):
    print(f"\n[Pick] {title}，在点云中选择两个点")
    print("  • Shift + 左键: 选择一个点")
    print("  • Q / Esc: 完成\n")
    while True:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=title, width=1280, height=800)
        vis.add_geometry(geom)
        vis.run()
        idxs = vis.get_picked_points()
        vis.destroy_window()

        if len(idxs) < min_points:
            print(f"ERROR: need at least {min_points} points, got {len(idxs)}.  Please pick again.")
            continue
        # success!
        return np.asarray(geom.points)[idxs]


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
    ap.add_argument("--base_dir", default= 'Training_set_orbbec_data', help="Directory containing frame_*.jpg, frame_*_depth.npy, etc.")
    ap.add_argument("--voxel", type=float, default=0.0, help="点云体素下采样（米）")
    ap.add_argument("--zmax", type=float, default=3.0, help="可视化最大深度范围（米）")
    ap.add_argument("--save_ply", type=str, default=None, help="保存点云为 PLY 文件路径（可选）")
    args = ap.parse_args()

    frames = find_frames(args.base_dir)
    if not frames:
        raise RuntimeError("找不到匹配的帧文件。")

    np.random.seed(42)

    # Outer loop: keep selecting random frames until user quits
    while True:
        idx = np.random.randint(0, len(frames))
        color_png, depth_png, intr_color, intr_depth, frame_num = frames[idx]
        print(f"[FRAME] 随机选取 索引={idx}, 图片={frame_num}\n  color={os.path.basename(color_png)}\n  depth={os.path.basename(depth_png)}")

        Kd = load_K(intr_color)
        pts, cols = backproject_aligned(depth_png, color_png, Kd, z_max=args.zmax)

        # build point cloud once per frame (apply voxel downsample if requested)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if cols is not None:
            pcd.colors = o3d.utility.Vector3dVector(cols)
        if args.voxel > 0:
            pcd = pcd.voxel_down_sample(args.voxel)

        # Inner loop: allow repeated picking on the same frame
        while True:
            picked_xyz = pick_points(pcd, title=f"图片 {frame_num} - 在点云中选择两个点", min_points=2)
            print(f"[Pick] 选择 {picked_xyz.shape[0]} 个点.")
            for i, p in enumerate(picked_xyz):
                print(f"  Pick[{i}]: ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})")
            if picked_xyz.shape[0] >= 2:
                dist = np.linalg.norm(picked_xyz[0] - picked_xyz[1])
                print(f"点 0 和 1 之间的距离： {dist*100.0:.6f} cm")

            if args.save_ply:
                o3d.io.write_point_cloud(args.save_ply, pcd)
                print(f"Point cloud saved to {args.save_ply}")

            choice = input("\nOptions: [n]ext 下一帧随机图片， [r]epick 重新在此图片选择点， [q]uit 退出 > ").strip().lower()
            if choice == "n":
                break  # go to next random frame
            if choice == "q":
                print("Exiting.")
                return
            # otherwise repeat picking on same frame (including empty input)


if __name__ == "__main__":
    main()
