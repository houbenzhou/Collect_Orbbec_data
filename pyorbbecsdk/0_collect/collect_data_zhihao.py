#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ubuntu系统下奥比中光相机数据采集脚本
用法示例：
  python3 data_collect_orb_ubuntu.py --base_dir ./data --save_images --save_imu --align hw
  python3 data_collect_orb_ubuntu.py --base_dir ./data --save_images --save_imu --align sw_d2c
"""
# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..','install','lib'))
import os, time, platform, argparse, signal, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','install','lib'))
import numpy as np
import cv2
# from pyorbbecsdk import *
from pyorbbecsdk import (
    Pipeline, Config,
    OBSensorType, OBFormat, OBFrameType
)

# 可能的可选枚举/类（不同版本缺失时自动忽略）
try:
    from pyorbbecsdk import OBAlignMode, AlignFilter, OBStreamType, OBFrameAggregateOutputMode
except Exception:
    OBAlignMode = AlignFilter = OBStreamType = OBFrameAggregateOutputMode = None

try:
    from pyorbbecsdk import OBPropertyID, OBPermissionType
except Exception:
    OBPropertyID = OBPermissionType = None

def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    print("\n[INFO] 收到中断信号，正在安全退出...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser(description="Ubuntu系统下奥比中光相机 RGB-D + IMU 数据采集，支持硬件/软件 D2C 对齐")
parser.add_argument("--base_dir", type=str, default="./data", help="数据保存目录")
parser.add_argument("--save_images", action="store_true", help="保存图像数据")
parser.add_argument("--save_imu", action="store_true", help="保存IMU数据")
parser.add_argument("--rgbd_fps", type=int, default=30, help="RGB-D帧率")
parser.add_argument("--align", type=str, choices=["hw", "sw_d2c"], default="hw",
                    help="对齐模式: hw (硬件 D2C) 或 sw_d2c (软件 D2C)")
parser.add_argument("--flip_display", action="store_true",
                    help="仅预览层水平镜像（显示效果，不影响保存/标定）")
parser.add_argument("--no_display", action="store_true", help="不显示预览窗口（无头模式）")
args = parser.parse_args()

base_dir = os.path.abspath(args.base_dir)  # Ubuntu使用绝对路径
save_images = args.save_images
save_imu = args.save_imu
align_mode = args.align

print(f"[INFO] 运行环境: {platform.system()} {platform.release()}")
print(f"[INFO] 保存图像: {save_images}, 保存IMU: {save_imu}, 对齐模式: {align_mode}")
print(f"[INFO] 数据目录: {base_dir}")

# Ubuntu系统优化设置
if platform.system() == 'Linux':
    try:
        # 设置进程优先级
        os.nice(-5)
        print("[INFO] 已提升进程优先级")
    except PermissionError:
        print("[WARN] 无法提升进程优先级，建议使用sudo运行以获得更好性能")

    # 检查显示环境
    if not args.no_display and not os.environ.get('DISPLAY'):
        print("[WARN] 未检测到DISPLAY环境变量，将启用无头模式")
        args.no_display = True

# 创建目录结构
os.makedirs(base_dir, exist_ok=True)
color_dir = os.path.join(base_dir, "color_frames")
depth_dir = os.path.join(base_dir, "depth_frames")
imu_file = os.path.join(base_dir, "imu_data.csv")

if save_images:
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    print(f"[INFO] 图像保存路径: {color_dir}, {depth_dir}")

if save_imu:
    with open(imu_file, "w", encoding="utf-8") as f:
        f.write("timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")
    print(f"[INFO] IMU数据保存路径: {imu_file}")

pipeline = Pipeline()
config = Config()

# ---------- 设备层镜像关闭 ----------
def try_toggle_mirror(dev, prop_id, want=False):
    """按权限检测与写入镜像布尔属性（want=False=关闭镜像）"""
    if OBPropertyID is None or OBPermissionType is None:
        return
    try:
        if dev.is_property_supported(prop_id, OBPermissionType.PERMISSION_READ):
            cur = dev.get_bool_property(prop_id)
            if dev.is_property_supported(prop_id, OBPermissionType.PERMISSION_WRITE):
                if cur != (True if want else False):
                    dev.set_bool_property(prop_id, True if want else False)
                    print(f"[OK] 镜像设置 {prop_id} -> {want}")
    except Exception as e:
        # 静默忽略：不同机型可能不支持
        pass

def try_disable_all_mirror(dev):
    """尝试关闭所有镜像属性"""
    if OBPropertyID is None:
        print("[INFO] OBPropertyID 不可用，跳过镜像设置")
        return
    try_toggle_mirror(dev, OBPropertyID.OB_PROP_IR_MIRROR_BOOL, False)
    try_toggle_mirror(dev, OBPropertyID.OB_PROP_DEPTH_MIRROR_BOOL, False)
    try_toggle_mirror(dev, OBPropertyID.OB_PROP_COLOR_MIRROR_BOOL, False)
    print("[INFO] 已尝试关闭所有镜像属性")

# ---------- 选择流配置 ----------
def choose_color_profile(pipeline, fps):
    """选择彩色流配置"""
    try:
        lst = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        prof = lst.get_video_stream_profile(640, 480, OBFormat.RGB, fps)
        print(f"[OK] 彩色流配置: 640x480@{fps} RGB")
        return prof
    except Exception as e:
        print(f"[INFO] RGB 640x480@{fps} 不支持: {e}")
        try:
            lst = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            prof = lst.get_video_stream_profile(640, 480, OBFormat.MJPG, fps)
            print(f"[OK] 彩色流配置: 640x480@{fps} MJPG")
            return prof
        except Exception as e2:
            print(f"[ERR] 彩色流配置选择失败: {e2}")
            return None

def choose_depth_profile_default(pipeline):
    """选择深度流配置"""
    try:
        lst = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        prof = lst.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
        print("[OK] 深度流配置: 640x480@30 Y16")
        return prof
    except Exception as e:
        print(f"[INFO] 深度 640x480@30 不支持: {e}")
        try:
            lst = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            prof = lst.get_video_stream_profile(640, 400, OBFormat.Y16, 15)
            print("[OK] 深度流配置: 640x400@15 Y16")
            return prof
        except Exception as e2:
            print(f"[ERR] 深度流配置选择失败: {e2}")
            return None

# ---------- D2C 对齐配置 ----------
def setup_hw_d2c(pipeline, config, color_prof):
    """硬件 D2C 配置：返回用于启动时启用的 depth_prof；失败返回 None"""
    if OBAlignMode is None:
        return None
    try:
        d2c_list = pipeline.get_d2c_depth_profile_list(color_prof, OBAlignMode.HW_MODE)
        if len(d2c_list) == 0:
            print("[INFO] 该彩色配置无可用的硬件 D2C 深度配置")
            return None
        depth_prof = d2c_list[0]
        config.enable_stream(color_prof)
        config.enable_stream(depth_prof)
        config.set_align_mode(OBAlignMode.HW_MODE)
        print("[OK] 硬件 D2C 已启用")
        return depth_prof
    except Exception as e:
        print(f"[INFO] 硬件 D2C 不可用: {e}")
        return None

def setup_sw_d2c_filter(config):
    """软件 D2C 配置：设置 FULL_FRAME_REQUIRE 并创建 AlignFilter（COLOR）"""
    if AlignFilter is None or OBStreamType is None:
        print("[ERR] 当前SDK版本不支持软件 D2C")
        return None
    try:
        if OBFrameAggregateOutputMode is not None:
            config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            print("[OK] 已设置 FULL_FRAME_REQUIRE 用于软件对齐")
    except Exception:
        pass
    print("[OK] 软件 D2C 滤波器准备就绪")
    return AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

# ---------- IMU 配置 ----------
def enable_imu_streams_if_needed(config):
    """根据需要启用IMU流"""
    if not save_imu:
        return
    try:
        config.enable_accel_stream()
        print("[OK] 已启用加速度计流")
    except Exception as e:
        print(f"[ERR] 启用加速度计流失败: {e}")
    try:
        config.enable_gyro_stream()
        print("[OK] 已启用陀螺仪流")
    except Exception as e:
        print(f"[ERR] 启用陀螺仪流失败: {e}")

# ---------- 流配置和对齐设置 ----------
print("[INFO] 正在配置数据流...")
color_prof = choose_color_profile(pipeline, args.rgbd_fps)
depth_prof = None
align_filter = None

if align_mode == "hw":
    if color_prof is None:
        raise RuntimeError("硬件 D2C 需要有效的彩色流配置")
    depth_prof = setup_hw_d2c(pipeline, config, color_prof)
    if depth_prof is None:
        # 回退到软件 D2C
        print("[INFO] 回退到软件 D2C")
        if color_prof:
            config.enable_stream(color_prof)
        depth_prof = choose_depth_profile_default(pipeline)
        if depth_prof:
            config.enable_stream(depth_prof)
        align_filter = setup_sw_d2c_filter(config)
else:  # sw_d2c
    lst = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_prof = lst.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
    config.enable_stream(color_prof)
    print("[OK] 彩色流配置(软件D2C): 640x480@30 RGB")

    lstd = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_prof = lstd.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
    config.enable_stream(depth_prof)
    print("[OK] 深度流配置(软件D2C): 640x480@30 Y16")

    align_filter = setup_sw_d2c_filter(config)

# IMU配置
enable_imu_streams_if_needed(config)

# 设备层镜像关闭
try:
    dev = pipeline.get_device()
    try_disable_all_mirror(dev)
except Exception:
    print("[WARN] 无法获取设备或设置镜像属性")

# 启动管道
print("[INFO] 正在启动相机管道...")
pipeline.start(config)
print("[OK] 相机管道已启动")

# ---------- 保存标定参数 ----------
def save_calibration(base_dir, color_prof, depth_prof, aligned_to_color=True):
    """保存相机标定参数"""
    try:
        if color_prof is None or depth_prof is None:
            print("[INFO] 跳过标定参数保存：需要彩色和深度流配置")
            return

        c_intr = color_prof.get_intrinsic()
        d_intr = depth_prof.get_intrinsic()

        def K_of(intr):
            return np.array([[intr.fx, 0.0, intr.cx],
                             [0.0, intr.fy, intr.cy],
                             [0.0,  0.0,   1.0]], dtype=np.float32)

        K_color = K_of(c_intr)
        K_depth = K_of(d_intr)
        # 对齐后的深度K = 彩色K
        K_depth_aligned = K_color if aligned_to_color else K_depth

        np.savetxt(os.path.join(base_dir, "intrinsics_color.txt"), K_color)
        np.savetxt(os.path.join(base_dir, "intrinsics_depth_raw.txt"), K_depth)
        np.savetxt(os.path.join(base_dir, "intrinsics_depth_aligned.txt"), K_depth_aligned)
        print("[OK] 已保存内参矩阵 (color / depth_raw / depth_aligned)")

        # 保存外参（原始深度->彩色）
        try:
            extr = depth_prof.get_extrinsic_to(color_prof)
            def save_extr(extr_obj, path_txt):
                R = t = None
                for n in ("rotation", "rot", "R"):
                    if hasattr(extr_obj, n):
                        R = getattr(extr_obj, n)
                        break
                for n in ("translation", "trans", "t"):
                    if hasattr(extr_obj, n):
                        t = getattr(extr_obj, n)
                        break

                with open(path_txt, "w", encoding="utf-8") as f:
                    if R is not None:
                        Rm = np.array(R).reshape(3, 3)
                        f.write("R:\n")
                        for r in Rm:
                            f.write("  " + " ".join(f"{v:.9f}" for v in r) + "\n")
                    if t is not None:
                        tv = np.array(t).reshape(3)
                        f.write("t:\n  " + " ".join(f"{v:.9f}" for v in tv) + "\n")

            save_extr(extr, os.path.join(base_dir, "extrinsic_depth_to_color.txt"))
            print("[OK] 已保存外参 (depth->color)")
        except Exception as e:
            print(f"[WARN] 保存外参失败: {e}")

    except Exception as e:
        print(f"[INFO] 跳过标定参数保存: {e}")

save_calibration(base_dir, color_prof, depth_prof, aligned_to_color=True)

# ---------- IMU 数据缓冲/写入 ----------
imu_buffer = {"accel": None, "gyro": None}
imu_lines = []
last_flush = time.time()

def parse_imu(frame):
    """解析IMU数据帧"""
    for getter in ("get_value", "value"):
        if hasattr(frame, getter):
            v = getattr(frame, getter)()
            try:
                if isinstance(v, (tuple, list)) and len(v) >= 3:
                    return float(v[0]), float(v[1]), float(v[2])
                if all(hasattr(v, a) for a in ("x", "y", "z")):
                    return float(v.x), float(v.y), float(v.z)
            except Exception:
                pass
    try:
        buf = np.frombuffer(frame.get_data(), dtype=np.float32)
        if buf.size >= 3:
            return float(buf[0]), float(buf[1]), float(buf[2])
    except Exception:
        pass
    raise RuntimeError("无法解析IMU数据帧")

def ts_of(frame):
    """获取帧时间戳"""
    for n in ("get_timestamp", "timestamp"):
        if hasattr(frame, n):
            try:
                return float(getattr(frame, n)())
            except Exception:
                pass
    return time.time() * 1000.0

def flush_imu():
    """刷新IMU数据到文件"""
    if save_imu and imu_lines:
        with open(imu_file, "a", encoding="utf-8") as f:
            f.writelines(imu_lines)
        imu_lines.clear()

# ---------- 主采集循环 ----------
print("[INFO] 开始数据采集... 按 'q' 或 ESC 键停止")
if not args.no_display:
    cv2.namedWindow("Orbbec RGB-D Overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Orbbec RGB-D Overlay", 800, 600)

last_depth_cmap = None
frame_count = 0
start_time = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames(200)
        if frames is None:
            continue

        frame_count += 1

        # 软件 D2C：对 frames 做对齐处理
        if align_mode == "sw_d2c" and align_filter is not None:
            f2 = align_filter.process(frames)
            if not f2:
                # 容错：对齐失败就跳过这帧
                continue
            try:
                frames = f2.as_frame_set()
            except Exception:
                # 少数版本返回的就是 frameset，不需要转换
                pass

        # ---------- 彩色图像处理 ----------
        overlay = None
        color = frames.get_color_frame()
        if color is not None:
            try:
                w, h = color.get_width(), color.get_height()
                fmt = color.get_format()

                # 获取原始缓冲区数据并立即确保连续性
                raw_buf = color.get_data()
                buf = np.frombuffer(raw_buf, dtype=np.uint8)
                # 立即创建连续的副本
                buf = np.ascontiguousarray(buf)

                if fmt == OBFormat.MJPG:
                    c_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if c_bgr is None:
                        # 尝试另一种解码方式
                        buf_copy = np.copy(buf)
                        c_bgr = cv2.imdecode(buf_copy, cv2.IMREAD_COLOR)

                    # 确保解码后的图像是连续的
                    if c_bgr is not None and not c_bgr.flags['C_CONTIGUOUS']:
                        c_bgr = np.ascontiguousarray(c_bgr)
                else:
                    # RGB格式处理
                    c_rgb = buf.reshape((h, w, 3))
                    # 立即创建连续副本
                    c_rgb = np.ascontiguousarray(c_rgb)
                    c_bgr = cv2.cvtColor(c_rgb, cv2.COLOR_RGB2BGR)
                    # 确保转换后也是连续的
                    c_bgr = np.ascontiguousarray(c_bgr)

                if save_images and c_bgr is not None:
                    ts_c = color.get_timestamp()
                    cv2.imwrite(os.path.join(color_dir, f"{ts_c:.6f}.png"), c_bgr)
                overlay = c_bgr

            except Exception as e:
                print(f"[WARN] 彩色图像处理错误: {e}")
                overlay = None

        # ---------- 深度图像处理（已对齐：HW 或 SW） ----------
        depth = frames.get_depth_frame()
        if depth is not None:
            try:
                dw, dh = depth.get_width(), depth.get_height()

                # 获取原始深度数据并立即确保连续性
                raw_depth_data = depth.get_data()
                d = np.frombuffer(raw_depth_data, dtype=np.uint16)
                # 立即创建连续副本
                d = np.ascontiguousarray(d)
                d = d.reshape((dh, dw))
                # 再次确保reshape后的数组是连续的
                d = np.ascontiguousarray(d)

                # 深度范围过滤减少噪点
                d_filtered = np.where((d > 300) & (d < 5000), d, 0)
                d = np.ascontiguousarray(d_filtered.astype(np.uint16))

                if save_images:
                    ts_d = depth.get_timestamp()
                    cv2.imwrite(os.path.join(depth_dir, f"{ts_d:.6f}.png"), d)

                # 深度可视化
                d_vis = cv2.convertScaleAbs(d, alpha=255.0/5000.0)
                depth_cmap = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)
                # 确保深度彩色图是连续的
                depth_cmap = np.ascontiguousarray(depth_cmap)
                last_depth_cmap = depth_cmap

            except Exception as e:
                print(f"[WARN] 深度图像处理错误: {e}")
                last_depth_cmap = None

        # 显示叠加预览
        if not args.no_display and overlay is not None and last_depth_cmap is not None:
            try:
                dc = np.ascontiguousarray(last_depth_cmap)
                overlay_work = np.ascontiguousarray(overlay)

                if (dc.shape[0], dc.shape[1]) != (overlay_work.shape[0], overlay_work.shape[1]):
                    dc = cv2.resize(dc, (overlay_work.shape[1], overlay_work.shape[0]), interpolation=cv2.INTER_NEAREST)
                    dc = np.ascontiguousarray(dc)

                # 执行叠加操作
                overlay_result = cv2.addWeighted(overlay_work, 0.7, dc, 0.3, 0)
                overlay_result = np.ascontiguousarray(overlay_result)

                if args.flip_display:
                    overlay_result = cv2.flip(overlay_result, 1)
                    overlay_result = np.ascontiguousarray(overlay_result)

                # 添加信息文本
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                info_text = f"Frames: {frame_count} | FPS: {fps:.1f} | Mode: {align_mode}"
                cv2.putText(overlay_result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Orbbec RGB-D Overlay", overlay_result)

            except Exception as e:
                print(f"[WARN] 显示处理错误: {e}")
                # 如果叠加失败，尝试只显示彩色图像
                try:
                    if overlay is not None:
                        display_img = np.ascontiguousarray(overlay)
                        cv2.imshow("Orbbec RGB-D Overlay", display_img)
                except Exception as e2:
                    print(f"[WARN] 备用显示也失败: {e2}")

        # ---------- IMU 数据处理 ----------
        if save_imu:
            acc_fr = gyr_fr = None
            try:
                acc_fr = frames.get_frame(OBFrameType.ACCEL_FRAME)
                if acc_fr is not None:
                    acc_fr = acc_fr.as_accel_frame()
            except Exception:
                pass

            try:
                gyr_fr = frames.get_frame(OBFrameType.GYRO_FRAME)
                if gyr_fr is not None:
                    gyr_fr = gyr_fr.as_gyro_frame()
            except Exception:
                pass

            if acc_fr is not None:
                try:
                    ax, ay, az = parse_imu(acc_fr)
                    imu_buffer["accel"] = (ts_of(acc_fr), ax, ay, az)
                except Exception as e:
                    pass

            if gyr_fr is not None:
                try:
                    gx, gy, gz = parse_imu(gyr_fr)
                    imu_buffer["gyro"] = (ts_of(gyr_fr), gx, gy, gz)
                except Exception as e:
                    pass

            # 当加速度计和陀螺仪数据都有时，写入文件
            if imu_buffer["accel"] and imu_buffer["gyro"]:
                ts_a, ax, ay, az = imu_buffer["accel"]
                ts_g, gx, gy, gz = imu_buffer["gyro"]
                ts_use = max(ts_a, ts_g)
                imu_lines.append(f"{ts_use:.6f},{ax:.6f},{ay:.6f},{az:.6f},{gx:.6f},{gy:.6f},{gz:.6f}\n")
                imu_buffer["accel"] = None
                imu_buffer["gyro"] = None

            # 定期刷新IMU数据到文件
            if time.time() - last_flush > 1.0:
                flush_imu()
                last_flush = time.time()

        # 检查退出键
        if not args.no_display:
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):  # ESC 或 'q'
                print("[INFO] 检测到退出键 - 停止采集")
                break

        # 控制循环频率
        time.sleep(0.005)

except KeyboardInterrupt:
    print("\n[INFO] 收到Ctrl+C - 停止采集")
except Exception as e:
    print(f"[ERR] 采集过程中发生错误: {e}")
finally:
    # 清理资源
    print("[INFO] 正在清理资源...")
    if save_imu:
        flush_imu()
        print("[OK] IMU数据已保存")

    try:
        pipeline.stop()
        print("[OK] 相机管道已停止")
    except Exception as e:
        print(f"[WARN] 停止管道时出错: {e}")

    if not args.no_display:
        cv2.destroyAllWindows()

    # 显示采集统计
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"[INFO] 采集完成 - 总帧数: {frame_count}, 总时间: {total_time:.1f}秒, 平均FPS: {avg_fps:.1f}")
    print(f"[INFO] 数据已保存到: {base_dir}")
