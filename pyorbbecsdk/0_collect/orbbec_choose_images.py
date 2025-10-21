#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奥比中光相机图像采集脚本 - 优化版
功能：采集彩色图像、深度图像和相机内参数据，支持并列显示和点击获取深度信息
基于官方示例优化：depth.py, color.py, sync_align.py
支持命令行参数配置分辨率和保存路径
"""

import os
import sys
import cv2
import numpy as np
import time
import argparse

# 添加pyorbbecsdk路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'install', 'lib'))

from pyorbbecsdk import *
from utils import frame_to_bgr_image

# 全局变量
current_depth_data = None
current_color_image = None
current_depth_colormap = None

# 常量定义
ESC_KEY = 27
MIN_DEPTH = 0.02  # 0.02m (20mm)
MAX_DEPTH = 10.0  # 10.0m (10000mm)
PRINT_INTERVAL = 1  # seconds


class TemporalFilter:
    """时间滤波器，用于平滑深度数据"""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='奥比中光相机数据采集脚本')

    # 分辨率参数
    parser.add_argument('--width', type=int, default=640,
                        help='图像宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='图像高度 (默认: 480)')
    parser.add_argument('--fps', type=int, default=30,
                        help='帧率 (默认: 30)')

    # 保存路径参数
    parser.add_argument('--output-dir', '-o', type=str, default='Training_set_orbbec_data',
                        help='数据保存目录 (默认: Training_set_orbbec_data)')

    # 功能开关参数
    parser.add_argument('--enable-sync', action='store_true', default=True,
                        help='启用帧同步 (默认: 启用)')
    parser.add_argument('--disable-sync', dest='enable_sync', action='store_false',
                        help='禁用帧同步')
    parser.add_argument('--enable-align', action='store_true', default=True,
                        help='启用深度对齐 (默认: 启用)')
    parser.add_argument('--disable-align', dest='enable_align', action='store_false',
                        help='禁用深度对齐')

    # 时间滤波器参数
    parser.add_argument('--temporal-alpha', type=float, default=0.5,
                        help='时间滤波器Alpha值 (默认: 0.5)')

    return parser.parse_args()


def setup_pipeline(width, height, fps, enable_sync=True, enable_align=True):
    """设置奥比中光相机管道，参考官方示例的最佳实践"""
    pipeline = Pipeline()
    config = Config()

    print(f"配置参数: 分辨率 {width}x{height}@{fps}fps")

    try:
        # 配置彩色流（参考color.py）
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if color_profile_list is None:
            print("无法获取彩色流配置")
            return None, None, None

        color_profile = None
        try:
            # 尝试用户指定的分辨率和帧率，RGB格式
            color_profile = color_profile_list.get_video_stream_profile(width, height, OBFormat.RGB, fps)
            print(f"✓ 使用RGB格式: {width}x{height}@{fps}fps")
        except OBError as e:
            print(f"RGB格式 {width}x{height}@{fps}fps 配置失败: {e}")
            try:
                # 尝试用户指定的分辨率和帧率，MJPG格式
                color_profile = color_profile_list.get_video_stream_profile(width, height, OBFormat.MJPG, fps)
                print(f"✓ 使用MJPG格式: {width}x{height}@{fps}fps")
            except OBError as e:
                print(f"MJPG格式 {width}x{height}@{fps}fps 配置失败: {e}")
                try:
                    # 尝试用户指定的宽度，自动高度，RGB格式
                    color_profile = color_profile_list.get_video_stream_profile(width, 0, OBFormat.RGB, fps)
                    print(f"✓ 使用RGB格式（自动高度）: {width}x0@{fps}fps")
                except OBError as e:
                    print(f"使用默认彩色配置: {e}")
                    color_profile = color_profile_list.get_default_video_stream_profile()

        if color_profile:
            config.enable_stream(color_profile)
            print(f"彩色流配置成功: {color_profile}")

        # 配置深度流（参考depth.py）
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is None:
            print("无法获取深度流配置")
            return None, None, None

        depth_profile = None
        try:
            # 尝试用户指定的分辨率和帧率
            depth_profile = depth_profile_list.get_video_stream_profile(width, height, OBFormat.Y16, fps)
            print(f"✓ 深度流: {width}x{height}@{fps}fps")
        except OBError as e:
            print(f"深度流 {width}x{height}@{fps}fps 配置失败: {e}")
            # 使用默认配置
            depth_profile = depth_profile_list.get_default_video_stream_profile()

        if depth_profile:
            config.enable_stream(depth_profile)
            print(f"深度流配置成功: {depth_profile}")

        # 启用帧同步（参考sync_align.py）
        align_filter = None
        if enable_sync:
            try:
                pipeline.enable_frame_sync()
                print("✓ 启用帧同步")
            except Exception as e:
                print(f"帧同步启用失败: {e}")

        # 启用对齐过滤器（参考sync_align.py）
        if enable_align:
            try:
                align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                print("✓ 启用深度到彩色对齐")
            except Exception as e:
                print(f"对齐过滤器创建失败: {e}")
                align_filter = None

        # 启动pipeline
        pipeline.start(config)
        print("✓ Pipeline启动成功")

        return pipeline, config, align_filter

    except Exception as e:
        print(f"Pipeline设置失败: {e}")
        return None, None, None


def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，用于处理点击事件获取深度信息"""
    global current_depth_data

    if event == cv2.EVENT_LBUTTONDOWN and current_depth_data is not None:
        height, width = current_depth_data.shape
        window_name = param

        if window_name == "Combined View":
            # 组合视图，需要判断点击位置
            if x < width:  # 点击在彩色图像区域
                if 0 <= x < width and 0 <= y < height:
                    depth_value_mm = current_depth_data[y, x]  # 毫米单位
                    if depth_value_mm > 0:
                        print(f"彩色图像位置 ({x}, {y}): {depth_value_mm / 1000:.3f}m ({depth_value_mm:.1f}mm)")
                    else:
                        print(f"彩色图像位置 ({x}, {y}): 无效深度")
            elif x >= width:  # 点击在深度图像区域
                x_depth = x - width
                if 0 <= x_depth < width and 0 <= y < height:
                    depth_value_mm = current_depth_data[y, x_depth]  # 毫米单位
                    if depth_value_mm > 0:
                        print(f"深度图像位置 ({x_depth}, {y}): {depth_value_mm / 1000:.3f}m ({depth_value_mm:.1f}mm)")
                    else:
                        print(f"深度图像位置 ({x_depth}, {y}): 无效深度")
        else:
            # 单独窗口
            if 0 <= x < width and 0 <= y < height:
                depth_value_mm = current_depth_data[y, x]  # 毫米单位
                if depth_value_mm > 0:
                    print(f"位置 ({x}, {y}): {depth_value_mm / 1000:.3f}m ({depth_value_mm:.1f}mm)")
                else:
                    print(f"位置 ({x}, {y}): 无效深度")


def create_combined_view(color_image, depth_colormap):
    """创建彩色图和深度图的并列显示"""
    if color_image is None or depth_colormap is None:
        return None

    # 确保两个图像具有相同的高度
    height = min(color_image.shape[0], depth_colormap.shape[0])
    color_resized = cv2.resize(color_image, (color_image.shape[1], height))
    depth_resized = cv2.resize(depth_colormap, (depth_colormap.shape[1], height))

    # 水平拼接
    combined = np.hstack((color_resized, depth_resized))

    # 添加分割线和标题
    cv2.line(combined, (color_resized.shape[1], 0), (color_resized.shape[1], height), (255, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Color Image (Click for depth)", (10, 30), font, 0.6, (255, 255, 255), 2)
    cv2.putText(combined, "Depth Image", (color_resized.shape[1] + 10, 30), font, 0.6, (255, 255, 255), 2)

    return combined


def process_frames(color_frame, depth_frame, align_filter, temporal_filter):
    """处理彩色帧和深度帧，参考官方示例的处理方式"""
    global current_depth_data, current_color_image, current_depth_colormap

    if not color_frame or not depth_frame:
        return None, None, None

    # 处理彩色帧（参考color.py）
    color_image = frame_to_bgr_image(color_frame)
    if color_image is None:
        print("彩色帧转换失败")
        return None, None, None

    current_color_image = color_image

    # 处理深度帧（参考depth.py）
    depth_format = depth_frame.get_format()
    if depth_format != OBFormat.Y16:
        print("深度格式不是Y16")
        return color_image, None, None

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()

    try:
        # 获取深度数据（参考depth.py的处理方式）
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))

        # 转换为毫米单位（scale本身就是转换为毫米的系数）
        depth_data_mm = depth_data.astype(np.float32) * scale  # 转换为毫米
        depth_data_mm = np.where((depth_data_mm > MIN_DEPTH * 1000) & (depth_data_mm < MAX_DEPTH * 1000), depth_data_mm,
                                 0)

        # 应用时间滤波（参考depth.py，使用毫米数据）
        # if temporal_filter:
        #     depth_data_mm = temporal_filter.process(depth_data_mm.astype(np.uint16))
        #     depth_data_mm = depth_data_mm.astype(np.float32)

        # 毫米单位用于可视化和交互
        current_depth_data = depth_data_mm  # 全局变量保存毫米单位用于点击交互

        # 创建深度可视化图像（使用毫米数据）
        depth_image = cv2.normalize(depth_data_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        depth_data_mm = np.where((depth_data_mm > 300) & (depth_data_mm < 5000), depth_data_mm, 0).astype(np.uint16)
        current_depth_colormap = depth_colormap

        # 转换为米单位用于保存
        depth_data_meters = depth_data_mm / 1000.0

        return color_image, depth_data_meters, depth_colormap

    except ValueError as e:
        print(f"深度数据处理失败: {e}")
        return color_image, None, None


def get_camera_intrinsics(pipeline):
    """获取相机内参"""
    try:
        camera_param = pipeline.get_camera_param()
        if camera_param:
            color_intrinsic = camera_param.rgb_intrinsic
            depth_intrinsic = camera_param.depth_intrinsic

            color_K = np.array([
                [color_intrinsic.fx, 0, color_intrinsic.cx],
                [0, color_intrinsic.fy, color_intrinsic.cy],
                [0, 0, 1]
            ])

            depth_K = np.array([
                [depth_intrinsic.fx, 0, depth_intrinsic.cx],
                [0, depth_intrinsic.fy, depth_intrinsic.cy],
                [0, 0, 1]
            ])

            return color_K, depth_K
    except Exception as e:
        print(f"获取相机内参失败: {e}")

    return None, None


def main():
    """主函数"""
    global current_depth_data, current_color_image, current_depth_colormap

    # 解析命令行参数
    args = parse_arguments()

    print("=== 奥比中光相机数据采集程序（可配置版） ===")
    print(f"配置信息:")
    print(f"  - 分辨率: {args.width}x{args.height}@{args.fps}fps")
    print(f"  - 保存目录: {args.output_dir}")
    print(f"  - 帧同步: {'启用' if args.enable_sync else '禁用'}")
    print(f"  - 深度对齐: {'启用' if args.enable_align else '禁用'}")
    print(f"  - 时间滤波Alpha: {args.temporal_alpha}")

    # 创建输出文件夹
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ 创建输出目录: {output_dir}")
    else:
        print(f"✓ 使用现有目录: {output_dir}")

    # 设置pipeline
    pipeline, config, align_filter = setup_pipeline(
        args.width, args.height, args.fps,
        args.enable_sync, args.enable_align
    )
    if pipeline is None:
        print("✗ 初始化相机失败")
        return

    # 创建时间滤波器
    temporal_filter = TemporalFilter(alpha=args.temporal_alpha)

    # 获取相机内参
    color_K, depth_K = get_camera_intrinsics(pipeline)
    if color_K is not None:
        print("✓ 彩色相机内参矩阵:")
        print(color_K)
        print("✓ 深度相机内参矩阵:")
        print(depth_K)

    frame_count = 0
    last_print_time = time.time()

    print("\n=== 开始数据采集 ===")
    print("操作说明:")
    print("  - 点击图像任意位置获取深度信息")
    print("  - 按 's' 键保存当前帧")
    print("  - 按 'c' 键切换显示模式（并列/分离）")
    print("  - 按 'Esc' 或 'q' 键退出程序")

    # 显示模式控制
    combined_mode = True
    cv2.namedWindow("Combined View", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Combined View", mouse_callback, "Combined View")

    try:
        while True:
            try:
                # 等待帧数据
                frames = pipeline.wait_for_frames(100)
                if frames is None:
                    continue

                # 获取彩色帧和深度帧
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # 应用对齐过滤器（参考sync_align.py）
                if align_filter:
                    try:
                        aligned_frames = align_filter.process(frames)
                        if aligned_frames:
                            aligned_frames = aligned_frames.as_frame_set()
                            color_frame = aligned_frames.get_color_frame()
                            depth_frame = aligned_frames.get_depth_frame()
                    except Exception as e:
                        print(f"对齐处理失败: {e}")

                # 处理帧数据
                color_image, depth_data, depth_colormap = process_frames(
                    color_frame, depth_frame, align_filter, temporal_filter)

                if color_image is None:
                    continue

                # 显示中心点深度（参考depth.py）
                if depth_data is not None:
                    center_y = int(depth_data.shape[0] / 2)
                    center_x = int(depth_data.shape[1] / 2)
                    center_distance = depth_data[center_y, center_x]

                    current_time = time.time()
                    if current_time - last_print_time >= PRINT_INTERVAL:
                        print(f"中心点深度: {center_distance:.3f}m ({center_distance * 1000:.1f}mm)")
                        last_print_time = current_time

                # 显示图像
                if combined_mode:
                    if color_image is not None and depth_colormap is not None:
                        combined_view = create_combined_view(color_image, depth_colormap)
                        if combined_view is not None:
                            cv2.imshow("Combined View", combined_view)

                    # 关闭单独窗口
                    try:
                        cv2.destroyWindow("Color Image")
                        cv2.destroyWindow("Depth Image")
                    except:
                        pass
                else:
                    # 分离显示模式
                    cv2.imshow("Color Image", color_image)
                    cv2.setMouseCallback("Color Image", mouse_callback, "Color Image")

                    if depth_colormap is not None:
                        cv2.imshow("Depth Image", depth_colormap)
                        cv2.setMouseCallback("Depth Image", mouse_callback, "Depth Image")

                    # 关闭组合窗口
                    try:
                        cv2.destroyWindow("Combined View")
                    except:
                        pass

                # 处理按键
                key = cv2.waitKey(1) & 0xFF

                # 切换显示模式
                if key == ord('c'):
                    combined_mode = not combined_mode
                    print("切换到并列显示模式" if combined_mode else "切换到分离显示模式")

                # 保存数据
                elif key == ord('s'):
                    frame_count += 1
                    frame_name = f"frame_{frame_count:04d}"

                    if color_image is not None:
                        color_filename = os.path.join(output_dir, f"{frame_name}.jpg")
                        success = cv2.imwrite(color_filename, color_image)
                        print(f"{'✓' if success else '✗'} 彩色图像: {color_filename}")

                    if depth_data is not None:
                        # 保存深度数据（米单位）
                        depth_filename = os.path.join(output_dir, f"{frame_name}_depth.npy")
                        try:
                            np.save(depth_filename, depth_data)
                            print(f"✓ 深度数据: {depth_filename}")
                        except Exception as e:
                            print(f"✗ 深度数据保存失败: {e}")

                    if color_K is not None:
                        color_intrinsics_filename = os.path.join(output_dir, f"{frame_name}_intrinsics.npy")
                        depth_intrinsics_filename = os.path.join(output_dir, f"{frame_name}_depth_intrinsics.npy")
                        try:
                            np.save(color_intrinsics_filename, color_K)
                            np.save(depth_intrinsics_filename, depth_K)
                            print(f"✓ 相机内参已保存")
                        except Exception as e:
                            print(f"✗ 内参保存失败: {e}")

                    print(f"=== 第 {frame_count} 帧数据保存完成 ===\n")

                # 退出程序
                elif key == ESC_KEY or key == ord('q'):
                    print("\n准备退出程序...")
                    break

            except KeyboardInterrupt:
                print("\n用户中断程序")
                break
            except Exception as e:
                print(f"处理帧时出错: {e}")
                continue

    finally:
        # 清理资源
        try:
            pipeline.stop()
            print("Pipeline已停止")
        except:
            pass
        cv2.destroyAllWindows()
        print(f"\n程序结束，共保存了 {frame_count} 帧数据到目录: {output_dir}")
        print("数据文件说明:")
        print("  - frame_XXXX.jpg: 彩色图像")
        print("  - frame_XXXX_depth.npy: 深度数据(米单位)")
        print("  - frame_XXXX_color_intrinsics.npy: 彩色相机内参矩阵")
        print("  - frame_XXXX_depth_intrinsics.npy: 深度相机内参矩阵")


if __name__ == "__main__":
    main()
