import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 创建文件夹以保存图像和数据
output_dir = "Training_set_plan_EU4622"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置RealSense管道
pipeline = rs.pipeline()

# 配置流
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)  # RGB流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)   # 深度流

# 启动管道
pipeline.start(config)

align_to = rs.stream.color  # 对齐的目标是RGB流
align = rs.align(align_to)  # 创建对齐对象

correction_value = 90  # 假设你想减去的系统误差为x米

# 获取相机内参
profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
depth_stream = profile.get_stream(rs.stream.depth)

# 获取RGB相机内参
color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

# 获取相机内参矩阵K（3x3）
K = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
              [0, color_intrinsics.fy, color_intrinsics.ppy],
              [0, 0, 1]])

# K = np.array([
#           [464.875581, 0, 297.194931], 
#           [0, 619.310364, 247.931554],
#           [0, 0, 1]
#                 ])

# 循环获取帧数据并保存
frame_count = 0
try:
    while True:
        # 等待一帧
        frames = pipeline.wait_for_frames()

        # 对齐深度帧到RGB帧
        aligned_frames = align.process(frames)
        # 获取对齐后的RGB和深度图像
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # # 获取RGB和深度图像
        # color_frame = frames.get_color_frame()
        # depth_frame = frames.get_depth_frame()

        # if not color_frame or not depth_frame:
        #     continue

        # 转换为NumPy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        #depth_image = depth_image + correction_value
        depth_image_in_meters = depth_image.astype(np.float32) / 1000.0

        # 补偿深度图像的系统误差
        #depth_image_in_meters = depth_image + correction_value  # 从每个深度值中减去correction_value

        # 可选：如果想避免负深度值，可以使用 np.clip 来确保值大于零
        # depth_image = np.clip(depth_image - correction_value, 0, np.inf)


        # 显示RGB图像
        cv2.imshow("RGB Image", color_image)

        # 显示深度图像
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("Depth Image", depth_colormap)
        #cv2.imshow("Depth Image", depth_image)

        # 检测按键
        key = cv2.waitKey(1)


        # 如果按下 's' 键，保存图像和数据
        if key == ord('s'):
        # if True:
            frame_count += 1
            frame_name = f"frame_{frame_count:04d}"

            # 保存RGB图像为PNG
            # color_image_filename = os.path.join(output_dir, f"{frame_name}.png")
            # cv2.imwrite(color_image_filename, color_image)

            color_image_filename = os.path.join(output_dir, f"{frame_name}.jpg")
            cv2.imwrite(color_image_filename, color_image)


            # # 保存深度图和内参矩阵为NPZ格式
            # depth_filename = os.path.join(output_dir, f"{frame_name}_depth.npz")
            # intrinsics_filename = os.path.join(output_dir, f"{frame_name}_intrinsics.npz")

            # 保存深度图和内参矩阵为.npy格式
            depth_filename = os.path.join(output_dir, f"{frame_name}_depth.npy")
            intrinsics_filename = os.path.join(output_dir, f"{frame_name}_intrinsics.npy")
            

            np.save(depth_filename, depth_image_in_meters)  # 保存深度图像，单位为米
            # np.save(depth_filename, depth_image)         # 保存深度图像为.npy格式
            np.save(intrinsics_filename, K)              # 保存内参矩阵K为.npy格式

            # np.savez(depth_filename, depth_image=depth_image)
            # np.savez(intrinsics_filename, K=K)
            print(f"Saved {frame_name}")
            print(K)

        # 按Esc键退出
        if key == 27:
            break
finally:
    # 停止管道
    pipeline.stop()
    cv2.destroyAllWindows()
