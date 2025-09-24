import argparse, os, time, math
import numpy as np
import cv2
import pyrealsense2 as rs

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_intrinsics_file(intrin, width, height, out_folder):
    # Save as a simple text file SplaTAM-style: fx fy cx cy width height
    path = os.path.join(out_folder, "intrinsic", "camera.txt")
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(f"{intrin.fx} {intrin.fy} {intrin.ppx} {intrin.ppy} {width} {height}\n")
    print(f"[intrinsics] -> {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output scene folder, e.g., ./data/my_room")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=10.0, help="seconds to record")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--visualize", action="store_true", help="Show preview windows")
    args = ap.parse_args()

    scene = args.out
    color_dir = os.path.join(scene, "color")
    depth_dir = os.path.join(scene, "depth")
    intr_dir  = os.path.join(scene, "intrinsic")
    ensure_dir(color_dir); ensure_dir(depth_dir); ensure_dir(intr_dir)
    ts_path = os.path.join(scene, "timestamps.txt")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    # Use native depth format (Z16), typically 16-bit depth in millimeters
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    # Start pipeline
    profile = pipeline.start(config)

    # Align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()  # contains fx, fy, ppx, ppy, coeffs, width, height

    save_intrinsics_file(intr, args.width, args.height, scene)

    idx = 0
    start = time.time()
    next_frame_time = start
    frame_interval = 1.0 / max(1, args.fps)

    print("[recording] Press Ctrl+C to stop early.")
    with open(ts_path, "w") as ts_file:
        try:
            while True:
                now = time.time()
                if now < next_frame_time:
                    time.sleep(max(0.0, next_frame_time - now))

                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                depth = aligned.get_depth_frame()
                color = aligned.get_color_frame()
                if not depth or not color:
                    continue

                # Convert to numpy
                color_np = np.asanyarray(color.get_data())
                depth_np = np.asanyarray(depth.get_data())  # uint16, in millimeters

                # Save frames
                color_path = os.path.join(color_dir, f"{idx}.jpg")
                depth_path = os.path.join(depth_dir, f"{idx}.png")
                cv2.imwrite(color_path, color_np, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                # Save 16-bit PNG
                cv2.imwrite(depth_path, depth_np)

                # Timestamp (seconds since epoch)
                ts = time.time()
                ts_file.write(f"{idx} {ts:.6f}\n")
                ts_file.flush()

                if args.visualize:
                    cv2.imshow("color", color_np)
                    # Normalize depth for display only
                    disp = (np.clip(depth_np, 0, 3000) / 3000.0 * 255.0).astype(np.uint8)
                    cv2.imshow("depth(mm)", disp)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                idx += 1
                next_frame_time += frame_interval
                if (time.time() - start) >= args.duration:
                    break
        except KeyboardInterrupt:
            print("\n[stop] KeyboardInterrupt")
        finally:
            pipeline.stop()
            if args.visualize:
                cv2.destroyAllWindows()

    print(f"[done] Saved {idx} frames to: {scene}")
    print("  color/: JPG, depth/: 16-bit PNG in millimeters")
    print("  intrinsic/camera.txt with fx fy cx cy width height")
    print("  timestamps.txt with frame index + epoch time")

if __name__ == "__main__":
    main()
