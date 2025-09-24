import pyrealsense2 as rs

def main():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(cfg)
    try:
        vs = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = vs.get_intrinsics()
        print("fx, fy, cx, cy, width, height =",
              intr.fx, intr.fy, intr.ppx, intr.ppy, intr.width, intr.height)
        print("distortion:", intr.model, intr.coeffs)
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
