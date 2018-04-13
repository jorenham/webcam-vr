from vpython import *
import position

POSITION_SCALE_FACTOR = 5
FRAME_WIDTH = 400  # smaller is lower accuracy but faster


def main():
    vs = position.get_stream()

    scene = draw_scene()
    link_camera_user_position(scene, vs)


def draw_scene():
    scene = canvas(title='Interactive scene', width=1200, height=800)
    scene.camera.pos = vector(0, 1, 2)

    sphere(pos=vector(0, 0, 0), radius=1, color=color.green)
    sphere(pos=vector(0, 0, -10), radius=2, color=color.red)

    return scene


def link_camera_user_position(scene, vs):
    try:
        prev = None
        while prev is None:
            f = position.get_frame(stream=vs, width=FRAME_WIDTH)
            prev = position.get_eye_position(f)

        while True:
            f = position.get_frame(stream=vs, width=FRAME_WIDTH)
            pos = position.get_eye_position(f)

            # use previous value if no face is detected
            if pos is None:
                pos = prev
                print('[WARN] No face found')
            else:
                prev = pos

            vpos = vector(1-pos[0], 1-pos[1], 1+pos[2])
            vpos -= vector(.5, .5, .5)
            vpos *= POSITION_SCALE_FACTOR

            scene.camera.pos = vpos
    except KeyboardInterrupt:
        pass
    finally:
        vs.stop()


if __name__ == '__main__':
    main()
