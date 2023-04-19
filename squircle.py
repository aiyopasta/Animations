import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from playsound import playsound

# Save the animation?
save_anim = False

# Pygame + gameloop setup
alpha = 1 / 1.9
width = np.ceil(888 * alpha / 2) * 2
height = np.ceil(1920 * alpha / 2) * 2
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Squircle TikTok")
pygame.init()


# Coordinate Shift
def A(val):
    global width, height
    return np.array([val[0] + width / 2, -val[1] + height / 2])


def A_inv(val):
    global width, height
    return np.array([val[0] - width / 2, -(val[1] - height / 2)])


def A_many(vals):
    return [A(v) for v in vals]   # Galaxy brain function


# Keyframe / timing params
FPS = 60
t = 0
dt = 0.02    # i.e., 1 frame corresponds to +0.01 in parameter space = 0.01 * FPS = +0.6 per second (assuming 60 FPS)
keys = [0,   # Keyframe 0. Wait for logo to flip in! (Do this in post.)
        4,   # Keyframe 1. Draw the pointy square.
        5,   # Keyframe 2. Drop to the "ground".
        8,   # Keyframe 3. Stick the landing
        10,  # Keyframe 4. Reset / undo fall.
        11,  # Keyframe 5. Pause.
        11.5,  # Keyframe 6. Introduce rounded rect (expanding circles + outline).
        20   # Done.
        ]


# Helper functions
# Squash t parameter into given intervals.
def squash(t_, intervals=None):
    global keys
    if intervals is None:
        intervals = keys
    for i in range(len(intervals) - 1):
        if intervals[i] <= t_ < intervals[i + 1]:
            return (t_ - intervals[i]) / (intervals[i + 1] - intervals[i]), i

    return intervals[-1], len(intervals) - 2


# Specific case of the above. We squash t into equally sized intervals.
def squash2(t_, n_intervals=1):
    intervals = [float(i) / n_intervals for i in range(n_intervals + 1)]
    return squash(t_, intervals)


# Easing functions.
# TODO: Add more.
def ease_inout(t_):
    return np.power(t_, 2) / (np.power(t_, 2) + np.power(1 - t_, 2))


def ease_out(t_):
    return 1.0 - np.power(1.0 - t_, 2.0)


# From ChatGPT
def rgb_to_hex(rgb):
    return '#' + ''.join(['%02x' % int(val) for val in rgb])


# Parametric shapes
def line(u, start, fin):
    return ((1 - u) * start) + (u * fin)


def rectangle(u, center_x, center_y, tall, wide):
    u = 2 * np.pi * u
    sec = lambda p: 1. / np.cos(p)

    dist = sec(u - (np.pi / 2 * np.floor(2. / np.pi * (u + (np.pi / 4)))))
    center = np.array([center_x, center_y])
    return np.array([wide * np.cos(u) * dist / 2, tall * np.sin(u) * dist / 2]) + center


def circle(u, radius, center_x=0, center_y=0):
    tau = 2 * np.pi * u
    return (radius * np.array([np.cos(tau), np.sin(tau)])) + np.array([center_x, center_y])


def rounded_rect(u, center_x, center_y, inner_tall, inner_wide, corner_radius):
    '''
        "Inner tall/wide" means how tall / wide it would be for corner_radius = 0.
        Let's just do it the dumb way: 8 segments, trace each one.
    '''
    outer_tall, outer_wide = inner_tall + (2 * corner_radius), inner_wide + (2 * corner_radius)
    tau, segment = squash2(u, 8)
    # 0. Right straight
    if segment == 0:
        start = np.array([center_x + (outer_wide / 2), center_y - (inner_wide / 2)])
        finish = np.array([start[0], center_y + (inner_wide / 2)])
        return line(tau, start, finish)
    # 1. Topright rounded corner
    elif segment == 1:
        circle_center = np.array([center_x + (inner_wide / 2), center_y + (inner_tall / 2)])
        quarter_tau = tau / 4
        return circle(quarter_tau, corner_radius, *circle_center)
    # 2. Top straight
    elif segment == 2:
        start = np.array([center_x + (inner_wide / 2), center_y + (outer_wide / 2)])
        finish = np.array([center_x - (inner_wide / 2), start[1]])
        return line(tau, start, finish)
    # 3. Topleft rounded corner
    elif segment == 3:
        circle_center = np.array([center_x - (inner_wide / 2), center_y + (inner_tall / 2)])
        quarter_tau = (tau / 4) + 0.25
        return circle(quarter_tau, corner_radius, *circle_center)
    # 4. Left straight
    elif segment == 4:
        start = np.array([center_x - (outer_wide / 2), center_y + (inner_wide / 2)])
        finish = np.array([start[0], center_y - (inner_wide / 2)])
        return line(tau, start, finish)
    # 5. Bottom left rounded corner
    elif segment == 5:
        circle_center = np.array([center_x - (inner_wide / 2), center_y - (inner_tall / 2)])
        quarter_tau = (tau / 4) + 0.5
        return circle(quarter_tau, corner_radius, *circle_center)
    # 6. Bottom straight
    elif segment == 6:
        start = np.array([center_x - (inner_wide / 2), center_y - (outer_wide / 2)])
        finish = np.array([center_x + (inner_wide / 2), start[1]])
        return line(tau, start, finish)
    # 7. Bottom right rounded corner
    circle_center = np.array([center_x + (inner_wide / 2), center_y - (inner_tall / 2)])
    quarter_tau = (tau / 4) + 0.75
    return circle(quarter_tau, corner_radius, *circle_center)


# Shape sampling methods
def get_rekt_pts(u, tall, wide, center_x=0., center_y=0., du=0.001):
    pts = []
    for u_ in np.arange(0, u + du, du):
        pts.append(rectangle(u_, center_x, center_y, tall, wide))
    return pts


def get_circle_pts(u, radius, center_x, center_y, du=0.001):
    pts = []
    for u_ in np.arange(0, u + du, du):
        pts.append(circle(u_, radius, center_x, center_y))
    return pts


def get_rounded_rekt_pts(u, center_x, center_y, inner_tall, inner_wide, corner_radius, du):
    pts = []
    for u_ in np.arange(0, u + du, du):
        pts.append(rounded_rect(u_, center_x, center_y, inner_tall, inner_wide, corner_radius))
    return pts


# Rotate points array
def rotate(pts, radians, anchor):
    assert len(anchor) > 1
    M = np.array([[np.cos(radians), -np.sin(radians)],
                  [np.sin(radians), np.cos(radians)]])

    return [(M @ (v - anchor)) + anchor for v in pts]


# Make points drop downwards with gravity
def drop_down(pts, tau, vy0, grav):
    return [np.array([p[0], p[1] + (vy0 * tau) + (0.5 * grav * np.power(tau, 2))]) for p in pts]


# Output positions of points in camera ref
def cam_ref(pts, cam_y):
    return [np.array([p[0], p[1] - cam_y]) for p in pts]


# Main animation params / data-structure
darkgray = (20, 20, 20)
white = (255, 255, 255)
black = (0, 0, 0)

# Additional params / knobs
play_music = True


def main():
    global t, dt, keys, FPS, save_anim, play_music, darkgray, black, white

    # Pre-animation setup
    clock = pygame.time.Clock()
    run = True

    # Animation saving setup
    path_to_save = '/Users/adityaabhyankar/Desktop/Programming/Animations/pygame_output'
    if save_anim:
        for filename in os.listdir(path_to_save):
            # Check if the file name follows the required format
            b1 = filename.startswith("frame") and filename.endswith(".png")
            b2 = filename.startswith("output.mp4")
            if b1 or b2:
                os.remove(os.path.join(path_to_save, filename))
                print('Deleted frame ' + filename)

    # Load images
    logo = pygame.image.load('/Users/adityaabhyankar/Desktop/Programming/Animations/tiktok_logo.png')
    logo_size = logo.get_size()
    factor = 0.4
    logo = pygame.transform.scale(logo, (logo_size[0] * factor, logo_size[1] * factor))

    # Set text font
    font = pygame.font.SysFont("Avenir Next", 45)
    font2 = pygame.font.SysFont("Avenir Next", 35)
    text = font.render("Cool TikTok :)", True, white)  # create the text object
    ow = font2.render("oww.", True, (0, 200, 200))  # create the text object
    too_sharp = font.render("Too sharp!", True, white)  # create the text object

    # Ground y
    ground_y_base = 2100

    # Play music
    if not save_anim and play_music:
        playsound('木(剪辑版).mov', block=False)

    # Main Game loop ———————————————————————————————————————————————————————————————————————————————————————
    count = 0
    while run:
        # Draw preliminary stuff for animation
        count += 1
        window.fill(black)

        # Animation!
        u, keyframe = squash(t)
        # Keyframe 0 —— Blank (will do in post)
        if keyframe == 0:
            window.blit(text, (100, 100))
            window.blit(logo, ((width / 2) - (logo_size[1] * factor / 2) + 20, height / 3))

        # Keyframe 1 —— Draw the square.
        if keyframe == 1:
            window.blit(text, (100, 100))
            u = ease_inout(u)
            side = logo_size[1] * factor
            x, y = (width / 2) - (side / 2) + 20, height / 3
            center = A_inv([x + (side / 2.3), y + (side / 2)])
            pts = get_rekt_pts(u, side * 1.5, side * 1.5, *center, du=0.001)
            if len(pts) > 1:
                if len(pts) > 2:
                    pygame.draw.polygon(window, darkgray, A_many(pts), 0)

                pygame.draw.lines(window, white, False, A_many(pts), width=5)

            window.blit(logo, (x, y))

        # Keyframe 2 —— Drop to the "ground".
        if keyframe == 2:
            side = logo_size[1] * factor
            x, y = (width / 2) - (side / 2) + 20, height / 3  # topleft of the image
            center = A_inv([x + (side / 2.3), y + (side / 2)])
            pts = get_rekt_pts(1, side * 1.5, side * 1.5, *center, du=0.25/2)  # more efficient sampling

            # Drop downwards computation
            tau = u * 10  # parameter for quadratic (so that squaring doesn't make tau shrink)
            raw_xy = A_inv([x, y])
            py0, vy0, grav = raw_xy[1], 50, -40
            omega = -np.pi / 40  # angular velocity
            rotated_logo = pygame.transform.rotate(logo, np.rad2deg(omega * tau))
            rotated_rekt = rotated_logo.get_rect(center=A(center))  # we rotate first, then translate
            logo_update = (vy0 * tau) + (0.5 * grav * np.power(tau, 2))
            pts = drop_down(rotate(pts, omega * tau, anchor=center), tau, vy0, grav)

            # Draw in camera reference frame
            cam_y = u * (-vy0 * tau * 2.6)  # shift camera downwards (subtract this value from y-coord of all objects)
            window.blit(too_sharp, A(A_inv([120, 1500]) - np.array([0, cam_y])))
            pts = cam_ref(pts, cam_y)
            pygame.draw.polygon(window, darkgray, A_many(pts), 0)
            pygame.draw.lines(window, white, False, A_many(pts), width=5)
            rotated_rekt.y -= logo_update - cam_y
            window.blit(rotated_logo, rotated_rekt)
            window.blit(text, A(A_inv([100, 100]) - np.array([0, cam_y])))

            # Compute + draw ground.
            ground_y = ground_y_base + cam_y
            ground_rekt = (0, ground_y, width, height - ground_y)
            pygame.draw.rect(surface=window, color=(0, 50, 50), rect=ground_rekt, width=0)

        # Keyframe 3 —— Stick the landing.
        if keyframe == 3:
            side = logo_size[1] * factor
            x, y = (width / 2) - (side / 2) + 20, height / 3  # topleft of the image
            center = A_inv([x + (side / 2.3), y + (side / 2)])
            pts = get_rekt_pts(1, side * 1.5, side * 1.5, *center, du=0.25 / 2)  # more efficient sampling

            # Drop downwards computation
            tau = 10  # Set it all to the final state (cam_y's u = 1 and tau = 10)
            raw_xy = A_inv([x, y])
            py0, vy0, grav = raw_xy[1], 50, -40
            omega = -np.pi / 40  # angular velocity
            rotated_logo = pygame.transform.rotate(logo, np.rad2deg(omega * tau))
            rotated_rekt = rotated_logo.get_rect(center=A(center))  # we rotate first, then translate
            logo_update = (vy0 * tau) + (0.5 * grav * np.power(tau, 2))
            pts = drop_down(rotate(pts, omega * tau, anchor=center), tau, vy0, grav)

            # Draw in camera reference frame
            u = ease_out(min(2 * u, 1))
            cam_y = (-vy0 * tau * 2.6) - (u * 110)  # shift camera downwards (subtract this value from y-coord of all objects)
            pts = cam_ref(pts, cam_y)
            pygame.draw.polygon(window, darkgray, A_many(pts), 0)
            pygame.draw.lines(window, white, False, A_many(pts), width=5)
            rotated_rekt.y -= logo_update - cam_y
            window.blit(rotated_logo, rotated_rekt)
            window.blit(text, A(A_inv([100, 100]) - np.array([0, cam_y])))

            # Compute + draw ground.
            ground_y = ground_y_base + cam_y
            ground_rekt = (0, ground_y, width, height - ground_y)
            pygame.draw.rect(surface=window, color=(0, 50, 50), rect=ground_rekt, width=0)

            window.blit(too_sharp, A(A_inv([120, 1500]) - np.array([0, cam_y])))

            if u > 0.7:
                window.blit(ow, A(A_inv([170, 2200]) - np.array([0, cam_y])))

        # Keyframe 4 —— Reset / undo fall.
        if keyframe == 4:
            side = logo_size[1] * factor
            x, y = (width / 2) - (side / 2) + 20, height / 3  # topleft of the image
            center = A_inv([x + (side / 2.3), y + (side / 2)])
            pts = get_rekt_pts(1, side * 1.5, side * 1.5, *center, du=0.25 / 2)  # more efficient sampling

            # Drop downwards computation
            u = ease_inout(u)
            tau = (10 * (1 - u))  # Set it all to the final state (cam_y's u = 1 and tau = 10)
            raw_xy = A_inv([x, y])
            py0, vy0, grav = raw_xy[1], 50, -40
            omega = -np.pi / 40  # angular velocity
            theta_radians = (1 - u) * omega * tau
            rotated_logo = pygame.transform.rotate(logo, np.rad2deg(theta_radians))
            rotated_rekt = rotated_logo.get_rect(center=A(center))  # we rotate first, then translate
            logo_update = (vy0 * tau) + (0.5 * grav * np.power(tau, 2))
            pts = drop_down(rotate(pts, theta_radians, anchor=center), tau, vy0, grav)

            # Draw in camera reference frame
            cam_y = ((-vy0 * tau * 2.6) - (ease_out(min(2 * 1, 1)) * 110)) * (1 - u)
            pts = cam_ref(pts, cam_y)
            pygame.draw.polygon(window, darkgray, A_many(pts), 0)
            pygame.draw.lines(window, white, False, A_many(pts), width=5)
            rotated_rekt.y -= logo_update - cam_y
            window.blit(rotated_logo, rotated_rekt)
            window.blit(text, A(A_inv([100, 100]) - np.array([0, cam_y])))

            # Compute + draw ground.
            ground_y = ground_y_base + cam_y
            ground_rekt = (0, ground_y, width, height - ground_y)
            pygame.draw.rect(surface=window, color=(0, 50, 50), rect=ground_rekt, width=0)

            window.blit(too_sharp, A(A_inv([120, 1500]) - np.array([0, cam_y])))

            if u > 0.7:
                window.blit(ow, A(A_inv([170, 2200]) - np.array([0, cam_y])))

        # Keyframe 5 —— Pause...
        if keyframe == 5:
            window.blit(text, (100, 100))
            u = ease_inout(u)
            side = logo_size[1] * factor
            x, y = (width / 2) - (side / 2) + 20, height / 3
            center = A_inv([x + (side / 2.3), y + (side / 2)])
            pts = get_rekt_pts(1, side * 1.5, side * 1.5, *center, du=0.25 / 2)
            if len(pts) > 1:
                if len(pts) > 2:
                    pygame.draw.polygon(window, darkgray, A_many(pts), 0)

                pygame.draw.lines(window, white, False, A_many(pts), width=5)

            window.blit(logo, (x, y))

        # Keyframe 6 —— Introduce rounded rect (expanding circles + outline).
        if keyframe == 6:
            window.blit(text, (100, 100))
            side = logo_size[1] * factor
            x, y = (width / 2) - (side / 2) + 20, height / 3  # top-left x,y of LOGO in screen coords (y is down)
            center = A_inv([x + (side / 2.3), y + (side / 2)])  # center of surrounding RECTANGLE in GOOD coords
            rect_sidelen = side * 1.5
            pts = get_rekt_pts(1, rect_sidelen, rect_sidelen, *center, du=0.25 / 2)
            if len(pts) > 1:
                if len(pts) > 2:
                    pygame.draw.polygon(window, darkgray, A_many(pts), 0)

                pygame.draw.lines(window, white, False, A_many(pts), width=5)

            window.blit(logo, (x, y))

            # Draw circles + rounded rect
            radius_keys = [20, 40, 60]
            tau, idx = squash2(u, len(radius_keys))
            tau = ease_inout(tau)
            if idx == 0:
                # Circles
                radius = 1 * radius_keys[idx]
                circle_center = center + (np.array([1, 1]) * rect_sidelen / 2)
                pygame.draw.circle(window, white, A(circle_center), radius, width=3)
                circle_center = center + (np.array([-1, 1]) * rect_sidelen / 2)
                pygame.draw.circle(window, white, A(circle_center), radius, width=3)
                circle_center = center + (np.array([-1, -1]) * rect_sidelen / 2)
                pygame.draw.circle(window, white, A(circle_center), radius, width=3)
                circle_center = center + (np.array([1, -1]) * rect_sidelen / 2)
                pygame.draw.circle(window, white, A(circle_center), radius, width=3)

                # Rounded Rect
                pts = get_rounded_rekt_pts(tau, *center, rect_sidelen, rect_sidelen, corner_radius=radius, du=0.001)
                pygame.draw.lines(window, white, False, A_many(pts), width=5)

            elif idx == 1:
                print('2')
            else:
                print('3')


        # Handle keys pressed
        keys_pressed = pygame.key.get_pressed()
        # do stuff....


        # End run (1. Tick clock, 2. Save the frame, and 3. Detect if window closed)
        pygame.display.flip()
        t += dt
        clock.tick(FPS)
        if save_anim:
            pygame.image.save(window, path_to_save+'/frame'+str(count)+'.png')
            print('Saved frame '+str(count))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()


    # Post game-loop stuff
    # Do more stuff...
    # Use ffmpeg to combine the PNG images into a video
    if save_anim:
        input_files = path_to_save + '/frame%d.png'
        output_file = path_to_save + '/output.mp4'
        ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
        os.system(f'{ffmpeg_path} -r 60 -i {input_files} -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "eq=brightness=0.00:saturation=10.0" {output_file} > /dev/null 2>&1')
        print('Saved video to ' + output_file)


if __name__ == "__main__":
    main()
