import copy
import tkinter
from tkinter import *
import time
import numpy as np
from playsound import playsound


# Window size
window_w = 1720
window_h = 1080
np.set_printoptions(suppress=True)

# Tkinter Setup
root = Tk()
root.title("Simulator")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded
w = tkinter.Canvas(root, width=window_w, height=window_h)
w.pack()


# Coordinate Shift
def A(val):
    return np.array([val[0] + window_w / 2, -val[1] + window_h / 2])


# Camera parameters (rho = distance from origin, phi = angle from world's +z-axis, theta = angle from world's +x-axis)
rho, theta, phi = 1000., -np.pi/2, 0  # Rho is the distance from world origin to near clipping plane
v_rho, v_theta, v_phi = 0, 0, 0
focus = 100000.  # Distance from near clipping plane to eye

# Key handling function
def vanilla_key_pressed(event):
    global rho, theta, phi, focus, w

    m = 30
    drho, dphi, dtheta, dfocus = 10, np.pi / m, np.pi / m, 10
    if event.char == 'a':
        theta -= dtheta

    if event.char == 'd':
        theta += dtheta

    if event.char == 'w':
        phi -= dphi

    if event.char == 's':
        phi += dphi

    if event.char == 'p':
        rho -= drho

    if event.char == 'o':
        rho += drho

    if event.char == 'k':
        focus -= dfocus

    if event.char == 'l':
        focus += dfocus

    if event.char == 'm':
        w.bind("<KeyPress>", speedy_key_pressed)


def speedy_key_pressed(event):
    global v_rho, v_theta, v_phi, focus, w

    max_clicks = 10
    m = 800
    d2rho, d2phi, d2theta, dfocus = 3, np.pi / m, np.pi / m, 10
    if event.char == 'a':
        v_theta = max(v_theta - d2theta, -d2theta * max_clicks)

    if event.char == 'd':
        v_theta = min(v_theta + d2theta, d2theta * max_clicks)

    if event.char == 'w':
        v_phi = max(v_phi - d2phi, -d2phi * max_clicks)

    if event.char == 's':
        v_phi = min(v_phi + d2phi, d2phi * max_clicks)

    if event.char == 'p':
        v_rho = max(v_rho - d2rho, -d2rho * max_clicks / 2)

    if event.char == 'o':
        v_rho = min(v_rho + d2rho, d2rho * max_clicks / 2)

    if event.char == 'k':
        focus -= dfocus

    if event.char == 'l':
        focus += dfocus

    # Change mode
    if event.char == 'm':
        v_rho, v_theta, v_phi = 0, 0, 0
        w.bind("<KeyPress>", vanilla_key_pressed)


# Key binding
w.bind("<KeyPress>", speedy_key_pressed)
w.bind("<1>", lambda event: w.focus_set())
w.pack()


def world_to_plane(v):
    '''
        Converts from point in 3D to its 2D perspective projection, based on location of camera.

        v: vector in R^3 to convert.
    '''
    # Camera params
    global rho, theta, phi, focus

    # Radial distance to eye from world's origin.
    eye_rho = rho + focus

    # Vector math from geometric computation (worked out on white board, check iCloud for possible picture)
    eye_to_origin = -np.array([eye_rho * np.sin(phi) * np.cos(theta),
                               eye_rho * np.sin(phi) * np.sin(theta), eye_rho * np.cos(phi)])

    eye_to_ei = eye_to_origin + v
    origin_to_P = np.array(
        [rho * np.sin(phi) * np.cos(theta), rho * np.sin(phi) * np.sin(theta), rho * np.cos(phi)])

    # Formula for intersecting t: t = (n•(a-b)) / (n•v)
    t = np.dot(eye_to_origin, origin_to_P - v) / np.dot(eye_to_origin, eye_to_ei)
    r_t = v + (t * eye_to_ei)

    # Location of image coords in terms of world coordinates.
    tile_center_world = -origin_to_P + r_t

    # Spherical basis vectors
    theta_hat = np.array([-np.sin(theta), np.cos(theta), 0])
    phi_hat = -np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), -np.sin(phi)])

    # Actual transformed 2D coords
    tile_center = A(np.array([np.dot(tile_center_world, theta_hat), np.dot(tile_center_world, phi_hat)]))

    return tile_center


# # Input: A point in 3D world space. Output: Corresponding point in range [-width/2, width/2]x[-height/2, height/2].
# def world_to_plane(v):
#     global rho, theta, phi, focus
#     # -1. Adjust focus based on
#     # 0. Turn vector into a homogeneous one.
#     v = np.append(v, 1)
#     # 1. Convert vector from world into camera space
#     # a) Get camera basis vectors in terms of world coordinates (x right, y up, z out of page).
#     xhat = np.array([-np.sin(theta), np.cos(theta), 0, 0])
#     yhat = -np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), -np.sin(phi), 0])
#     zhat = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi), 0])
#     # b) Build the 4th column of the matrix
#     cam_pos = np.append((rho * zhat)[:3], 1)
#     # c) Construct the matrix and do the conversion
#     world_to_cam = np.linalg.inv(np.array([xhat, yhat, zhat, cam_pos]).T)
#     v_cam = np.dot(world_to_cam, v)
#     # 2. Convert from camera space to screen space (using similar triangles math)
#     cam_to_screen = np.array([[-focus, 0, 0, 0],
#                               [0, -focus, 0, 0],
#                               [0, 0, -far_clip/(-far_clip+focus), (-far_clip*focus)/(-far_clip+focus)],
#                               [0, 0, 1, 0]])
#     v_screen = np.dot(cam_to_screen, v_cam)
#     v_screen /= v_screen[3]  # division by z
#     return (v_screen[:2] * np.array([1, -1])) + np.array([window_w/2, window_h/2])


# Key binding
w.bind("<1>", lambda event: w.focus_set())
w.pack()

# Cube parameters
side = 140  # technically "half-side"
radius = 25
translation = None
w_angle = 0


# 4D Projections / Rotations
def trivial_proj(centers):
    global side
    for i in range(len(centers)):
        centers[i] = world_to_plane(centers[i][:-1] * side)
    return centers


def stereo_proj(centers):
    global side
    light = 2
    for i in range(len(centers)):
        centers[i] = world_to_plane(side * centers[i][:-1] / (light - centers[i][-1]))
    return centers


def rot4d(centers, angle):
    R = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.cos(angle), -np.sin(angle)],
        [0, 0, np.sin(angle), np.cos(angle)]
    ])
    for i in range(len(centers)):
        centers[i] = np.dot(R, centers[i])

    return centers


# Animation params
t = 0
dt = 0.01
keys = [0,   # Keyframe 0. Start drawing circle.
        3,   # Keyframe 1. Reveal we're in 3D + it's actually a line.
        6,   # Keyframe 2. Reveal it's actually a square
        7,   # Keyframe 3. Casual rotate 1 (pan up again)
        10,  # Keyframe 4. Reveal it's actually a cube (phi down + reveal)
        12,  # Keyframe 5. Casual rotate 2 (pan up yet again)
        14,  # Keyframe 6. Reveal it's actually a hypercube!
        21,  # Keyframe 7. Collapse into tesseract.
        23,  # Keyframe 8. Rotate in 4D!
        35]


# Squash t parameter to keys
def squash(t_, intervals=None):
    global keys
    if intervals is None:
        intervals = keys
    for i in range(len(intervals) - 1):
        if intervals[i] <= t_ < intervals[i + 1]:
            return (t_ - intervals[i]) / (intervals[i + 1] - intervals[i]), i

    return intervals[-1], len(intervals) - 2


def squash2(t_, n_intervals=1):
    intervals = [float(i) / n_intervals for i in range(n_intervals + 1)]
    return squash(t_, intervals)


# Ease functions
def ease_inout(t_):
    return np.power(t_, 2) / (np.power(t_, 2) + np.power(1 - t_, 2))


# From ChatGPT
def rgb_to_hex(rgb):
    return '#' + ''.join(['%02x' % int(val) for val in rgb])


# Parametric shapes
def line(u, start, fin):
    return ((1 - u) * start) + (u * fin)


# Parametric shapes
# NOTE: Given a center in R3, we return the 2D point corresponding to the u value (from 0 to 1).
# Meaning the circle will have the same radius and will always "point towards the screen".
def circle(u, radius=30, center3d=np.array([0, 0, 0])):
    tau = np.pi * 2
    return (radius * np.array([np.cos(tau * u), np.sin(tau * u)])) + world_to_plane(center3d)


def get_circle_pts(u, radius, center3d, du):
    pts = []
    for u_ in np.arange(0, u + du, du):
        pts.extend(circle(u_, radius, center3d))
    return pts


# Show / hide display objects
show_axes = True


# Main function
def run():
    global t, dt, keys, show_axes, rho, phi, theta, focus, side, radius, translation, w_angle

    playsound('close_eyes.MP4', block=False)

    w.configure(background='black')
    while t <= keys[-1]:
        w.delete('all')

        # Camera Velocity Update
        rho += v_rho
        phi += v_phi
        theta += v_theta

        u, frame = squash(t)

        # Horizontal Lines drawing ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        mag = 200
        spacing = 100
        color = np.array([1., 1., 1.]) * 80
        for y in np.arange(-mag, mag+spacing, spacing):
            p0 = world_to_plane(np.array([-mag, y, 0]))
            p1 = world_to_plane(np.array([+mag, y, 0]))
            w.create_line(*p0, *p1, fill=rgb_to_hex(color))

        for x in np.arange(-mag, mag+spacing, spacing):
            p0 = world_to_plane(np.array([x, -mag, 0]))
            p1 = world_to_plane(np.array([x, mag, 0]))
            w.create_line(*p0, *p1, fill=rgb_to_hex(color))

        if show_axes:
            mag = 300
            offset = np.array([0, 0, 0])
            colors = np.array([[150., 0., 0.], [0., 0., 150.], [180., 0., 180.]])
            if frame == 7:
                colors = (1 - u) * colors
            elif frame > 7:
                colors = np.array([[150., 0., 0.], [0., 0., 150.], [180., 0., 180.]]) * 0.0

            direction_vectors = np.array(
                [np.array([mag, 0, 0]) + offset, np.array([0, mag, 0]) + offset, np.array([0, 0, mag]) + offset])
            if show_axes:
                for i, v_i in enumerate(direction_vectors):
                    start = world_to_plane(offset)
                    end = world_to_plane(v_i)
                    fill = rgb_to_hex(colors[i]) if frame <= 7 else ''
                    w.create_line(start[0], start[1], end[0], end[1], fill=fill, width=5)


        # Animation TODO: Change focus in and out by interpolating logarithmically for cooler effect.
        # Keyframe 0 — Draw the circle
        if frame == 0:
            draw_u = ease_inout(min(1.0, u * 4.0))
            u = ease_inout(u)
            rho0, rho1 = 200, 20
            rho = ((1 - u) * rho0) + (u * rho1)
            col = np.array([255, 255, 255]) * draw_u
            # Draw circle
            pts = get_circle_pts(draw_u, radius, np.array([0, 0, side]), du=0.01)
            w.create_polygon(*pts, outline='white', width=1, fill=rgb_to_hex(col))

        # Keyframe 1 — Reveal we're in 3D + it's actually a line.
        elif frame == 1:
            col = np.array([1, 1, 1]) * 255
            center_top = world_to_plane(np.array([0, 0, side]))
            center_bot = world_to_plane(np.array([0, 0, -side]))
            # Draw line
            w.create_line(*center_top, *center_bot, fill='white', width=1)
            # Draw bottom circle + line
            w.create_oval(*(center_bot - radius), *(center_bot + radius), outline='white', width=1, fill=rgb_to_hex(col))
            # Redraw top circle
            w.create_oval(*(center_top - radius), *(center_top + radius), outline='white', width=1, fill=rgb_to_hex(col))
            # Move
            vtheta0, vtheta1 = 0, np.radians(1.2)
            theta += ((1 - u) * vtheta0) + (u * vtheta1)
            # theta0, theta1 = -np.pi/2, np.radians(20)
            # theta = ((1 - u) * theta0) + (u * theta1)
            u = ease_inout(u)
            phi0, phi1 = 0, np.pi/2
            phi = ((1 - u) * phi0) + (u * phi1)

        # Keyframe 2 — Reveal it's actually a square
        elif frame == 2:
            col = np.array([1, 1, 1]) * 255
            u = ease_inout(u)
            vtheta0, vtheta1 = np.radians(1.5), np.radians(0.3)
            theta += ((1 - u) * vtheta0) + (u * vtheta1)
            if theta < np.pi/2:
                # Draw line
                center_top = world_to_plane(np.array([0, 0, side]))
                center_bot = world_to_plane(np.array([0, 0, -side]))
                w.create_line(*center_top, *center_bot, fill='white', width=1)
                # Redraw top circle
                w.create_oval(*(center_top - radius), *(center_top + radius), outline='white', width=1, fill=rgb_to_hex(col))
                # Draw bottom circle + line
                w.create_oval(*(center_bot - radius), *(center_bot + radius), outline='white', width=1, fill=rgb_to_hex(col))
            else:
                # Initialize centers
                centers = [
                    world_to_plane(np.array([0, side, side])),
                    world_to_plane(np.array([0, -side, side])),
                    world_to_plane(np.array([0, -side, -side])),
                    world_to_plane(np.array([0, side, -side]))
                ]
                # Connect 'em with 4 lines to form square
                w.create_line(*centers[0], *centers[1], fill='white', width=1)
                w.create_line(*centers[1], *centers[2], fill='white', width=1)
                w.create_line(*centers[2], *centers[3], fill='white', width=1)
                w.create_line(*centers[3], *centers[0], fill='white', width=1)
                # Draw the four vertices (c.c. order, starting from top-right)
                for c in centers:
                    w.create_oval(*(c - radius), *(c + radius), outline='white', width=1, fill=rgb_to_hex(col))

        # Keyframe 3 — Casual rotate 1 (phi up again)
        elif frame == 3:
            col = np.array([1, 1, 1]) * 255
            u = ease_inout(u)
            theta += np.radians(0.3)
            phi0, phi1 = np.pi/2, np.pi/3
            phi = ((1 - u) * phi0) + (u * phi1)
            # Redraw square
            centers = [
                world_to_plane(np.array([0, side, side])),
                world_to_plane(np.array([0, -side, side])),
                world_to_plane(np.array([0, -side, -side])),
                world_to_plane(np.array([0, side, -side]))
            ]
            # Connect 'em with 4 lines to form square
            w.create_line(*centers[0], *centers[1], fill='white', width=1)
            w.create_line(*centers[1], *centers[2], fill='white', width=1)
            w.create_line(*centers[2], *centers[3], fill='white', width=1)
            w.create_line(*centers[3], *centers[0], fill='white', width=1)
            # Draw the four vertices (c.c. order, starting from top-right)
            for c in centers:
                w.create_oval(*(c - radius), *(c + radius), outline='white', width=1, fill=rgb_to_hex(col))

        # Keyframe 4. Reveal it's actually a cube (phi down + reveal)
        elif frame == 4:
            col = np.array([1, 1, 1]) * 255
            u = ease_inout(u)
            vtheta0, vtheta1 = np.radians(0.4), np.radians(0.5)
            theta += ((1 - u) * vtheta0) + (u * vtheta1)
            vphi0, vphi1 = 0., np.radians(0.4)
            phi = min(phi + ((1 - u) * vphi0) + (u * vphi1), np.pi/2)

            if theta < np.radians(360):
                # Redraw square
                centers = [
                    world_to_plane(np.array([0, side, side])),
                    world_to_plane(np.array([0, -side, side])),
                    world_to_plane(np.array([0, -side, -side])),
                    world_to_plane(np.array([0, side, -side]))
                ]
                # Connect 'em with 4 lines to form square
                w.create_line(*centers[0], *centers[1], fill='white', width=1)
                w.create_line(*centers[1], *centers[2], fill='white', width=1)
                w.create_line(*centers[2], *centers[3], fill='white', width=1)
                w.create_line(*centers[3], *centers[0], fill='white', width=1)
                # Draw the four vertices (c.c. order, starting from top-right)
                for c in centers:
                    w.create_oval(*(c - radius), *(c + radius), outline='white', width=1, fill=rgb_to_hex(col))
            else:
                # Redraw cube
                centers = [
                    world_to_plane(np.array([side, side, side])),
                    world_to_plane(np.array([side, -side, side])),
                    world_to_plane(np.array([side, -side, -side])),
                    world_to_plane(np.array([side, side, -side])),

                    world_to_plane(np.array([-side, side, side])),
                    world_to_plane(np.array([-side, -side, side])),
                    world_to_plane(np.array([-side, -side, -side])),
                    world_to_plane(np.array([-side, side, -side]))
                ]
                # Connect 'em with 12 lines to form square
                w.create_line(*centers[0], *centers[1], fill='white', width=1)
                w.create_line(*centers[1], *centers[2], fill='white', width=1)
                w.create_line(*centers[2], *centers[3], fill='white', width=1)
                w.create_line(*centers[3], *centers[0], fill='white', width=1)

                w.create_line(*centers[4], *centers[5], fill='white', width=1)
                w.create_line(*centers[5], *centers[6], fill='white', width=1)
                w.create_line(*centers[6], *centers[7], fill='white', width=1)
                w.create_line(*centers[7], *centers[4], fill='white', width=1)

                w.create_line(*centers[0], *centers[4], fill='white', width=1)
                w.create_line(*centers[1], *centers[5], fill='white', width=1)
                w.create_line(*centers[2], *centers[6], fill='white', width=1)
                w.create_line(*centers[3], *centers[7], fill='white', width=1)
                # Draw the four vertices (c.c. order, starting from top-right)
                for c in centers:
                    w.create_oval(*(c - radius), *(c + radius), outline='white', width=1, fill=rgb_to_hex(col))

        # Keyframe 5. Pan up casually.
        elif frame == 5:
            col = np.array([1, 1, 1]) * 255
            u = ease_inout(u)
            phi = ((1 - u) * (np.pi / 2)) + (u * np.radians(70))
            vtheta0, vtheta1 = np.radians(0.5), np.radians(0.2)
            theta += ((1 - u) * vtheta0) + (u * vtheta1)
            # Redraw cube
            centers = [
                world_to_plane(np.array([-side, side, side])),
                world_to_plane(np.array([-side, -side, side])),
                world_to_plane(np.array([-side, -side, -side])),
                world_to_plane(np.array([-side, side, -side])),

                world_to_plane(np.array([side, side, side])),
                world_to_plane(np.array([side, -side, side])),
                world_to_plane(np.array([side, -side, -side])),
                world_to_plane(np.array([side, side, -side]))
            ]
            # Connect 'em with 12 lines to form square
            w.create_line(*centers[0], *centers[1], fill='white', width=1)
            w.create_line(*centers[1], *centers[2], fill='white', width=1)
            w.create_line(*centers[2], *centers[3], fill='white', width=1)
            w.create_line(*centers[3], *centers[0], fill='white', width=1)

            w.create_line(*centers[4], *centers[5], fill='white', width=1)
            w.create_line(*centers[5], *centers[6], fill='white', width=1)
            w.create_line(*centers[6], *centers[7], fill='white', width=1)
            w.create_line(*centers[7], *centers[4], fill='white', width=1)

            w.create_line(*centers[0], *centers[4], fill='white', width=1)
            w.create_line(*centers[1], *centers[5], fill='white', width=1)
            w.create_line(*centers[2], *centers[6], fill='white', width=1)
            w.create_line(*centers[3], *centers[7], fill='white', width=1)
            # Draw the four vertices (c.c. order, starting from top-right)
            for c in centers:
                w.create_oval(*(c - radius), *(c + radius), outline='white', width=1, fill=rgb_to_hex(col))

        # Keyframe 6. Reveal it's a hypercube!
        elif frame == 6:
            col = np.array([1, 1, 1]) * 255
            f0, f1 = 100000., 1234.
            focus = f0 * np.power(f1 / f0, min(1.0, 2.0 * u))
            u = ease_inout(u)
            vtheta0, vtheta1 = np.radians(0.2), np.radians(0.1)
            theta += ((1 - u) * vtheta0) + (u * vtheta1)
            # rho0, rho1 = 200, 600
            # rho = ((1 - u) * rho0) + (u * rho1)
            r = ((1 - u) * radius) + (u * radius * 0.8)

            if u < 0.02:
                # Redraw cube
                centers = [
                    world_to_plane(np.array([-side, side, side])),
                    world_to_plane(np.array([-side, -side, side])),
                    world_to_plane(np.array([-side, -side, -side])),
                    world_to_plane(np.array([-side, side, -side])),

                    world_to_plane(np.array([side, side, side])),
                    world_to_plane(np.array([side, -side, side])),
                    world_to_plane(np.array([side, -side, -side])),
                    world_to_plane(np.array([side, side, -side]))
                ]
                # Connect 'em with 12 lines to form square
                w.create_line(*centers[0], *centers[1], fill='white', width=1)
                w.create_line(*centers[1], *centers[2], fill='white', width=1)
                w.create_line(*centers[2], *centers[3], fill='white', width=1)
                w.create_line(*centers[3], *centers[0], fill='white', width=1)

                w.create_line(*centers[4], *centers[5], fill='white', width=1)
                w.create_line(*centers[5], *centers[6], fill='white', width=1)
                w.create_line(*centers[6], *centers[7], fill='white', width=1)
                w.create_line(*centers[7], *centers[4], fill='white', width=1)

                w.create_line(*centers[0], *centers[4], fill='white', width=1)
                w.create_line(*centers[1], *centers[5], fill='white', width=1)
                w.create_line(*centers[2], *centers[6], fill='white', width=1)
                w.create_line(*centers[3], *centers[7], fill='white', width=1)

                # Draw the four vertices (c.c. order, starting from top-right)
                for i, c in enumerate(centers):
                    w.create_oval(*(c - r), *(c + r), outline='white', width=1, fill=rgb_to_hex(col))
            else:
                if translation is None:
                    # Radial distance to eye from world's origin.
                    eye_rho = rho + focus
                    # Vector math from geometric computation (worked out on white board, check iCloud for possible picture)
                    eye_to_origin = -np.array([eye_rho * np.sin(phi) * np.cos(theta),
                                               eye_rho * np.sin(phi) * np.sin(theta), eye_rho * np.cos(phi)])
                    translation = (eye_to_origin / np.linalg.norm(eye_to_origin)) * side * 3

                # Redraw cube
                centers = [
                    world_to_plane(np.array([-side, side, side])),
                    world_to_plane(np.array([-side, -side, side])),
                    world_to_plane(np.array([-side, -side, -side])),
                    world_to_plane(np.array([-side, side, -side])),

                    world_to_plane(np.array([side, side, side])),
                    world_to_plane(np.array([side, -side, side])),
                    world_to_plane(np.array([side, -side, -side])),
                    world_to_plane(np.array([side, side, -side])),

                    world_to_plane(np.array([-side, side, side]) + translation),
                    world_to_plane(np.array([-side, -side, side]) + translation),
                    world_to_plane(np.array([-side, -side, -side]) + translation),
                    world_to_plane(np.array([-side, side, -side]) + translation),

                    world_to_plane(np.array([side, side, side]) + translation),
                    world_to_plane(np.array([side, -side, side]) + translation),
                    world_to_plane(np.array([side, -side, -side]) + translation),
                    world_to_plane(np.array([side, side, -side]) + translation)
                ]
                # Connect 'em with 36 lines to form square
                w.create_line(*centers[0], *centers[1], fill='white', width=1)
                w.create_line(*centers[1], *centers[2], fill='white', width=1)
                w.create_line(*centers[2], *centers[3], fill='white', width=1)
                w.create_line(*centers[3], *centers[0], fill='white', width=1)
                w.create_line(*centers[4], *centers[5], fill='white', width=1)
                w.create_line(*centers[5], *centers[6], fill='white', width=1)
                w.create_line(*centers[6], *centers[7], fill='white', width=1)
                w.create_line(*centers[7], *centers[4], fill='white', width=1)
                w.create_line(*centers[0], *centers[4], fill='white', width=1)
                w.create_line(*centers[1], *centers[5], fill='white', width=1)
                w.create_line(*centers[2], *centers[6], fill='white', width=1)
                w.create_line(*centers[3], *centers[7], fill='white', width=1)

                w.create_line(*centers[8], *centers[9], fill='white', width=1)
                w.create_line(*centers[9], *centers[10], fill='white', width=1)
                w.create_line(*centers[10], *centers[11], fill='white', width=1)
                w.create_line(*centers[11], *centers[8], fill='white', width=1)
                w.create_line(*centers[12], *centers[13], fill='white', width=1)
                w.create_line(*centers[13], *centers[14], fill='white', width=1)
                w.create_line(*centers[14], *centers[15], fill='white', width=1)
                w.create_line(*centers[15], *centers[12], fill='white', width=1)
                w.create_line(*centers[8], *centers[12], fill='white', width=1)
                w.create_line(*centers[9], *centers[13], fill='white', width=1)
                w.create_line(*centers[10], *centers[14], fill='white', width=1)
                w.create_line(*centers[11], *centers[15], fill='white', width=1)

                # cross-edges
                for i in range(8):
                    w.create_line(*centers[i], *centers[i + 8], fill='orange', width=1)

                # Draw the four vertices (c.c. order, starting from top-right)
                for c in centers:
                    w.create_oval(*(c - radius), *(c + radius), outline='white', width=1, fill=rgb_to_hex(col))

        # Keyframe 7. Collapse into tesseract.
        elif frame == 7:
            u = ease_inout(u)
            theta += np.radians(0.1)

            # Redraw cube
            centers0 = [
                world_to_plane(np.array([-side, side, side])),
                world_to_plane(np.array([-side, -side, side])),
                world_to_plane(np.array([-side, -side, -side])),
                world_to_plane(np.array([-side, side, -side])),

                world_to_plane(np.array([side, side, side])),
                world_to_plane(np.array([side, -side, side])),
                world_to_plane(np.array([side, -side, -side])),
                world_to_plane(np.array([side, side, -side])),

                world_to_plane(np.array([-side, side, side]) + translation),
                world_to_plane(np.array([-side, -side, side]) + translation),
                world_to_plane(np.array([-side, -side, -side]) + translation),
                world_to_plane(np.array([-side, side, -side]) + translation),

                world_to_plane(np.array([side, side, side]) + translation),
                world_to_plane(np.array([side, -side, side]) + translation),
                world_to_plane(np.array([side, -side, -side]) + translation),
                world_to_plane(np.array([side, side, -side]) + translation)
            ]

            centers1 = [
                np.array([-1, 1, 1, 1]),
                np.array([-1, -1, 1, 1]),
                np.array([-1, -1, -1, 1]),
                np.array([-1, 1, -1, 1]),

                np.array([1, 1, 1, 1]),
                np.array([1, -1, 1, 1]),
                np.array([1, -1, -1, 1]),
                np.array([1, 1, -1, 1]),

                np.array([-1, 1, 1, -1]),
                np.array([-1, -1, 1, -1]),
                np.array([-1, -1, -1, -1]),
                np.array([-1, 1, -1, -1]),

                np.array([1, 1, 1, -1]),
                np.array([1, -1, 1, -1]),
                np.array([1, -1, -1, -1]),
                np.array([1, 1, -1, -1])
            ]

            centers1 = rot4d(centers1, w_angle)
            centers1 = stereo_proj(centers1)

            # LERP between two positions
            centers = ((1 - u) * np.array(centers0)) + (u * np.array(centers1))

            # Connect 'em with 36 lines to form square
            w.create_line(*centers[0], *centers[1], fill='white', width=1)
            w.create_line(*centers[1], *centers[2], fill='white', width=1)
            w.create_line(*centers[2], *centers[3], fill='white', width=1)
            w.create_line(*centers[3], *centers[0], fill='white', width=1)
            w.create_line(*centers[4], *centers[5], fill='white', width=1)
            w.create_line(*centers[5], *centers[6], fill='white', width=1)
            w.create_line(*centers[6], *centers[7], fill='white', width=1)
            w.create_line(*centers[7], *centers[4], fill='white', width=1)
            w.create_line(*centers[0], *centers[4], fill='white', width=1)
            w.create_line(*centers[1], *centers[5], fill='white', width=1)
            w.create_line(*centers[2], *centers[6], fill='white', width=1)
            w.create_line(*centers[3], *centers[7], fill='white', width=1)

            w.create_line(*centers[8], *centers[9], fill='white', width=1)
            w.create_line(*centers[9], *centers[10], fill='white', width=1)
            w.create_line(*centers[10], *centers[11], fill='white', width=1)
            w.create_line(*centers[11], *centers[8], fill='white', width=1)
            w.create_line(*centers[12], *centers[13], fill='white', width=1)
            w.create_line(*centers[13], *centers[14], fill='white', width=1)
            w.create_line(*centers[14], *centers[15], fill='white', width=1)
            w.create_line(*centers[15], *centers[12], fill='white', width=1)
            w.create_line(*centers[8], *centers[12], fill='white', width=1)
            w.create_line(*centers[9], *centers[13], fill='white', width=1)
            w.create_line(*centers[10], *centers[14], fill='white', width=1)
            w.create_line(*centers[11], *centers[15], fill='white', width=1)

            # cross-edges
            for i in range(8):
                w.create_line(*centers[i], *centers[i + 8], fill='orange', width=1)

            # Draw the four vertices (c.c. order, starting from top-right)
            r = ((1 - u) * radius) + (u * (radius / 3))
            for c in centers:
                w.create_oval(*(c - r), *(c + r), outline='white', width=1, fill='white')

        # Keyframe 8. Rotate in 4D!
        elif frame == 8:
            theta += np.radians(0.1)
            centers = [
                np.array([-1, 1, 1, 1]),
                np.array([-1, -1, 1, 1]),
                np.array([-1, -1, -1, 1]),
                np.array([-1, 1, -1, 1]),

                np.array([1, 1, 1, 1]),
                np.array([1, -1, 1, 1]),
                np.array([1, -1, -1, 1]),
                np.array([1, 1, -1, 1]),

                np.array([-1, 1, 1, -1]),
                np.array([-1, -1, 1, -1]),
                np.array([-1, -1, -1, -1]),
                np.array([-1, 1, -1, -1]),

                np.array([1, 1, 1, -1]),
                np.array([1, -1, 1, -1]),
                np.array([1, -1, -1, -1]),
                np.array([1, 1, -1, -1])
            ]

            u = ease_inout(min(3.0 * u, 1.0))
            angle_vel0, angle_vel1 = 0, np.radians(0.8)
            w_angle += ((1 - u) * angle_vel0) + (u * angle_vel1)
            centers = rot4d(centers, w_angle)
            centers = stereo_proj(centers)

            vtheta0, vtheta1 = np.radians(0.1), np.radians(0.3)
            theta += ((1 - u) * vtheta0) + (u * vtheta1)

            # Connect 'em with 36 lines to form square
            w.create_line(*centers[0], *centers[1], fill='white', width=1)
            w.create_line(*centers[1], *centers[2], fill='white', width=1)
            w.create_line(*centers[2], *centers[3], fill='white', width=1)
            w.create_line(*centers[3], *centers[0], fill='white', width=1)
            w.create_line(*centers[4], *centers[5], fill='white', width=1)
            w.create_line(*centers[5], *centers[6], fill='white', width=1)
            w.create_line(*centers[6], *centers[7], fill='white', width=1)
            w.create_line(*centers[7], *centers[4], fill='white', width=1)
            w.create_line(*centers[0], *centers[4], fill='white', width=1)
            w.create_line(*centers[1], *centers[5], fill='white', width=1)
            w.create_line(*centers[2], *centers[6], fill='white', width=1)
            w.create_line(*centers[3], *centers[7], fill='white', width=1)

            w.create_line(*centers[8], *centers[9], fill='white', width=1)
            w.create_line(*centers[9], *centers[10], fill='white', width=1)
            w.create_line(*centers[10], *centers[11], fill='white', width=1)
            w.create_line(*centers[11], *centers[8], fill='white', width=1)
            w.create_line(*centers[12], *centers[13], fill='white', width=1)
            w.create_line(*centers[13], *centers[14], fill='white', width=1)
            w.create_line(*centers[14], *centers[15], fill='white', width=1)
            w.create_line(*centers[15], *centers[12], fill='white', width=1)
            w.create_line(*centers[8], *centers[12], fill='white', width=1)
            w.create_line(*centers[9], *centers[13], fill='white', width=1)
            w.create_line(*centers[10], *centers[14], fill='white', width=1)
            w.create_line(*centers[11], *centers[15], fill='white', width=1)

            # cross-edges
            for i in range(8):
                w.create_line(*centers[i], *centers[i + 8], fill='orange', width=1)

            # Draw the four vertices (c.c. order, starting from top-right)
            for c in centers:
                w.create_oval(*(c - radius/3), *(c + radius/3), outline='white', width=1, fill='white')









        # End run
        t += dt
        w.update()
        time.sleep(0.001)


# Main function
if __name__ == '__main__':
    run()

# Necessary line for Tkinter
mainloop()
