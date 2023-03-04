import copy
import tkinter
from tkinter import *
import time
import numpy as np

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
def A(v):
    return np.array([v[0] + window_w/2, -v[1] + window_h/2])

# Key handling function
def vanilla_key_pressed(event):
    global rho, theta, phi, focus, w

    m = 30
    drho, dphi, dtheta, dfocus = 10, np.pi/m, np.pi/m, 10
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
        v_theta = max(v_theta - d2theta, -d2theta*max_clicks)

    if event.char == 'd':
        v_theta = min(v_theta + d2theta, d2theta*max_clicks)

    if event.char == 'w':
        v_phi = max(v_phi - d2phi, -d2phi*max_clicks)

    if event.char == 's':
        v_phi = min(v_phi + d2phi, d2phi*max_clicks)

    if event.char == 'p':
        v_rho = max(v_rho - d2rho, -d2rho*max_clicks/2)

    if event.char == 'o':
        v_rho = min(v_rho + d2rho, d2rho*max_clicks/2)

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

# ye parameters (rho = distance from origin, phi = angle from world's +z-axis, theta = angle from world's +x-axis)
rho, theta, phi = 1000., (np.pi/3) + np.pi/6, np.pi/3  # These provide location of the eye
v_rho, v_theta, v_phi = 0, 0, 0
focus = 1000000.  # Distance from eye to near clipping plane, i.e. the screen.
far_clip = rho * 3  # Distance from eye to far clipping plane
# assert far_clip > focus

# offset = np.array([0, 0, 0])


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


# Input: A point in 3D world space. Output: Corresponding point in range [-width/2, width/2]x[-height/2, height/2].
# def world_to_plane(v):
#     global rho, theta, phi, focus, offset
#     # -1. Adjust focus based on
#     # 0. Turn vector into a homogeneous one.
#     v = np.append(v, 1)
#     # 1. Convert vector from world into camera space
#     # a) Get camera basis vectors in terms of world coordinates (x right, y up, z out of page).
#     xhat = np.array([-np.sin(theta), np.cos(theta), 0, 0])
#     yhat = -np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), -np.sin(phi), 0])
#     zhat = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi), 0])
#     # b) Build the 4th column of the matrix
#     cam_pos = np.append((rho * zhat)[:3] + offset, 1)
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


# 1000 IQ Genius Function
def list_world_to_plane(l):
    new_l = []
    for v in l:
        new_l.extend(world_to_plane(v))
    return new_l


def draw_cube(sidelen=400, vertex_radius=5):
    # Vertices in World Cartesian Coordinates
    l = sidelen / 2
    v_0 = np.array([l, -l, l])
    v_1 = np.array([l, l, l])
    v_2 = np.array([-l, l, l])
    v_3 = np.array([-l, -l, l])
    v_4 = np.array([l, -l, -l])
    v_5 = np.array([l, l, -l])
    v_6 = np.array([-l, l, -l])
    v_7 = np.array([-l, -l, -l])
    vertices = [v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7]
    # Vertex radius
    vr = vertex_radius
    # Edges (each e_i is the set of vertices the ith vertex is connected to (unmarked))
    e_0 = [v_1, v_3, v_4]
    e_1 = [v_2, v_5]
    e_2 = [v_3, v_6]
    e_3 = [v_7]
    e_4 = [v_7, v_5]
    e_5 = [v_6]
    e_6 = [v_7]
    edges_from = [e_0, e_1, e_2, e_3, e_4, e_5, e_6, []]
    # Draw cube
    for i, v in enumerate(vertices):
        v = world_to_plane(v)
        edges_from_i = [world_to_plane(v_j) for v_j in edges_from[i]]
        if show_cube:
            w.create_oval(v[0] - (vr / 2), v[1] - (vr / 2), v[0] + (vr / 2), v[1] + (vr / 2), outline='white',
                          fill='white', tag='v_' + str(i))

            for j, v_j in enumerate(edges_from_i):
                w.create_line(v[0], v[1], v_j[0], v_j[1], fill='orange', tag='edge_' + str(i) + str(j))


# Animation params
t = 0.4999
dt = 0.001
keys = [0,          # 1. Start
        0.4,        # 2. Initial 3d rotate
        0.5,        # 3. Switch to top view
        0.7,        # 4. Draw parabola
        0.9,        # 5. Switch out of top view to 3D
        1.3,
        1.7,
        2.0,
        2.3]


# Get keyframe number
def frame(t_):
    global keys
    for i in range(len(keys)-1):
        if keys[i] <= t_ < keys[i+1]:
            return i


# Squash t parameter to keys
def squash(t_):
    global keys
    for i in range(len(keys)-1):
        if keys[i] <= t_ < keys[i+1]:
            return (t_ - keys[i]) / (keys[i+1] - keys[i])


# Ease functions
def ease_in(t_):
    return np.power(t_, 2) / (np.power(t_, 2) + np.power(1 - t_, 2))


def ease_out(t_):
    return np.power(t_ - 1, 2) / (np.power(t_, 2) + np.power(t_ - 1, 2))


# Parabola
def parabola(x):
    return (np.power(x, 2.0) / 100) - 300


# Parabola points thus far
points = []
points2 = []
points3 = []


# Show / hide display objects
show_axes = True
show_cube = False


def rgb_to_hex(rgb):
    return '#' + ''.join(['%02x' % int(val) for val in rgb])


k = 0
# Main function
def run():
    global rho, phi, theta, focus, v_rho, v_theta, v_phi, vector, t, show_axes, show_cube, show_floor, dt, points

    w.configure(background='black')

    while t <= keys[-2]:
        w.delete('all')
        # Camera Velocity Update
        rho += v_rho
        phi += v_phi
        theta += v_theta

        # 3D Axes Drawing ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        colors = np.array([[255., 0., 0.], [0., 0., 255.], [128., 0., 128.]])
        # if frame(t) == 6:
        #     for col in colors:
        #         col *= ease_out(squash(t))
        #
        # elif frame(t) == 7:
        #     cols0 = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        #     cols1 = np.array([[255., 0., 0.], [0., 0., 255.], [128., 0., 128.]])
        #     t_ = ease_in(squash(t))
        #     colors = ((1 - t_) * cols0) + (t_ * cols1)

        mag = 200
        offset = np.array([0, -0, 0])
        direction_vectors = np.array([np.array([mag, 0, 0]) + offset, np.array([0, mag, 0]) + offset, np.array([0, 0, mag]) + offset])
        if show_axes:
            for i, v_i in enumerate(direction_vectors):
                start = world_to_plane(offset)
                end = world_to_plane(v_i)
                w.create_line(start[0], start[1], end[0], end[1], fill=rgb_to_hex(colors[i]), width=5)

        # Horizontal Lines drawing ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        mag = 500
        spacing = 100
        color = np.array([1., 1., 1.]) * 100
        for y in np.arange(-mag, mag+spacing, spacing):
            p0 = world_to_plane(np.array([-mag, y, 0]))
            p1 = world_to_plane(np.array([+mag, y, 0]))
            w.create_line(*p0, *p1, fill=rgb_to_hex(color))

        for x in np.arange(-mag, mag+spacing, spacing):
            p0 = world_to_plane(np.array([x, -mag, 0]))
            p1 = world_to_plane(np.array([x, mag, 0]))
            w.create_line(*p0, *p1, fill=rgb_to_hex(color))

        # Cube Drawing –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        sidelen, vertex_radius = 400, 5
        if show_cube:
            draw_cube(sidelen, vertex_radius)

        # Animation –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # 1. Initial 3D Rotation
        t_ = squash(t)
        z = 0
        if frame(t) == 0:
            theta0, theta1 = 0, (np.pi/3) + np.pi/6
            theta = ((1 - t_) * theta0) + (t_ * theta1)

        elif frame(t) == 1:
            theta0, theta1 = (np.pi/3) + np.pi/6, -np.pi / 2
            phi0, phi1 = np.pi/3, 0
            t_ = ease_in(t_)
            theta = ((1 - t_) * theta0) + (t_ * theta1)
            phi = ((1 - t_) * phi0) + (t_ * phi1)


        # Animation actually starts here
        elif frame(t) == 2:
            rho0, rho1 = 1000., 550
            rho = ((1 - t_) * rho0) + (t_ * rho1)
            t_ = ((ease_in(t_) * 2) - 1) * 1500
            pt = np.array([t_, parabola(t_), z])
            points.append(pt)
            if len(points) > 4:
                w.create_line(*list_world_to_plane(points), fill='green', width=6)

        elif frame(t) == 3:
            w.create_line(*list_world_to_plane(points), fill='green', width=6)
            theta0, theta1 = -np.pi / 2, np.pi/150
            phi0, phi1 = 0, np.pi/6
            t_ = ease_in(t_)
            theta = ((1 - t_) * theta0) + (t_ * theta1)
            phi = ((1 - t_) * phi0) + (t_ * phi1)

        elif frame(t) == 4:
            w.create_line(*list_world_to_plane(points), fill='green', width=6)
            rho0, rho1 = 550., 1000
            theta0, theta1 = np.pi/150, -np.pi/2
            phi0, phi1 = np.pi/6, np.pi / 2.05
            t_ = ease_in(t_)
            theta = ((1 - t_) * theta0) + (t_ * theta1)
            phi = ((1 - t_) * phi0) + (t_ * phi1)
            rho = ((1 - t_) * rho0) + (t_ * rho1)

        elif frame(t) == 5:
            w.create_line(*list_world_to_plane(points), fill='green', width=6)
            focus0, focus1 = 1000000, 1000
            t_ = ease_in(t_)
            focus = ((1 - t_) * focus0) + (t_ * focus1)

        if frame(t) == 6:
            w.create_line(*list_world_to_plane(points), fill='green', width=6)
            t_0 = -700
            t_1 = t_0 - 20000
            t_2 = ((1 - t_) * t_0) + (t_ * t_1)
            pt = np.array([t_2, parabola(t_2), z])
            points2.append(pt)
            if len(points2) > 4:
                w.create_line(*list_world_to_plane(points2), fill='green', width=6)

            t_0 = 700
            t_1 = t_0 + 20000
            t_2 = ((1 - t_) * t_0) + (t_ * t_1)
            pt = np.array([t_2, parabola(t_2), z])
            points3.append(pt)
            if len(points3) > 4:
                w.create_line(*list_world_to_plane(points3), fill='green', width=6)

        if frame(t) == 7:
            w.create_line(*list_world_to_plane(points), fill='green', width=6)
            w.create_line(*list_world_to_plane(points2), fill='green', width=6)
            w.create_line(*list_world_to_plane(points3), fill='green', width=6)
            rho0, rho1 = 1000., 1000
            phi0, phi1 = np.pi / 2.05, 0
            t_ = ease_in(t_)
            phi = ((1 - t_) * phi0) + (t_ * phi1)
            rho = ((1 - t_) * rho0) + (t_ * rho1)

        # elif frame(t) == 5:
        #     w.create_line(*list_world_to_plane(points), fill='green', width=6)
        #     w.create_line(*list_world_to_plane(points2), fill='green', width=6)
        #     w.create_line(*list_world_to_plane(points3), fill='green', width=6)

        # End run
        t += dt

        w.update()
        time.sleep(0.001)


# From https://stackoverflow.com/questions/51591456/can-i-use-rgb-in-tkinter
def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb


# Main function
if __name__ == '__main__':
    run()

# Necessary line for Tkinter
mainloop()
