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
def A(val):
    return np.array([val[0] + window_w/2, -val[1] + window_h/2])


# Camera parameters (rho = distance from origin, phi = angle from world's +z-axis, theta = angle from world's +x-axis)
rho, theta, phi = -200., -np.pi/2, np.pi/2  # Rho is the distance from world origin to near clipping plane
v_rho, v_theta, v_phi = 0, 0, 0
focus = 1230.  # Distance from near clipping plane to eye
# -431.0 1230.0

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


# Animation params
t = 0
dt = 0.01
keys = [0,       # Keyframe 0. Begin drawing box.
        3,       # Keyframe 1. Begin drawing spheres.
        4,       # Keyframe 2. Zoom out a bit and scan through the pixels.
        8,       # Keyframe 3. Pause.
        9,       # Keyframe 4. Switch to top-view
        12]


# Get keyframe number
# def frame(t_, interval=None):
#     global keys
#     if interval is None:
#         interval = keys
#     for i in range(len(interval)-1):
#         if interval[i] <= t_ < interval[i+1]:
#             return i


# Squash t parameter to keys
def squash(t_, intervals=None):
    global keys
    if intervals is None:
        intervals = keys
    for i in range(len(intervals)-1):
        if intervals[i] <= t_ < intervals[i+1]:
            return (t_ - intervals[i]) / (intervals[i+1] - intervals[i]), i

    return intervals[-1], len(intervals)-2


def squash2(t_, n_intervals=1):
    intervals = [float(i) / n_intervals for i in range(n_intervals+1)]
    return squash(t_, intervals)


# Ease functions
def ease_in(t_):
    return np.power(t_, 2) / (np.power(t_, 2) + np.power(1 - t_, 2))


# From ChatGPT
def rgb_to_hex(rgb):
    return '#' + ''.join(['%02x' % int(val) for val in rgb])


# Parametric shapes
def line(u, start, fin):
    return ((1 - u) * start) + (u * fin)


# Parametric shapes
# NOTE: Given a center in R3, we return the 2D point corresponding to the tau value.
# Meaning the circle will have the same radius and will always "point towards the screen".
def circle(u, radius=30, center3d=np.array([0, 0, 0])):
    tau = np.pi * 2
    return (radius * np.array([np.cos(tau * u), np.sin(tau * u)])) + world_to_plane(center3d)


def get_circle_pts(u, radius, center3d, du):
    pts = []
    for u_ in np.arange(0, u+du, du):
        pts.extend(circle(u_, radius, center3d))
    return pts


# Show / hide display objects
show_axes = False

# Scene data. It's basically a Cornell Box with 2 spheres in it.
du = 0.01
y_far = 250.  # y-value of the far back wall
y_close = (rho * 0.25) - 250.
room_width, room_height = 500., 600.
# Wall points
left_wall_points = [
    np.array([-room_width/2, y_close, -room_height/2]), np.array([-room_width/2, y_far, -room_height/2]),
    np.array([-room_width/2, y_far, room_height/2]), np.array([-room_width/2, y_close, room_height/2])
]
right_wall_points = [
    np.array([room_width/2, y_close, -room_height/2]), np.array([room_width/2, y_far, -room_height/2]),
    np.array([room_width/2, y_far, room_height/2]), np.array([room_width/2, y_close, room_height/2])
]
back_wall_points = [
    np.array([-room_width/2, y_far, -room_height/2]), np.array([room_width/2, y_far, -room_height/2]),
    np.array([room_width/2, y_far, room_height/2]), np.array([-room_width/2, y_far, room_height/2])
]
ceiling_points = [
    np.array([-room_width/2, y_far, room_height/2]), np.array([room_width/2, y_far, room_height/2]),
    np.array([room_width/2, y_close, room_height/2]), np.array([-room_width/2, y_close, room_height/2])
]
floor_points = [
    np.array([-room_width / 2, y_far, -room_height / 2]), np.array([room_width / 2, y_far, -room_height / 2]),
    np.array([room_width / 2, y_close, -room_height / 2]), np.array([-room_width / 2, y_close, -room_height / 2])
]
wall_points = [back_wall_points, ceiling_points, floor_points, left_wall_points, right_wall_points]
wall_cols = [np.array([100, 100, 100]), np.array([255, 255, 255]), np.array([255, 255, 255]),
             np.array([255, 0, 0]), np.array([0, 255, 0])]

# Sphere data
radius = 110
sphere_col = np.array([0, 0, 1]) * 60
z = (-room_height/2)+radius
x1 = (-room_width/2) + (0.2 * room_width)
y1 = (0.8 * y_far) + (0.2 * y_close)
center1 = np.array([x1, y1, z])
x2 = (room_width/2) - (0.25 * room_width)
y2 = ((1-0.8) * y_far) + (0.8 * y_close)
center2 = np.array([x2, y2, z])

# Area light
room_width /= 3
y_far /= 4
y_close /= 3
light_points = [
    np.array([-room_width/2, y_far, room_height/2]), np.array([room_width/2, y_far, room_height/2]),
    np.array([room_width/2, y_close, room_height/2]), np.array([-room_width/2, y_close, room_height/2])
]
room_width *= 3
y_far *= 3
y_close *= 3


# Redrawing methods
def redraw_box():
    global wall_points, wall_cols
    for i in range(5):
        wall = wall_points[i]
        border_col = rgb_to_hex(wall_cols[i])
        fill_col = rgb_to_hex(wall_cols[i] * 0.1)
        w.create_polygon(*world_to_plane(wall[0]), *world_to_plane(wall[1]),
                         *world_to_plane(wall[2]), *world_to_plane(wall[3]),
                         fill=fill_col)
        for j in range(4):
            p0, p1 = wall[j], wall[(j + 1) % 4]
            w.create_line(*world_to_plane(p0), *world_to_plane(p1), fill=border_col, width=5)

        fill_col = np.array([100, 100, 1])
        w.create_polygon(*world_to_plane(light_points[0]), *world_to_plane(light_points[1]),
                         *world_to_plane(light_points[2]), *world_to_plane(light_points[3]),
                         fill=rgb_to_hex(fill_col), outline='', width=5)


def redraw_spheres():
    global center1, center2, radius, du, sphere_col
    r2d = np.array([radius, radius])
    for i, c in enumerate([center1, center2]):
        center = world_to_plane(c)
        w.create_oval(*(center - r2d), *(center + r2d), outline='white', width=1, fill=rgb_to_hex(sphere_col))


def redraw_grid(y=-1000, u=0):
    global room_width, room_height
    col = np.array([1, 1, 1]) * 100
    spacing = 10
    halfwidth = room_width / 10
    halfheight = room_height / 10
    for x in np.arange(-halfwidth, halfwidth + spacing, spacing):
        stroke = 5 if abs(x) == halfwidth else 1
        col = np.array([1, 1, 1]) * 180 if abs(x) == halfwidth else col
        p0, p1 = np.array([x, y, -halfheight]), np.array([x, y, halfheight])
        w.create_line(*world_to_plane(p0), *world_to_plane(p1), fill=rgb_to_hex(col), width=stroke)
    for z in np.arange(-halfheight, halfheight + spacing, spacing):
        stroke = 5 if abs(z) == halfheight else 1
        p0, p1 = np.array([-halfwidth, y, z]), np.array([halfwidth, y, z])
        w.create_line(*world_to_plane(p0), *world_to_plane(p1), fill=rgb_to_hex(col), width=stroke)





# Main function
def run():
    global t, dt, keys, show_axes, rho, phi, theta, y_far, y_close, room_width, room_height, wall_points, wall_cols, \
           center1, center2, radius, du, sphere_col, focus, light_points

    w.configure(background='black')
    while t <= keys[-1]:
        w.delete('all')

        # Camera Velocity Update
        # rho += v_rho
        # phi += v_phi
        # theta += v_theta

        if show_axes:
            mag = 200
            offset = np.array([0, 0, 0])
            colors = np.array([[255., 0., 0.], [0., 0., 255.], [128., 0., 128.]])
            direction_vectors = np.array([np.array([mag, 0, 0]) + offset, np.array([0, mag, 0]) + offset, np.array([0, 0, mag]) + offset])
            if show_axes:
                for i, v_i in enumerate(direction_vectors):
                    start = world_to_plane(offset)
                    end = world_to_plane(v_i)
                    w.create_line(start[0], start[1], end[0], end[1], fill=rgb_to_hex(colors[i]), width=5)

        # Animation
        # Keyframe 0 — Draw the box
        u, frame = squash(t)
        if frame == 0:
            u = ease_in(u)
            # Rotate camera very slightly to give 3d feeling
            theta0, theta1 = (-np.pi/2) + (np.pi/2), -np.pi/2
            theta = ((1 - u) * theta0) + (u * theta1)

            u, wall_num = squash2(u, 6)  # 5 walls + 1 area light
            for i in range(5):
                wall = wall_points[i]
                border_col = rgb_to_hex(wall_cols[i])
                if i < wall_num:
                    fill_col = rgb_to_hex(wall_cols[i] * 0.1)
                    w.create_polygon(*world_to_plane(wall[0]), *world_to_plane(wall[1]),
                                     *world_to_plane(wall[2]), *world_to_plane(wall[3]),
                                     fill=fill_col)
                    for j in range(4):
                        p0, p1 = wall[j], wall[(j+1) % 4]
                        w.create_line(*world_to_plane(p0), *world_to_plane(p1), fill=border_col, width=5)
                elif i == wall_num:
                    fill_col = rgb_to_hex(wall_cols[i] * u * 0.1)
                    u, edge_num = squash2(u, 4)  # 4 edges per wall
                    w.create_polygon(*world_to_plane(wall[0]), *world_to_plane(wall[1]),
                                     *world_to_plane(wall[2]), *world_to_plane(wall[3]),
                                     fill=fill_col)
                    for j in range(4):
                        start, finish = wall[j], wall[(j + 1) % 4]
                        if j < edge_num:
                            w.create_line(*world_to_plane(start), *world_to_plane(finish), fill=border_col, width=5)
                        elif j == edge_num:
                            p1 = line(u, start, finish)
                            w.create_line(*world_to_plane(start), *world_to_plane(p1), fill=border_col, width=5)

            if wall_num == 5:
                fill_col = np.array([100, 100, 1]) * u
                w.create_polygon(*world_to_plane(light_points[0]), *world_to_plane(light_points[1]),
                                 *world_to_plane(light_points[2]), *world_to_plane(light_points[3]),
                                 fill=rgb_to_hex(fill_col), outline='', width=5)

        # Keyframe 1 — Draw the spheres
        elif frame == 1:
            u, sphere_num = squash2(ease_in(u), 2)  # 2 walls
            col = u * sphere_col
            redraw_box()
            if sphere_num == 0:
                w.create_polygon(*get_circle_pts(u, radius, center1, du), outline='white', width=1, fill=rgb_to_hex(col))
            else:
                c1 = world_to_plane(center1)
                r2d = np.array([radius, radius])
                w.create_oval(*(c1 - r2d), *(c1 + r2d), outline='white', width=1, fill=rgb_to_hex(sphere_col))
                w.create_polygon(*get_circle_pts(u, radius, center2, du), outline='white', width=1, fill=rgb_to_hex(col))

        # Keyframe 2 — Zoom out to reveal pixels
        elif frame == 2:
            u = ease_in(u)
            redraw_box()
            redraw_spheres()

            # Draw raster (pixel grid)
            col = np.array([1, 1, 1]) * 120 * ease_in(u)
            spacing = 10
            halfwidth = room_width/10.
            halfheight = room_height/10.
            y = -1000
            for x in np.arange(-halfwidth, halfwidth + spacing, spacing):
                stroke = 5 if abs(x) == halfwidth else 1
                col = np.array([1, 1, 1]) * 180 * ease_in(u) if abs(x) == halfwidth else col
                p0, p1 = np.array([x, y, -halfheight]), np.array([x, y, halfheight])
                w.create_line(*world_to_plane(p0), *world_to_plane(p1), fill=rgb_to_hex(col), width=stroke)
            for z in np.arange(-halfheight, halfheight + spacing, spacing):
                stroke = 5 if abs(z) == halfheight else 1
                p0, p1 = np.array([-halfwidth, y, z]), np.array([halfwidth, y, z])
                w.create_line(*world_to_plane(p0), *world_to_plane(p1), fill=rgb_to_hex(col), width=stroke)

            # Pan the camera
            rho0, rho1 = -200, -55
            rho = ((1 - u) * rho0) + (u * rho1)
            theta0, theta1 = -np.pi/2, (-np.pi/2) + 0
            theta = ((1 - u) * theta0) + (u * theta1)

            # Scan through all the windows
            u, i = squash2(u, 2)
            if i == 1:
                spacing = 10
                halfwidth = room_width/10.
                halfheight = room_height/10.
                y = -1000
                n_xspaces, n_zspaces = int(2 * halfwidth / spacing) - 1, int(2 * halfheight / spacing) - 1
                u_ = (u * n_xspaces) % 1.0
                x = -halfwidth + (np.round(u_ * n_xspaces) * spacing)
                z = halfheight - (np.round(u * n_zspaces) * spacing)
                topleft = np.array([x, y, z])
                botright = topleft + np.array([spacing, 0, -spacing])
                u_, i = np.power(squash2(u, 2), 0.33)
                max_col = np.array([255, 255, 255])
                min_col = np.array([50, 50, 50])
                col = (u_ * max_col) + ((1 - u_) * min_col) if i == 0 else ((1.0 - u_) * max_col) + (u_ * min_col)
                w.create_rectangle(*world_to_plane(topleft), *world_to_plane(botright), fill=rgb_to_hex(col),
                                   outline='white', width=1)

        # Keyframe 3 — Pause.
        elif frame == 3:
            redraw_box()
            redraw_spheres()
            redraw_grid(u=ease_in(u))

        # Keyframe 4 — Switch to top view
        elif frame == 4:
            focus0, focus1 = 1230., 1600
            # u = np.power(u, 1.5)
            focus = ((1 - u) * focus0) + (u * focus1)
            u = ease_in(u)
            redraw_box()
            redraw_spheres()
            y = ((1 - u) * -1000) + (u * -400)
            redraw_grid(y)
            theta0, theta1 = -np.pi/2, 0
            theta = ((1 - u) * theta0) + (u * theta1)
            phi0, phi1 = np.pi/2, 0
            phi = ((1 - u) * phi0) + (u * phi1)
            radius0, radius1 = 110, 80
            radius = ((1 - u) * radius0) + (u * radius1)











            

        # End run
        t += dt
        w.update()
        time.sleep(0.001)


# Main function
if __name__ == '__main__':
    run()

# Necessary line for Tkinter
mainloop()
