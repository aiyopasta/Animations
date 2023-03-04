import copy
import tkinter
from tkinter import *
import time
import numpy as np
from PIL import Image, ImageTk

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


# Key binding
w.bind("<1>", lambda event: w.focus_set())
w.pack()


# Animation params
t = 0
dt = 0.01
keys = [0,       # Frame 0. Circle drawing begins
        0.5,     # Frame 1. Triangle drawing begins
        1.5,     # Frame 2. Begin drawing inner circle
        2.5,     # Frame 3. Begin drawing 3 inside edges
        3,       # Frame 4. Begin fading in the colors of the three formed triangles
        4,       # Frame 5. Begin fading in the color of the inner circle
        5,       # Frame 6. Move around the inner point to each of the inner positions.
        20,      # Frame 7. Begin fading out the inner triangles and their edges
        22,      # Frame 8. Move inner circle around to paint the gradient!
        25,      # Frame 9. Fade out the inner circle
        26]  # 25


# Get keyframe number
def frame(t_, interval=None):
    global keys
    if interval is None:
        interval = keys
    for i in range(len(interval)-1):
        if interval[i] <= t_ < interval[i+1]:
            return i


# Squash t parameter to keys
def squash(t_, intervals=None):
    global keys
    if intervals is None:
        intervals = keys
    for i in range(len(intervals)-1):
        if intervals[i] <= t_ < intervals[i+1]:
            return (t_ - intervals[i]) / (intervals[i+1] - intervals[i])


# Ease functions
def ease_in(t_):
    return np.power(t_, 2) / (np.power(t_, 2) + np.power(1 - t_, 2))


def ease_out(t_):
    return np.power(t_ - 1, 2) / (np.power(t_, 2) + np.power(t_ - 1, 2))


# Parametric shapes
def circle(u, radius=30, center=np.array([0, 0])):
    tau = np.pi * 2
    return (radius * np.array([np.cos(tau * u), np.sin(tau * u)])) + center


def line(u, start, fin):
    return ((1 - u) * start) + (u * fin)


# Get drawing points for special parametric shapes (lines only require 2 points)
# These will be transformed to the screen coordinates.
def circle_points(u, radius=30, center=np.array([0, 0]), du=0.01):
    pts = []
    for u_ in np.arange(0, u+du, du):
        pts.extend(A(circle(u_, radius, center)))
    return pts


# From ChatGPT
def rgb_to_hex(rgb):
    return '#' + ''.join(['%02x' % int(val) for val in rgb])


# Shape data representation
side = 500
height = side * np.sqrt(3) / 2
v0 = np.array([-side / 2, -height / 2])
v1 = np.array([+side / 2, -height / 2])
v2 = np.array([0, height / 2])
verts = [v0, v1, v2]
radius = 30
v = (v0 + v1 + v2) / 3

# Positions to move inner circle
positions = [v,
             (0.1 * v0) + (0.5 * v1) + (0.4 * v2),
             (0.0 * v0) + (0.8 * v1) + (0.2 * v2),
             (0.1 * v0) + (0.1 * v1) + (0.8 * v2),
             v2,
             (0.3 * v0) + (0.1 * v1) + (0.6 * v2),
             (0.5 * v0) + (0.5 * v1),
             (0.8 * v0) + (0.1 * v1) + (0.1 * v2),
             v]

# Formatting
black = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
apple_cols = np.array([[255., 59., 48.], [52, 199, 89], [0, 122, 255]])
pure_rgb = np.array([[255., 0., 0.], [0., 255., 0.], [0., 0., 255.]])
cols = pure_rgb
darker_cols = cols / 2.5

# Gradient image
pixels = np.zeros((int(height), int(side), 3), np.uint8)
pixels[:] = (0, 0, 0)


# Percent of a particular vertex color (idx in [0, 1, 2])
def percent(idx, pt=None):
    global verts, v, cols
    if pt is None:
        pt = v
    total_area = abs(np.cross(verts[1] - verts[0], verts[2] - verts[0])) / 2
    p1, p2 = verts[idx], verts[(idx+1) % 3]
    area = abs(np.cross(p2 - p1, pt - p1)) / 2
    return area / total_area


# Get the interpolated color
def mixed_color(pt=None):
    global verts, v, cols
    if pt is None:
        pt = v
    return sum([percent(idx, pt=pt) * np.array(cols[(idx+2) % 3]) for idx in range(3)])


# The final path that the inner circle will trace out
def path(u):
    global v0, v1, v2
    tau = (2 * np.pi) * 6.5
    r = 280 * u
    center = (v0 + v1 + v2) / 3
    pt = (r * np.array([np.cos(tau * u), np.sin(tau * u)])) + center

    # Clamp to inside of triangle
    for edge in [(v0, v1), (v1, v2), (v2, v0)]:
        e = edge[1] - edge[0]
        vert_to_pt = pt - edge[0]
        if np.cross(e, vert_to_pt) < 0 and np.dot(e, vert_to_pt) >= 0 and np.linalg.norm(vert_to_pt) <= np.linalg.norm(e):
            e_hat = e / np.linalg.norm(e)
            pt = (e_hat * np.dot(e_hat, vert_to_pt)) + edge[0]
            break

    return pt


# Redraw already drawn stuff
def redraw():
    global t, dt, keys, v0, v1, v2, verts, v, black, cols, positions, darker_cols
    # Redraw already drawn stuff
    # 1. Smaller triangles
    if 7 > frame(t) > 4:
        for i, edge in enumerate([(v0, v1), (v1, v2), (v2, v0)]):
            p1, p2, p3 = edge[0], edge[1], v
            w.create_polygon(*A(p1), *A(p2), *A(p3), fill=rgb_to_hex(darker_cols[(i + 2) % 3]), outline='')
    # 2. Inner triangle's edges
    if 7 > frame(t) > 3:
        for i in range(3):
            p1, p2 = verts[i], v
            w.create_line(*A(p1), *A(p2), fill='white', width=1)
    # 3. Inner circle as a BLACK one
    if 5 > frame(t) > 2:
        w.create_oval(*A(np.array([v[0] - radius, v[1] - radius])),
                      *A(np.array([v[0] + radius, v[1] + radius])),
                      fill='black', outline='white', width=3)
    # 4. Big triangle's edges
    if frame(t) > 1:
        for edge in [(v0, v1), (v1, v2), (v2, v0)]:
            w.create_line(*A(edge[0]), *A(edge[1]), fill='white', width=3)
    # 5. Big triangle's vertices
    if frame(t) > 0:
        for i in range(3):
            center = verts[i]
            w.create_oval(*A(np.array([center[0] - radius, center[1] - radius])),
                          *A(np.array([center[0] + radius, center[1] + radius])),
                          fill=rgb_to_hex(cols[i]), outline='white', width=3)


# Main function
def run():
    global t, dt, keys, v0, v1, v2, verts, v, black, cols, pixels
    RGB = ['R', 'G', 'B']

    w.configure(background='black')
    while t <= keys[-1]:
        w.delete('all')
        # 1. Draw the three circles
        u = squash(t)
        if frame(t) == 0:
            u = ease_in(u)
            color = ((1 - u) * black) + (u * cols)
            for i in range(3):
                w.create_polygon(*circle_points(u, radius, verts[i], du=0.001),
                                 outline='white', width=3, fill=rgb_to_hex(color[i]))

        # 2. Connect the circles to form triangle
        elif frame(t) == 1:
            # Connect the circles
            u = ease_in(u)
            for edge in [(v0, v1), (v1, v2), (v2, v0)]:
                p1, p2 = edge[0], line(u, edge[0], edge[1])
                w.create_line(*A(p1), *A(p2), fill='white', width=3)
            redraw()

        # 3. Draw the inner circle
        elif frame(t) == 2:
            redraw()
            u = ease_in(u)
            w.create_polygon(*circle_points(u, radius, v, du=0.001), outline='white', width=3, fill='')

        # 4. Draw the inner 3 edges
        elif frame(t) == 3:
            u = ease_in(u)
            for i in range(3):
                p1, p2 = verts[i], line(u, verts[i], v)
                w.create_line(*A(p1), *A(p2), fill='white', width=1)
            redraw()

        # 5. Color in the three formed triangles + add text
        elif frame(t) == 4:
            u = ease_in(u)
            for i, edge in enumerate([(v0, v1), (v1, v2), (v2, v0)]):
                color = ((1 - u) * black[0]) + (u * darker_cols[(i+2) % 3])
                p1, p2, p3 = edge[0], edge[1], v
                w.create_polygon(*A(p1), *A(p2), *A(p3), fill=rgb_to_hex(color), outline='')
            redraw()
            for i, edge in enumerate([(v0, v1), (v1, v2), (v2, v0)]):
                color = ((1 - u) * black[0]) + (u * np.array([255., 255., 255.]))
                p1, p2, p3 = edge[0], edge[1], v
                com = (p1 + p2 + p3) / 3
                font = ('CMU Serif', '30')
                w.create_text(*A(com), text=str(np.round(percent(i) * 100))[:-2] + '% ' + RGB[(i+2) % 3], fill=rgb_to_hex(color), font=font)

        # 6. Color in the inner circle
        elif frame(t) == 5:
            redraw()
            u = ease_in(u)
            color = ((1 - u) * black[0]) + (u * mixed_color())
            w.create_oval(*A(np.array([v[0] - radius, v[1] - radius])),
                          *A(np.array([v[0] + radius, v[1] + radius])),
                          fill=rgb_to_hex(color), outline='white', width=3)

            for i, edge in enumerate([(v0, v1), (v1, v2), (v2, v0)]):
                color = np.array([255., 255., 255.])
                p1, p2, p3 = edge[0], edge[1], v
                com = (p1 + p2 + p3) / 3
                font = ('CMU Serif', '30')
                w.create_text(*A(com), text=str(np.round(percent(i) * 100))[:-2] + '% ' + RGB[(i+2) % 3], fill=rgb_to_hex(color), font=font)

        # 7. Move around inner circle
        elif frame(t) == 6:
            redraw()
            n = len(positions)-1
            i = frame(u * n, np.arange(n+1))
            u = ease_in(squash(u * n, intervals=np.arange(n+1)))
            v = ((1 - u) * positions[i]) + (u * positions[i+1])
            w.create_oval(*A(np.array([v[0] - radius, v[1] - radius])),
                          *A(np.array([v[0] + radius, v[1] + radius])),
                          fill=rgb_to_hex(mixed_color()), outline='white', width=3)

            for i, edge in enumerate([(v0, v1), (v1, v2), (v2, v0)]):
                color = np.array([255., 255., 255.])
                p1, p2, p3 = edge[0], edge[1], v
                com = (p1 + p2 + p3) / 3
                size = int(percent(i) * 100)
                if size != 0:
                    font = ('CMU Serif', str(size))
                    w.create_text(*A(com), text=str(np.round(percent(i) * 100))[:-2] + '% ' + RGB[(i+2) % 3], fill=rgb_to_hex(color), font=font)

        # 8. Fade out the small triangles and their edges
        elif frame(t) == 7:
            u = ease_in(u)
            for i, edge in enumerate([(v0, v1), (v1, v2), (v2, v0)]):
                color = ((1 - u) * darker_cols[(i+2) % 3]) + (u * black[0])
                p1, p2, p3 = edge[0], edge[1], v
                w.create_polygon(*A(p1), *A(p2), *A(p3), fill=rgb_to_hex(color), outline='')
            color = ((1 - u) * np.array([255., 255., 255.])) + (u * black[0])
            for i in range(3):
                p1, p2 = verts[i], v
                w.create_line(*A(p1), *A(p2), fill=rgb_to_hex(color), width=1)
            for i, edge in enumerate([(v0, v1), (v1, v2), (v2, v0)]):
                color = ((1 - u) * np.array([255., 255., 255.])) + (u * black[0])
                p1, p2, p3 = edge[0], edge[1], v
                com = (p1 + p2 + p3) / 3
                font = ('CMU Serif', '30')
                w.create_text(*A(com), text=str(np.round(percent(i) * 100))[:-2] + '% ' + RGB[(i+2) % 3], fill=rgb_to_hex(color), font=font)
            redraw()
            w.create_oval(*A(np.array([v[0] - radius, v[1] - radius])),
                          *A(np.array([v[0] + radius, v[1] + radius])),
                          fill=rgb_to_hex(mixed_color()), outline='white', width=3)

        # 9. Inner circle shades in the gradient
        elif frame(t) == 8:
            u = ease_in(u)  # TODO: Make v trace a path based on u
            # Make v trace a path based on u
            v = path(u)

            # 1. Update the image
            tall, wide, _ = pixels.shape
            for r in range(tall):
                for c in range(wide):
                    pt = np.array([c - (wide/2), (tall/2) - r])
                    if np.linalg.norm(pt - v) < radius:
                        pixels[r][c] = mixed_color(pt=pt)

            # 2. Draw the image
            img = ImageTk.PhotoImage(Image.fromarray(pixels))
            w.create_image(*A(np.array([-wide/2, tall/2])), image=img, anchor=NW)

            # 3. Draw stupid mask
            w.create_polygon(*A(v0), *A(v2), window_w, 0, 0, 0, 0, window_h, fill='black')
            w.create_polygon(*A(v1), *A(v2), 0, 0, window_w, 0, window_w, window_h, fill='black')

            # Draw the rest
            redraw()
            w.create_oval(*A(np.array([v[0] - radius, v[1] - radius])),
                          *A(np.array([v[0] + radius, v[1] + radius])),
                          fill=rgb_to_hex(mixed_color()), outline='white', width=3)

        elif frame(t) == 9:
            u = ease_in(u)
            # Draw the image
            tall, wide, _ = pixels.shape
            img = ImageTk.PhotoImage(Image.fromarray(pixels))
            w.create_image(*A(np.array([-wide/2, tall/2])), image=img, anchor=NW)

            # Draw stupid mask
            w.create_polygon(*A(v0), *A(v2), window_w, 0, 0, 0, 0, window_h, fill='black')
            w.create_polygon(*A(v1), *A(v2), 0, 0, window_w, 0, window_w, window_h, fill='black')

            # Redraw the big triangle stuff
            redraw()

            # Move the inner circle to one of the vertices
            p0, p1 = path(1), v0
            v = ((1 - u) * p0) + (u * p1)

            # Draw the inner circle
            w.create_oval(*A(np.array([v[0] - radius, v[1] - radius])),
                          *A(np.array([v[0] + radius, v[1] + radius])),
                          fill=rgb_to_hex(mixed_color()), outline='white', width=3)




        # End run
        t += dt
        w.update()
        if frame(t) == frame(t - dt):
            time.sleep(0.001)
        else:
            time.sleep(0.1)


# Main function
if __name__ == '__main__':
    run()

# Necessary line for Tkinter
mainloop()
