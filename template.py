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


# TEMPLATE FUNCTIONS TO COPY/PASTE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Coordinate Shift
def A(val):
    return np.array([val[0] + window_w / 2, -val[1] + window_h / 2])


def A_many(val):
    return np.array([A(xy) for xy in val.reshape(int(len(val) / 2), 2)]).flatten()


# Ease functions
def linear(u):
    return u


ease_fns = [linear]
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# Global Parameters / Knobs 🎚
t = 0
t_max = 2
dt = 0.01
disp_num = 0


# Main function
def run():
    global t, t_max, dt, disp_num, ease_fns

    while t < t_max:
        w.delete('all')
        w.create_rectangle(0, 0, window_w, window_h, fill='black', outline='black')

        # Display graph / animation
        side = window_h / 2 * 0.8  # technically half-side
        w.create_line((window_w/2) - side, window_h/2, (window_w/2) + side, window_h/2, width=3)
        w.create_line(window_w/2, (window_h/2) - side, window_w/2, (window_h/2) + side, width=3)
        u = t / t_max
        x, y = (side * 2 * u) - side, (side * 2 * ease_fns[disp_num](u)) - side
        radius = 40
        w.create_oval(*A(np.array([x, y]) - radius), *A(np.array([x, y])), fill='red', outline='white', width=2)
        x_samples = np.linspace(0, 1, 20)
        ease = ease_fns[disp_num]
        w.create_line(*A_many(2 * side * np.vstack([x_samples, ease(x_samples)]).T.flatten()), fill='blue', width=3)

        # End run
        t += dt
        w.update()
        time.sleep(0.001)


# TKINTER STUFF ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def on_closing():
    root.destroy()


# Key handling function
def keyhandler(event):
    global t, t_max, dt, disp_num, ease_fns
    playing = t < t_max
    if event.char == ' ':
        t = 0
        run()

    if playing:
        if event.char == 'd':
            disp_num = (disp_num + 1) % len(ease_fns)
        if event.char == 'a':
            disp_num = (disp_num - 1) % len(ease_fns)


# Key bind
w.bind("<KeyPress>", keyhandler)
w.bind("<1>", lambda event: w.focus_set())
w.pack()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Main function
if __name__ == '__main__':
    run()

# Necessary line for Tkinter
mainloop()