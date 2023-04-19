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
pygame.display.set_caption("TikTok Template")


# Coordinate Shift
def A(val):
    return np.array([val[0] + width / 2, -val[1] + height / 2])


# Keyframe / timing params
FPS = 60
t = 0
dt = 0.01    # i.e., 1 frame corresponds to +0.01 in parameter space = 0.01 * FPS = +0.6 per second (assuming 60 FPS)
keys = [0,   # Keyframe 0. Start drawing circle.
        10   # Keyframe n. Animation finished.
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


# From ChatGPT
def rgb_to_hex(rgb):
    return '#' + ''.join(['%02x' % int(val) for val in rgb])


# Parametric shapes
def line(u, start, fin):
    return ((1 - u) * start) + (u * fin)


# Main animation params / data-structure
x = 0


# Additional params / knobs
play_music = True


def main():
    global t, dt, keys, FPS, x, save_anim, play_music

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

    # Play music
    if not save_anim and play_music:
        playsound('木(剪辑版).mov', block=False)

    # Game loop
    count = 0
    while run:
        # Basic drawing demo ––– (feel free to delete!)
        window.fill((0, 0, 0))
        rekt = (300 + x, 300, 100, 100)  # rectangle definition: (x_top, y_top, width, height)
        pygame.draw.rect(surface=window, color=(255, 0, 0), rect=rekt, width=0)  # Really dumb thing: it doesn't draw border + fill in a single line...
        pygame.draw.rect(surface=window, color=(255, 255, 255), rect=rekt, width=3)    # Need this additional line for border.
        x += 2
        pygame.draw.polygon(window, (0, 255, 0), [(100, 100), (200, 0), (300, 300), (100, 400)], 0)
        pygame.draw.lines(window, (0, 0, 255), False, [(width, height), (200, 800), (300 + x, 400), (100, 900)], width=2)  # false there means "open"

        # Draw preliminary stuff for animation
        # Do stuff...
        count += 1

        # Animation!
        # Keyframe 0 —— <something>
        u, frame = squash(t)


        # We handle keys pressed inside the gameloop in PyGame
        keys_pressed = pygame.key.get_pressed()
        # do stuff....


        # End run (1. Tick clock, 2. Save the frame, and 3. Detect if window closed)
        pygame.display.update()
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
        os.system(f'{ffmpeg_path} -r 60 -i {input_files} -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "eq=brightness=0.00:saturation=1.3" {output_file} > /dev/null 2>&1')
        print('Saved video to ' + output_file)


if __name__ == "__main__":
    main()
