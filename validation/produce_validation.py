#!/usr/bin/env python3

# Produce video sequences to validate Spatiotemporal energy model
# from Adelson & Bergen, 1985
#
# Copyright (C) 2022 Aravind Battaje
# Email: aravind@oxidification.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser Public License for more details.
#
# You should have received a copy of the GNU Lesser Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cv2
import numpy as np

IMG_WIDTH, IMG_HEIGHT = 480, 480
NUM_FRAMES = 60
NUM_FPS = 30

t = np.linspace(0, 1, IMG_WIDTH, endpoint=False)

######################################
# A. Continuous motion
vid_writer = cv2.VideoWriter(
    'continuous_motion.mp4', cv2.VideoWriter_fourcc('H','2','6','4'), NUM_FPS, (IMG_HEIGHT, IMG_WIDTH))

half_white_half_black_1d = (t < (1 / 2)).astype(np.uint8) * 255
half_white_half_black_2d = np.r_[[half_white_half_black_1d]*IMG_HEIGHT]

# First 10 frames no motion
for ix in range(NUM_FRAMES//6):
    vid_writer.write(half_white_half_black_2d)

# Next 40 frames sin-wave
for ix in range(4 * (NUM_FRAMES//6)):
    img_new = np.roll(half_white_half_black_2d, int(IMG_WIDTH / 40 * np.sin(2 * np.pi * 1 / (4*(NUM_FRAMES//6)) * ix))) 
    vid_writer.write(img_new)

# Next 10 frames no motion
for ix in range(NUM_FRAMES//6):
    vid_writer.write(half_white_half_black_2d)

vid_writer.release()
######################################


######################################
# B. Sampled motion
# Same as above EXCEPT the amplitude of sin wave is larger, simulating larger "sampled" displacements
vid_writer = cv2.VideoWriter(
    'sampled_motion.mp4', cv2.VideoWriter_fourcc('H','2','6','4'), NUM_FPS, (IMG_HEIGHT, IMG_WIDTH))

half_white_half_black_1d = (t < (1 / 2)).astype(np.uint8) * 255
half_white_half_black_2d = np.r_[[half_white_half_black_1d]*IMG_HEIGHT]

# First 10 frames no motion
for ix in range(NUM_FRAMES//6):
    vid_writer.write(half_white_half_black_2d)

# Next 40 frames sin-wave
for ix in range(4 * (NUM_FRAMES//6)):
    img_new = np.roll(half_white_half_black_2d, int(IMG_WIDTH / 20 * np.sin(2 * np.pi * 1 / (4*(NUM_FRAMES//6)) * ix))) 
    vid_writer.write(img_new)

# Next 10 frames no motion
for ix in range(NUM_FRAMES//6):
    vid_writer.write(half_white_half_black_2d)

vid_writer.release()
######################################


######################################
# C1. Reverse Phi negative test -- Random pattern
vid_writer = cv2.VideoWriter(
    'reverse_phi_negative.mp4', cv2.VideoWriter_fourcc('H','2','6','4'), NUM_FPS, (IMG_HEIGHT, IMG_WIDTH))

rng = np.random.default_rng(20221129)
random_nums_1d = rng.uniform(0, 255, IMG_WIDTH)
random_nums_2d = np.r_[[random_nums_1d]*IMG_HEIGHT]

for ix in range(NUM_FRAMES):
    img_new = np.roll(random_nums_2d, ix * 2).astype(np.uint8)
    vid_writer.write(img_new)

vid_writer.release()
######################################

######################################
# C2. Reverse Phi positive test -- Random pattern with alternating polarity
vid_writer = cv2.VideoWriter(
    'reverse_phi_positive.mp4', cv2.VideoWriter_fourcc('H','2','6','4'), NUM_FPS, (IMG_HEIGHT, IMG_WIDTH))

rng = np.random.default_rng(20221129)
random_nums_1d = rng.uniform(0, 255, IMG_WIDTH)


for ix in range(NUM_FRAMES):
    if ix % 2 == 0:
        random_nums_1d_for_img = np.roll(random_nums_1d, ix * 2)
    else:
        random_nums_1d_for_img = np.roll(255 - random_nums_1d, ix * 2)
    random_nums_2d_for_img = np.r_[[random_nums_1d_for_img]*IMG_HEIGHT]
    img_new = random_nums_2d_for_img.astype(np.uint8)
    vid_writer.write(img_new)

vid_writer.release()
######################################


######################################
# Helpers for fluted square wave tests
def gen_square_wave(t, freq=1, fluted_square_wave=False):
    ret = np.zeros_like(t)
    start_num = 2 if fluted_square_wave else 1
    for k in range(start_num, 500):
        ret += 1/(2*k - 1) * np.sin(2 * np.pi * freq * (2*k - 1) * t)
    return 4 / np.pi * ret

square_wave_freq = 36
######################################

######################################
# D1. Fluted square wave negative test -- Full square wave
vid_writer = cv2.VideoWriter(
    'fluted_square_negative.mp4', cv2.VideoWriter_fourcc('H','2','6','4'), NUM_FPS, (IMG_HEIGHT, IMG_WIDTH))

square_wave_2d = np.r_[[gen_square_wave(t, square_wave_freq, False) + 1]*IMG_HEIGHT]
for ix in range(NUM_FRAMES):
    img_new = (np.roll(square_wave_2d, ix * IMG_WIDTH // square_wave_freq // 4) * 255).astype(np.uint8)
    vid_writer.write(img_new)

vid_writer.release()
######################################

######################################
# D2. Fluted square wave positive test -- Fluted square wave
vid_writer = cv2.VideoWriter(
    'fluted_square_positive.mp4', cv2.VideoWriter_fourcc('H','2','6','4'), NUM_FPS, (IMG_HEIGHT, IMG_WIDTH))

square_wave_2d = np.r_[[gen_square_wave(t, square_wave_freq, True) + 1]*IMG_HEIGHT]
for ix in range(NUM_FRAMES):
    img_new = (np.roll(square_wave_2d, ix * IMG_WIDTH // square_wave_freq // 4) * 255).astype(np.uint8)
    vid_writer.write(img_new)

vid_writer.release()
######################################

