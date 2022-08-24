#!/usr/bin/env python3

# Motion Perception Tool
# An efficient, real-time implementation of Adelson's spatiotemporal energy model.
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
from math import factorial
from collections import deque
import torch

import sys
import argparse
import logging
from time import time

from gabor import gabor_kernel_separate, gabor_kernel_skimage

class MotionEnergy():

    class SplitConv2d(torch.nn.Module):
        def __init__(self, kernel_x, kernel_y):
            super().__init__()

            # NOTE unfortunately, native complex ops for conv are not currently
            # supported in PyTorch (<=1.11). So multiple "real" convolutions
            # must be performed on different parts of the complex number and
            # finally combined. However, some computational efficiences can still be achieved
            #   1. As image input only has real component, 
            #      the first pass (of the 2D conv split to 1D scheme)
            #      can be batched. So kernel_y only needs one conv operation
            #   2. The first pass results in a complex response, and the second pass
            #      can naively work with this to do 4 (real) conv operations
            #   3. Or to save a little bit more time, a variation of the 
            #      Karatsuba algorithm can be used, which needs 3 (real) conv ops
            self.conv_y = torch.nn.Conv2d(1, 2, kernel_y.real.shape[0], bias=False)
            kernel_y_real_and_imag = np.stack((kernel_y.real, kernel_y.imag))
            self.conv_y.weight = torch.nn.Parameter(
                torch.tensor(kernel_y_real_and_imag.reshape(2, 1, -1, 1), dtype=torch.float32))
            self.conv_x_real = torch.nn.Conv2d(1, 1, kernel_x.real.shape[0], bias=False)
            self.conv_x_real.weight = torch.nn.Parameter(
                torch.tensor(kernel_x.real.reshape(1, 1, 1, -1), dtype=torch.float32))
            self.conv_x_imag = torch.nn.Conv2d(1, 1, kernel_x.imag.shape[0], bias=False)
            self.conv_x_imag.weight = torch.nn.Parameter(
                torch.tensor(kernel_x.imag.reshape(1, 1, 1, -1), dtype=torch.float32))

            # To be used for the Karatsuba
            self.conv_x_comb = torch.nn.Conv2d(1, 1, kernel_x.imag.shape[0], bias=False)
            self.conv_x_comb.weight = torch.nn.Parameter(
                torch.tensor(
                    (kernel_x.real + kernel_x.imag).reshape(1, 1, 1, -1), dtype=torch.float32))

        def forward(self, incoming, expected_width=None):
            # If native complex was supported by PyTorch
            # intermediate = self.conv_y(incoming)
            # ret = self.conv_x(intermediate)
            # return ret

            intermediate = self.conv_y(incoming)
            intermediate_real = intermediate[:, 0].unsqueeze(1)
            intermediate_imag = intermediate[:, 1].unsqueeze(1)

            # Naive 4 conv ops on the second pass
            # resp_real = self.conv_x_real(intermediate_real) - self.conv_x_imag(intermediate_imag)
            # resp_imag = self.conv_x_real(intermediate_imag) + self.conv_x_imag(intermediate_real)

            # Variation of Karatsuba (3 conv ops)
            # https://en.wikipedia.org/wiki/Multiplication_algorithm#Complex_number_multiplication
            # https://github.com/pytorch/pytorch/issues/71108#issuecomment-1016889045
            k_a = self.conv_x_real(intermediate_real)
            k_b = self.conv_x_imag(intermediate_imag)
            k_c = self.conv_x_comb(intermediate_real + intermediate_imag)
            resp_real = k_a - k_b
            resp_imag = k_c - k_a - k_b

            # NOTE to save some more time, padding could be 
            # done after all kernels in one go.
            # But kernels can be of varying size.
            # Hence, padding to a common resolution now
            if expected_width is not None:
                # Expected pad difference to match image resolution
                # NOTE Careful using same pad_diff for all. 
                # That's assuming w and h of kernel is same
                pad_diff = expected_width - resp_real.shape[-1]
                padder = torch.nn.ZeroPad2d(pad_diff // 2)

                resp_real = padder(resp_real)
                resp_imag = padder(resp_imag)

            return resp_real, resp_imag


    def __init__(self, disable_video_scaling=False, accelerate_with_cuda=False):

        # By default, CPU is used
        self.device_type = torch.device("cpu")
        if accelerate_with_cuda:
            if torch.cuda.is_available():
                self.device_type = torch.device("cuda")
                logging.info('CUDA will be used to accelerate compute.')
            else:
                logging.warning('CUDA not available! Will be using CPU instead.')

        # Make kernels
        # NOTE below kernels work fine for 240 x 240 input
        # image resolution. With bigger images, change these too
        self.spatial_kernel_params = [
            (0.3, 0),
            (0.3, 90),
            (0.3, 45),
            (0.3, 135)
        ]
        self.spatial_convs = []
        logging.info('Initializing spatial kernels')
        for param in self.spatial_kernel_params:
            freq, orin = param
            kernel_x, kernel_y = gabor_kernel_separate(freq, theta=np.deg2rad(orin))

            # Validation check
            full_kernel_2d = gabor_kernel_skimage(freq, theta=np.deg2rad(orin))
            full_kernel_from_1d = np.outer(kernel_y, kernel_x)
            assert np.allclose(full_kernel_2d, full_kernel_from_1d)

            self.spatial_convs.append(
                self.SplitConv2d(kernel_x, kernel_y).to(self.device_type))

            logging.info(
                f'Freq. = {freq} cpp, Orien. = {orin} deg, Kern. size = {full_kernel_from_1d.shape}')


        # TODO Read parameters from config file
        dt = 1 / 30
        time_length = 90 / 1000 # ms to s
        self.num_frames = int(np.ceil(time_length / dt))
        time_array = np.linspace(0, time_length, self.num_frames+1)[1:]

        # Time functions due to George Mather
        # http://www.georgemather.com/Model.html
        slow_n = 6
        fast_n = 3
        beta = 0.99

        # k is scaling for time
        k = 125

        slow_t = (k * time_array)**slow_n * np.exp(-k * time_array) * (
            1 / factorial(slow_n) - beta / factorial(slow_n + 2) * (k * time_array)**2)
        fast_t = (k * time_array)**fast_n * np.exp(-k * time_array) * (
            1 / factorial(fast_n) - beta / factorial(fast_n + 2) * (k * time_array)**2)
        logging.info(f'Num frames = {self.num_frames}')
        logging.info(f'Slow time kernel = {slow_t}\nFast time kernel = {fast_t}')

        self.slow_t_conv = torch.nn.Conv2d(self.num_frames, 1, (1, 1), bias=False)
        self.slow_t_conv.weight = torch.nn.Parameter(
            torch.tensor(slow_t.reshape(1, -1, 1, 1),
            dtype=torch.float32, device=self.device_type))
        self.fast_t_conv = torch.nn.Conv2d(self.num_frames, 1, (1, 1), bias=False)
        self.fast_t_conv.weight = torch.nn.Parameter(
            torch.tensor(fast_t.reshape(1, -1, 1, 1),
            dtype=torch.float32, device=self.device_type))

        # FIFO deck to accumulate max. number of frames
        # This will be useful if each loop is faster than incoming
        # frames, and then the decks need not be flushed
        self.img_grey_stack = deque(maxlen=self.num_frames)

        self.motion_energy = None

        self.disable_video_scaling = disable_video_scaling

    
    def img_callback(self, img_in, ensure_contiguous=False):
        time_start = time()

        # Center crop, square image and resize to 240 x 240
        height_img_in, width_img_in, _ = img_in.shape
        if width_img_in > height_img_in:
            height_range_start = 0
            height_range_last = height_img_in
            width_range_start = (width_img_in - height_img_in) // 2
            width_range_last = width_range_start+height_img_in
        elif height_img_in >= width_img_in:
            width_range_start = 0
            width_range_last = width_img_in
            height_range_start = (height_img_in - width_img_in) // 2
            height_range_last = height_range_start+width_img_in
        img_in = img_in[
            height_range_start:height_range_last,
            width_range_start:width_range_last, :]
        if not self.disable_video_scaling:
            img_in = cv2.resize(img_in, (240, 240))

        img_grey = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        height_img, width_img = img_grey.shape
        
        img_grey_torch = torch.tensor(
            img_grey.reshape(1, 1, height_img, width_img),
            dtype=torch.float32, device=self.device_type)
        self.img_grey_stack.append(img_grey_torch)
        
        if len(self.img_grey_stack) == self.num_frames:
            # Show latest image used to compute motion energy
            cv2.imshow('input_image', img_in)
            
            with torch.no_grad():
                input_images  = torch.cat(tuple(self.img_grey_stack), dim=0)
                
                edge_resp = np.zeros((height_img, width_img))
                
                # Spatial convolutions
                stacked_gabor_resps = []
                for spatial_conv in self.spatial_convs:
                    gabor_resp_even, gabor_resp_odd = spatial_conv(input_images, width_img)
                    # Will stack even, odd, even, odd....
                    stacked_gabor_resps.extend(
                        (gabor_resp_even.squeeze(), gabor_resp_odd.squeeze()))

                    # Plain edge response (spatial convolutions)
                    edge_resp += (
                        gabor_resp_even[0] +
                        gabor_resp_odd[0]
                    ).squeeze().cpu().numpy()

                cv2.imshow('edge_response', edge_resp/len(self.spatial_convs))
                stacked_gabor_resps = torch.stack(stacked_gabor_resps)

                # Temporal convolutions 1x1 
                slow_time_resp = self.slow_t_conv(stacked_gabor_resps)
                fast_time_resp = self.fast_t_conv(stacked_gabor_resps)

                total_energy_x = np.zeros((height_img, width_img))
                total_energy_y = np.zeros((height_img, width_img))
                for ix, param in enumerate(self.spatial_kernel_params):
                    even_slow, even_fast = slow_time_resp[2*ix], fast_time_resp[2*ix]
                    odd_slow, odd_fast = slow_time_resp[2*ix+1], fast_time_resp[2*ix+1]

                    resp_negdir_1 =  odd_fast +  even_slow
                    resp_negdir_2 = -odd_slow +  even_fast
                    resp_posdir_1 = -odd_fast +  even_slow
                    resp_posdir_2 =  odd_slow +  even_fast

                    # NOTE converting to CPU is no-op when device type already CPU
                    energy_negdir = (
                        resp_negdir_1**2 + resp_negdir_2**2).squeeze().cpu().numpy()
                    energy_posdir = (
                        resp_posdir_1**2 + resp_posdir_2**2).squeeze().cpu().numpy()
                    energy_thisdir = energy_negdir - energy_posdir

                    orientation = np.deg2rad(param[1])
                    energy_in_image_space_x = energy_thisdir * np.cos(orientation)
                    energy_in_image_space_y = energy_thisdir * np.sin(orientation)

                    total_energy_x += energy_in_image_space_x
                    total_energy_y += energy_in_image_space_y

                # Total energy in polar coordinates
                total_energy_mag = np.sqrt(total_energy_x**2 + total_energy_y**2)
                total_energy_ang = np.arctan2(total_energy_y, total_energy_x)
                self.motion_energy = (total_energy_mag, total_energy_ang)

            motion_hue = ((total_energy_ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
            motion_val = np.clip(total_energy_mag * 200, 0, 255).astype(np.uint8)
            motion_sat = np.ones_like(img_grey) * 255
            motion_visu_img_hsv = np.stack((motion_hue, motion_sat, motion_val), axis=-1)
            motion_visu_img = cv2.cvtColor(motion_visu_img_hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('motion_energy', motion_visu_img)

            # ensure_contiguous is a hint that stuff does not run real-time.
            # So flush the image stack and accumulate self.num_frames contiguous frames
            if ensure_contiguous:
                self.img_grey_stack.clear()

            logging.info(f'Duration per loop = {time() - time_start} s')


if __name__ == "__main__":
    # Setup a command-line argument parser
    parser = argparse.ArgumentParser(
        description = 'Visualize spatiotemporal energy model on webcam feed or video file',
        # exit_on_error=True, needs Python 3.9
    )
    parser.add_argument(
        '-f', '--file',
        default=None,
        type=str,
        help='video filename as input; if not specified, defaults to a camera device'
    )
    parser.add_argument(
        '-c', '--cam-id',
        default=0,
        type=int,
        help="""camera ID as input; typically 0 is internal webcam, 1 is external camera
        (default: 0); NOTE ignored if --file is specified"""
    )
    parser.add_argument(
        '-x', '--disable-scale',
        action='store_true',
        help='disables video input scaling, else scales input to 240x240'
    )
    parser.add_argument(
        '-e', '--ensure-contiguous',
        action='store_true',
        help="""makes sure spatiotemporal volume contains continuously sampled
        images from the camera input. This is most useful if it is taking too
        long to produce an iteration of visualization (very choppy appearance).
        NOTE ineffective if --file is specified"""
    )
    parser.add_argument(
        '-a', '--accelerate-with-cuda',
        action='store_true',
        help="""tries to accelerate compute with NVIDIA CUDA instead of running on CPU.
        If your computer does not have a supporting GPU, CPU will be used as a fallback."""
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='print extra information on the command line'
    )
    args = parser.parse_args()

    # Setup a logger
    logging.basicConfig(
        format = '%(levelname)s: %(message)s',
        level = logging.INFO if args.verbose else logging.WARN)

    # The main object of this demo
    motion_energy_obj = MotionEnergy(
        disable_video_scaling=args.disable_scale,
        accelerate_with_cuda=args.accelerate_with_cuda)

    if args.disable_scale:
        logging.warning('Video scaling to 240x240 is disabled. This might lead to slow processing.')
        if not args.ensure_contiguous:
            logging.warning('Consider specifying --ensure-contiguous')

    print('Starting motion energy visualization. To exit, press ESC on any visualization window.')
    # Setup input to motion energy processor
    if args.file is not None:
        cap = cv2.VideoCapture(args.file)
        # Read first frame to see if file/codec exists
        ret_val, _ = cap.read()
        
        if ret_val is False:
            logging.error(
                f'{args.file} does not exist or is not a valid video file')
            sys.exit(1)
    else:
        # Try reading with DSHOW (works in most Windows)
        cap = cv2.VideoCapture(args.cam_id, cv2.CAP_DSHOW)
        ret_val, _ = cap.read()

        # Else, without DSHOW (works in most Linux/Mac)
        if ret_val is False:
            cap = cv2.VideoCapture(args.cam_id)
            ret_val, _ = cap.read()

        if ret_val is False:
            logging.error(
                f'{args.cam_id} is an invalid camera. '
                'Make sure camera is attached and usable in other programs.')
            sys.exit(1)

    # The main loop
    while True:
        ret_val, img = cap.read()

        if ret_val is False and args.file is not None:
            # End of video reached
            break

        motion_energy_obj.img_callback(
            img, ensure_contiguous=args.ensure_contiguous)
        
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
