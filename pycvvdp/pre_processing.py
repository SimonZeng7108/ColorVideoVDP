"""Perceptual Laplacian pyramid pruning.

This script converts a video to the DKL colour space, decomposes each channel
with :class:`weber_contrast_pyr` and prunes pyramid coefficients using the
castleCSF sensitivity model.  The pruned frames are reconstructed and written
back to a new 8‑bit sRGB video.  The reconstruction error is reported in
integer RGB units.

Example usage::

    python -m pycvvdp.pre_processing input.mp4 output.mp4 --csf-strength 1.0
"""

import argparse
import logging
import torch
import numpy as np

from .video_source_file import video_source_file
from .display_model import vvdp_display_photometry, vvdp_display_geometry
from .lpyr_dec import weber_contrast_pyr
from .csf import castleCSF
from .dump_channels import dkld65_to_rgb
from .video_writer import VideoWriter
from . import utils


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prune Laplacian pyramid coefficients using the CSF model")
    parser.add_argument("input", type=str, help="input video file")
    parser.add_argument("output", type=str, help="output video file")
    parser.add_argument("-d", "--display", type=str, default="standard_4k",
                        help="display model name or ? to list models")
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device: cpu, cuda, etc.")
    parser.add_argument("-c", "--config-paths", nargs="+", default=[],
                        help="paths to configuration files or directories")
    parser.add_argument("--fps", type=float, default=None, help="override frame rate")
    parser.add_argument("--nframes", type=int, default=-1, help="number of frames to process")
    parser.add_argument("--ffmpeg-cc", action="store_true", default=False,
                        help="use ffmpeg for color conversion")
    parser.add_argument(
        "--csf-strength",
        type=float,
        default=1.0,
        help="scale factor for CSF threshold (0 disables pruning, 2 is max)"
    )
    parser.add_argument("-q", "--quiet", action="store_true", default=False,
                        help="suppress informational messages")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="print additional information")
    return parser.parse_args()


def reconstruct_from_contrast(pyr: weber_contrast_pyr, bands, log_bkg):
    """Reconstruct the image from a Weber contrast pyramid."""
    img = bands[-1] * (10.0 ** log_bkg[-1])
    for i in reversed(range(len(bands) - 1)):
        layer = bands[i] * (10.0 ** log_bkg[i])
        img = pyr.gausspyr_expand(img, [layer.shape[-2], layer.shape[-1]]) + layer
    return img


def main():
    args = parse_arguments()

    if not 0.0 <= args.csf_strength <= 2.0:
        raise ValueError("--csf-strength must be in the range [0,2]")

    # Configure logging similar to run_cvvdp.py
    log_level = logging.ERROR if args.quiet else (
        logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=log_level)

    if args.display == "?":
        vvdp_display_photometry.list_displays(args.config_paths)
        return

    # Select computation device
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")

    # Open the video once as both test and reference for reconstruction error
    vs = video_source_file(
        args.input,
        args.input,
        display_photometry=args.display,
        config_paths=args.config_paths,
        frames=args.nframes,
        fps=args.fps,
        ffmpeg_cc=args.ffmpeg_cc,
        verbose=args.verbose,
    )

    height, width, n_frames = vs.get_video_size()
    fps = vs.get_frames_per_second()

    # Display geometry determines pixels-per-degree for the CSF
    display_geometry = vvdp_display_geometry.load(args.display, config_paths=args.config_paths)
    ppd = display_geometry.get_ppd()

    # Laplacian pyramid with Weber contrast, as used by cvvdp_metric
    pyr = weber_contrast_pyr(width, height, ppd, device, contrast="weber_g1")

    # Load metric parameters to get CSF version and calibration constants
    params_file = utils.config_files.find("cvvdp_parameters.json", args.config_paths)
    params = utils.json2dict(params_file)
    csf = castleCSF(params["csf"], device, config_paths=args.config_paths)
    csf_sigma = torch.as_tensor(params["csf_sigma"], device=device)
    sens_corr = torch.as_tensor(params["sensitivity_correction"], device=device)

    # Prepare the output writer in sRGB 8-bit space
    writer = VideoWriter(args.output, fps=fps, verbose=args.verbose)
    orig_writer = VideoWriter("original_video.mp4", fps=fps, verbose=args.verbose)
    diff_writer = VideoWriter("diff_output.mp4", fps=fps, verbose=args.verbose)

    rgb_frame_max = []
    frame_mses = []

    rho_band = pyr.band_freqs

    for fi in range(n_frames):
        # Fetch the frame converted to DKL. This matches the metric's preprocessing
        frame = vs.get_test_frame(fi, device=device, colorspace="DKLd65")[0, :, 0]
        orig_rgb_lin = dkld65_to_rgb(frame.view(1, 3, 1, height, width))[0, :, 0]
        recon_channels = []
        for ci in range(3):
            # Decompose each chromatic channel separately using the Laplacian pyramid
            ch = frame[ci:ci+1]
            pair = ch.unsqueeze(0).repeat(2, 1, 1, 1)
            bands, log_bkg = pyr.decompose(pair)

            for bi, band in enumerate(bands):
                # CSF threshold for this band using the same model as in cvvdp_metric
                logL = log_bkg[bi][1:2]
                S = csf.sensitivity(rho_band[bi], 0, logL, ci, csf_sigma)
                S = S * (10.0 ** (sens_corr / 20.0))
                # Remove coefficients below alpha*threshold where threshold = 1/S
                thr = args.csf_strength / S
                bands[bi] = torch.where(torch.abs(band) >= thr, band, torch.zeros_like(band))

            # Reconstruct the pruned channel back to contrast domain
            recon_pair = reconstruct_from_contrast(pyr, bands, log_bkg)
            recon = recon_pair[0, 0]
            recon_channels.append(recon)

        dkl_recon = torch.stack(recon_channels, dim=0)
        rgb_lin = dkld65_to_rgb(dkl_recon.view(1, 3, 1, height, width))[0, :, 0]

        dm = vs.vs.dm_photometry
        Y_black, Y_refl = dm.get_black_level()
        Y_peak = dm.get_peak_luminance()
        # Convert absolute luminance back to [0,1] linear RGB
        lin_rel = (rgb_lin - (Y_black + Y_refl)) / (Y_peak - Y_black)
        lin_rel = lin_rel.clamp(0.0, 1.0)
        srgb = utils.linear2srgb_torch(lin_rel)

        orig_lin_rel = (orig_rgb_lin - (Y_black + Y_refl)) / (Y_peak - Y_black)
        orig_lin_rel = orig_lin_rel.clamp(0.0, 1.0)
        orig_srgb = utils.linear2srgb_torch(orig_lin_rel)

        # Quantise both images and measure integer-domain error
        srgb_uint8 = (srgb.clamp(0.0, 1.0) * 255.0).round().to(torch.int16)
        orig_srgb_uint8 = (orig_srgb.clamp(0.0, 1.0) * 255.0).round().to(torch.int16)
        
        diff_int = srgb_uint8 - orig_srgb_uint8
        mse = diff_int.float().pow(2).mean().item()
        frame_mses.append(mse)
        rgb_diff = diff_int.abs()

        print(f"Max value in srgb_uint8: {srgb_uint8.max().item()}, Max value in orig_srgb_uint8: {orig_srgb_uint8.max().item()}")

        frame_max = int(rgb_diff.max().item())
        rgb_frame_max.append(frame_max)
        logging.info(f"Frame {fi} RGB max error: {frame_max}, MSE: {mse}")

        # orig_writer.write_frame_rgb(orig_srgb_uint8.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
        diff_writer.write_frame_rgb((rgb_diff).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
        writer.write_frame_rgb(srgb_uint8.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
    orig_writer.write_frame_rgb(orig_srgb_uint8.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
    writer.close()
    orig_writer.close()
    diff_writer.close()
    if rgb_frame_max:
        overall_max = max(rgb_frame_max)
        avg_mse = sum(frame_mses) / len(frame_mses) if frame_mses else 0
    else:
        overall_max = 0
        avg_mse = 0
    logging.info(f"Processed {n_frames} frames")
    logging.info(f"Maximum RGB reconstruction error: {overall_max}")
    logging.info(f"Average MSE: {avg_mse}")


if __name__ == "__main__":
    main()
