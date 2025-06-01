import cv2
import numpy as np


def overlay_image(background, overlay, top=0, left=0):
    h, w = overlay.shape[:2]
    background[top:top+h, left:left+w] = overlay
    return background


def overlay_image_alpha(background, overlay_rgba, top=0, left=0):
    h, w = overlay_rgba.shape[:2]
    overlay_rgb = overlay_rgba[:, :, :3]
    alpha = overlay_rgba[:, :, 3:]
    assert np.max(alpha) <= 1.0, f"alpha channel must be no bigger than 1.0: {np.max(alpha)}"
    if np.max(overlay_rgba[:, :, 0]) <= 1.0:
        overlay_rgba[:, :, :3] = (overlay_rgba[:, :, :3] * 255).astype(np.uint8)

    bg_crop = background[top:top+h, left:left+w]
    blended = (1 - alpha) * bg_crop + alpha * overlay_rgb
    background[top:top+h, left:left+w] = blended.astype(np.uint8)
    return background


def overlay_resized_image(background, overlay, scale=0.5, top=0, left=0):
    """
    Downscale overlay and place it on the background at (top, left).

    Args:
        background (np.ndarray): HxWx3 image
        overlay (np.ndarray): hxwx3 or hxwx4 image (RGBA)
        scale (float): Fraction of background width the overlay should cover
        top, left (int): Position to place the overlay
    """
    bg_h, bg_w = background.shape[:2]

    # Calculate target size for overlay
    new_width = int(bg_w * scale)
    aspect_ratio = overlay.shape[0] / overlay.shape[1]
    new_height = int(new_width * aspect_ratio)

    # Resize overlay
    resized_overlay = cv2.resize(overlay, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Blend or place depending on whether overlay has alpha
    if resized_overlay.shape[2] == 4:
        result = overlay_image_alpha(background.copy(), resized_overlay, top, left)
    else:
        result = overlay_image(background.copy(), resized_overlay, top, left)

    return result