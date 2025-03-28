import torch
import torch.nn.functional as F


def random_resized_crop(x, out_size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                       interpolation='bicubic'):
    """
    Args:
        x: (T, 2, 3, H, W) where T == action_repeat (frame stack)
        out_size: int
        scale: tuple of float
        ratio: tuple of float
        interpolation: str
    Returns:
        x_out: (T, 2, 3, out_size, out_size)
    """
    t, n, c, h, w = x.shape
    assert h == w

    # random scale
    scale = torch.rand(1) * (scale[1] - scale[0]) + scale[0]
    # random ratio
    ratio = torch.rand(1) * (ratio[1] - ratio[0]) + ratio[0]

    # compute new size
    new_h = int(h * scale)
    new_w = int(new_h * ratio)

    # random crop
    i = torch.randint(0, h - new_h + 1, (1,))
    j = torch.randint(0, w - new_w + 1, (1,))

    # crop
    x_out = x[:, :, :, i:i + new_h, j:j + new_w]

    # resize
    x_out = F.interpolate(x_out.view(-1, c, new_h, new_w),
                         size=(out_size, out_size),
                         mode=interpolation,
                         align_corners=False)
    x_out = x_out.view(t, n, c, out_size, out_size)

    return x_out


def horizontal_flip(x, p=0.5):
    """
    Args:
        x: (T, 2, 3, H, W) where T == action_repeat (frame stack)
        p: float
    Returns:
        x_out: (T, 2, 3, H, W)
    """
    if torch.rand(1) < p:
        x_out = torch.flip(x, [4])
    else:
        x_out = x
    return x_out