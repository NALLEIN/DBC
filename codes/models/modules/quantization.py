import torch
import torch.nn as nn
import cv2
import numpy as np
import subprocess as sp
import torch.nn.functional as F
import av
from io import BytesIO
import math
from fractions import Fraction
import logging


class DecodeEncode(nn.Module):
    def __init__(self):
        super(DecodeEncode, self).__init__()

    def forward(self, input, qp, test):
        return ffmpeg_DE_yuv420.apply(input, qp, test)


class ffmpeg_DE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, qp, test):
        batch, channel, height, width = input.shape
        output = torch.zeros(input.shape, dtype=torch.float32).cuda()
        out_mv = []
        total_batch = batch
        total_frames = int(channel / 3)
        # HM_ENC = "/home/jianghao/Code/HM/HM-16.20+SCM-8.8/bin/TAppEncoderStatic"
        # HM_BIN = "/home/jianghao/Code/HM/HM-16.20+SCM-8.8/bin/TAppDecoderStatic"
        # av.logging.set_level(av.logging.DEBUG)
        with av.logging.Capture() as logs:
            for batch_i in range(total_batch):
                s = BytesIO()
                container = av.open(s, mode='w', format='hevc')
                stream = container.add_stream('libx265', rate=24)
                stream.width = width
                stream.height = height
                stream.pix_fmt = 'yuv444p'
                opts = {}
                if (test):
                    opts['x265-params'] = "ipratio=1:qp=" + qp
                else:
                    opts['x265-params'] = "log-level=0:ipratio=1:qp=" + qp
                stream.options = opts

                img = (torch.clamp(input[batch_i, :, :, :], 0, 1) * 255).reshape((total_frames, 3, height, width)).clone().detach().cpu().numpy()
                img = np.round(np.transpose(img, (0, 2, 3, 1))).astype(np.uint8)
                # y_tensor = torch.nn.functional.pixel_shuffle(img[:, :4, :, :], 2)[:, 0, :, :].cpu().numpy()
                # u_tensor = np.reshape(img[:, 4, :, :].cpu().numpy(), (total_frames, height // 2, width * 2))
                # v_tensor = np.reshape(img[:, 5, :, :].cpu().numpy(), (total_frames, height // 2, width * 2))
                # img = np.round(np.clip(np.concatenate((y_tensor, u_tensor, v_tensor), axis=1), 0, 1) * 255.).astype(np.uint8)
                for frame_i in range(total_frames):
                    frame = av.VideoFrame.from_ndarray(img[frame_i, :], format='rgb24').reformat(format='yuv444p')
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

                container.close()

                # with open("/home/jianghao/Code/Graduation/PreProcess/output.txt", "wb") as f:
                #     f.write(s.getbuffer())
                container = av.open(s, mode='r', format='hevc')
                stream = container.streams.video[0]
                stream.pix_fmt = 'yuv444p'
                stream.width = width
                stream.height = height
                img_decode = np.zeros(img.shape)
                frame_i = 0
                for packet in container.demux():
                    for frame in packet.decode():
                        img_decode[frame_i, :, :, :] = frame.reformat(format='rgb24').to_ndarray()
                        frame_i += 1
                img_cuda = np.array(img_decode).astype(np.float32) / 255.
                img_cuda = torch.tensor(img_cuda).cuda()
                img_cuda = img_cuda.view(-1, height, width, 3).permute(0, 3, 1, 2)
                output[batch_i, :, :, :] = img_cuda
                container.close()
                # command_out = [HM_BIN, '-b', "/home/jianghao/Code/Graduation/PreProcess/output.txt"]
                # sp.run(command_out)

                # mv_lcu_height = math.ceil(height/64)
                # mv_lcu_width = math.ceil(width/64)
                # mv_cu_height = mv_lcu_height*16
                # mv_cu_width = mv_lcu_width*16

                # motion_vectors = np.fromfile("/home/gy4qf62/Invertible-Image-Rescaling/temp_MV", dtype='int16').reshape((total_frames,mv_lcu_height,mv_lcu_width,16,16,2))
                # motion_vectors = np.transpose(motion_vectors,(0,5,1,3,2,4)).reshape((total_frames*2,mv_cu_height,mv_cu_width))
                # out_mv.append(motion_vectors)
            out_mv = torch.tensor(out_mv).cuda()
            if (test):
                logger = logging.getLogger('val')
                for log in logs:
                    if ("kb" in log[2]):
                        logger.info(log[2])
        return output, out_mv

    @staticmethod
    def backward(ctx, grad_output, grad_output2):
        return grad_output, None, None


class ffmpeg_DE_yuv420(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, qp, test):
        batch, channel, height, width = input.shape
        output = torch.zeros(input.shape).cuda()
        out_mv = []
        total_batch = batch
        total_frames = int(channel / 6)
        # HM_ENC = "/home/jianghao/Code/HM/HM-16.20+SCM-8.8/bin/TAppEncoderStatic"
        # HM_BIN = "/home/jianghao/Code/HM/HM-16.20+SCM-8.8/bin/TAppDecoderStatic"
        #av.logging.set_level(av.logging.DEBUG)
        with av.logging.Capture() as logs:
            for batch_i in range(total_batch):
                s = BytesIO()
                container = av.open(s, mode='w', format='hevc')
                stream = container.add_stream('libx265', rate=24)
                stream.width = width * 2
                stream.height = height * 2
                stream.pix_fmt = 'yuv420p'
                opts = {}
                if (test):
                    opts['x265-params'] = "ipratio=1:qp=" + qp
                else:
                    opts['x265-params'] = "log-level=0:ipratio=1:qp=" + qp
                stream.options = opts
                img = input[batch_i, :, :, :].reshape((total_frames, 6, height, width)).clone().detach()
                y_tensor = torch.nn.functional.pixel_shuffle(img[:, :4, :, :], 2)[:, 0, :, :].cpu().numpy()
                u_tensor = np.reshape(img[:, 4, :, :].cpu().numpy(), (total_frames, height // 2, width * 2))
                v_tensor = np.reshape(img[:, 5, :, :].cpu().numpy(), (total_frames, height // 2, width * 2))
                img = np.round(np.clip(np.concatenate((y_tensor, u_tensor, v_tensor), axis=1), 0, 1) * 255.).astype(np.uint8)

                for frame_i in range(total_frames):
                    frame = av.VideoFrame.from_ndarray(img[frame_i, :, :], format='yuv420p')
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

                container.close()

                # with open("/home/jianghao/Code/Graduation/PreProcess/output.txt", "wb") as f:
                #     f.write(s.getbuffer())
                container = av.open(s, mode='r', format='hevc')
                stream = container.streams.video[0]
                stream.pix_fmt = 'yuv420p'
                stream.width = width * 2
                stream.height = height * 2
                img_decode = np.zeros(img.shape)
                frame_i = 0
                for packet in container.demux():
                    for frame in packet.decode():
                        img_decode[frame_i, :, :] = frame.reformat(format='yuv420p').to_ndarray()
                        frame_i += 1
                img_cuda = np.array(img_decode).astype(np.float32) / 255.
                img_cuda = torch.tensor(img_cuda).cuda()
                y_tensor = torch.nn.functional.pixel_unshuffle(img_cuda[:, :height * 2, :].view(total_frames, 1, height * 2, width * 2), 2)
                u_tensor = img_cuda[:, height * 2:height * 2 + height // 2, :].view(total_frames, 1, height, width)
                v_tensor = img_cuda[:, height * 2 + height // 2:, :].view(total_frames, 1, height, width)
                output[batch_i, :, :, :] = torch.cat((y_tensor, u_tensor, v_tensor), 1)
                container.close()

                # command_out = [HM_BIN, '-b', "/home/jianghao/Code/Graduation/PreProcess/output.txt"]
                # sp.run(command_out)

                # mv_lcu_height = math.ceil(height/64)
                # mv_lcu_width = math.ceil(width/64)
                # mv_cu_height = mv_lcu_height*16
                # mv_cu_width = mv_lcu_width*16

                # motion_vectors = np.fromfile("/home/gy4qf62/Invertible-Image-Rescaling/temp_MV", dtype='int16').reshape((total_frames,mv_lcu_height,mv_lcu_width,16,16,2))
                # motion_vectors = np.transpose(motion_vectors,(0,5,1,3,2,4)).reshape((total_frames*2,mv_cu_height,mv_cu_width))
                # out_mv.append(motion_vectors)
            out_mv = torch.tensor(out_mv).cuda()
            if (test):
                logger = logging.getLogger('val')
                for log in logs:
                    if ("kb" in log[2]):
                        logger.info(log[2])
        return output, out_mv

    @staticmethod
    def backward(ctx, grad_output, grad_output2):
        return grad_output, None, None


def rgb2yuv420(rgb):
    b, c, height, width = rgb.shape  # c = 3
    rgb = (torch.clamp(rgb, 0, 1) * 255).clone().detach().cpu().numpy()
    rgb = np.round(np.transpose(rgb, (0, 2, 3, 1))).astype(np.uint8)
    out = torch.zeros((b, 2 * c, height // 2, width // 2)).cuda()
    for i in range(b):
        yuv420p = (av.VideoFrame.from_ndarray(rgb[i, :, :, :], format='rgb24')).reformat(format='yuv420p').to_ndarray()
        img_cuda = np.array(yuv420p).astype(np.float32) / 255.
        img_cuda = torch.tensor(img_cuda).cuda()
        y_tensor = torch.nn.functional.pixel_unshuffle(img_cuda[:height, :].view(1, 1, height, width), 2)
        u_tensor = img_cuda[height:height + height // 4, :].view(1, 1, height // 2, width // 2)
        v_tensor = img_cuda[height + height // 4:, :].view(1, 1, height // 2, width // 2)
        out[i, :] = torch.cat((y_tensor, u_tensor, v_tensor), 1)
    return out


def yuv4202rgb(yuv):
    b, c, height, width = yuv.shape  # c = 6
    out = torch.zeros((b, c // 2, height * 2, width * 2)).cuda()
    for i in range(b):
        img = yuv[i, :, :, :].view(1, 6, height, width).clone().detach()
        y_tensor = torch.nn.functional.pixel_shuffle(img[:, :4, :, :], 2)[0, 0, :, :].cpu().numpy()
        u_tensor = np.reshape(img[0, 4, :, :].cpu().numpy(), (height // 2, width * 2))
        v_tensor = np.reshape(img[0, 5, :, :].cpu().numpy(), (height // 2, width * 2))
        img = np.round(np.clip(np.concatenate((y_tensor, u_tensor, v_tensor), axis=0), 0, 1) * 255.).astype(np.uint8)
        rgb = (av.VideoFrame.from_ndarray(img, format='yuv420p')).reformat(format='rgb24').to_ndarray()
        img_cuda = np.array(rgb).astype(np.float32) / 255.
        img_cuda = torch.tensor(img_cuda).cuda().view(1, height * 2, width * 2, 3).permute(0, 3, 1, 2)
        out[i, :] = img_cuda.view(1, 3, height * 2, width * 2)
    return out


def rgbdown(rgb, scalea, scaleb):
    b, c, height, width = rgb.shape
    h = int(height * scalea / scaleb)
    w = int(width * scalea / scaleb)

    out = torch.zeros((b, c, h, w)).cuda()

    rgb = (torch.clamp(rgb, 0, 1) * 255).clone().detach().cpu().numpy()
    rgb = np.round(np.transpose(rgb, (0, 2, 3, 1))).astype(np.uint8)
    for i in range(b):
        frame = (av.VideoFrame.from_ndarray(rgb[i, :, :, :], format='rgb24'))
        down = frame.reformat(width=w, height=h, interpolation='LANCZOS').to_ndarray()
        img_cuda = np.array(down).astype(np.float32) / 255.
        img_cuda = torch.tensor(img_cuda).cuda().view(h, w, 3).permute(2, 0, 1)
        out[i, :] = img_cuda
    return out


def rgb2yuv420_down(rgb, scale):
    b, c, height, width = rgb.shape
    assert c == 3, 'channel dose not match'
    rgb = (torch.clamp(rgb, 0, 1) * 255).cpu().numpy()
    rgb = np.round(np.transpose(rgb, (0, 2, 3, 1))).astype(np.uint8)

    out = torch.zeros((b, 2 * c, height // 2, width // 2)).cuda()
    down = torch.zeros((b, 2 * c, height // 4, width // 4)).cuda()
    for i in range(b):
        frame = av.VideoFrame.from_ndarray(rgb[i, :, :, :], format='rgb24').reformat(format='yuv420p')
        yuv420p = frame.to_ndarray()
        img_cuda = np.array(yuv420p).astype(np.float32) / 255.
        img_cuda = torch.tensor(img_cuda).cuda()
        y_tensor = torch.nn.functional.pixel_unshuffle(img_cuda[:height, :].view(1, 1, height, width), 2)
        u_tensor = img_cuda[height:height + height // 4, :].view(1, 1, height // 2, width // 2)
        v_tensor = img_cuda[height + height // 4:, :].view(1, 1, height // 2, width // 2)
        out[i, :] = torch.cat((y_tensor, u_tensor, v_tensor), 1).squeeze(0)

        h = int(height * scale)
        w = int(width * scale)
        yuv_down = frame.reformat(width=w, height=h, interpolation='LANCZOS').to_ndarray()
        img_down = np.array(yuv_down).astype(np.float32) / 255.
        img_down = torch.tensor(img_down).cuda()
        y = torch.nn.functional.pixel_unshuffle(img_down[:h, :].view(1, 1, h, w), 2)
        u = img_down[h:h + h // 4, :].view(1, 1, h // 2, w // 2)
        v = img_down[h + h // 4:, :].view(1, 1, h // 2, w // 2)
        down[i, :] = torch.cat((y, u, v), 1).squeeze(0)
    return out, down


def cubic(x):
    absx = np.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))


def lanczos3(x):
    return (np.sinc(x) * np.sinc(x / 3) * (abs(x) < 3))


def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


def kernel_info(name=None):
    method, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0)
    }.get(name)
    return method, kernel_width


def fix_scale_and_size(input_shape, output_shape, scale_factor):
    # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
    # same size as the number of input dimensions)
    if scale_factor is not None:
        # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]

        # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
    # to all the unspecified dimensions
    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(input_shape[len(output_shape):])

    # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
    # sub-optimal, because there can be different scales to the same output-shape.
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

    # Dealing with missing output-shape. calculating according to scale-factor
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

    return scale_factor, output_shape


def cubic_down(s):
    if s == 2:
        b = 1.5
        B = np.abs(np.linspace(-b, b, 4))
    elif s == 3:
        b = 5 / float(3)
        B = np.linspace(-b, b, 11)
    elif s == 4:
        b = 15 / float(8)
        B = np.linspace(-b, b, 16)

    A = cubic(B)

    return A / np.sum(A)


def lanczos_down(s):
    if s == 2:
        b = 2.75
        B = np.abs(np.linspace(-b, b, 12))
    elif s == 3:
        b = 5 / float(3)
        B = np.linspace(-b, b, 11)
    elif s == 4:
        b = 15 / float(8)
        B = np.linspace(-b, b, 16)

    A = lanczos3(B)

    return A / np.sum(A)


def cubic_up(s):
    A = np.zeros((s, 5), 'float32')
    delta = 1 / float(s)
    k = [-(s - 1) / float(s) / 2]
    for i in range(0, s - 1):
        k.append(k[-1] + delta)
    for i, b in enumerate(k[::-1]):
        B = np.array([b - 2, b - 1, b, b + 1, b + 2])
        A[i] = cubic(B)
        A[i] /= np.sum(A[i])

    return A


def lanczos_up(s):
    A = np.zeros((s, 7), 'float32')
    delta = 1 / float(s)
    k = [-(s - 1) / float(s) / 2]
    for i in range(0, s - 1):
        k.append(k[-1] + delta)
    for i, b in enumerate(k[::-1]):
        B = np.array([b - 3, b - 2, b - 1, b, b + 1, b + 2, b + 3])
        A[i] = lanczos3(B)
        A[i] /= np.sum(A[i])

    return A


def channel_shuffle(x, scale):
    assert len(x.size()) == 4, 'dimention not match'
    n, c, h, w = x.size()
    out = x[:, 0:1, :, :].repeat(1, scale**2, 1, 1)
    for i in range(1, c):
        tmp = x[:, i:i + 1, :, :].repeat(1, scale**2, 1, 1)
        out = torch.stack((out, tmp), dim=1)
    out = torch.nn.functional.pixel_shuffle(out, scale)
    return out


def cubic_up(s):
    A = np.zeros((s, 5), 'float32')
    delta = 1 / float(s)
    k = [-(s - 1) / float(s) / 2]
    for i in range(0, s - 1):
        k.append(k[-1] + delta)
    for i, b in enumerate(k[::-1]):
        B = np.array([b - 2, b - 1, b, b + 1, b + 2])
        A[i] = cubic(B)
        A[i] /= np.sum(A[i])

    return A


def lanczos_up(s):
    A = np.zeros((s, 7), 'float32')
    delta = 1 / float(s)
    k = [-(s - 1) / float(s) / 2]
    for i in range(0, s - 1):
        k.append(k[-1] + delta)
    for i, b in enumerate(k[::-1]):
        B = np.array([b - 3, b - 2, b - 1, b, b + 1, b + 2, b + 3])
        A[i] = lanczos3(B)
        A[i] /= np.sum(A[i])

    return A


class DownSample(nn.Module):
    def __init__(self, scale: int):
        assert scale in [2, 3, 4], "scale must be 2, 3, or 4!"
        super(DownSample, self).__init__()
        self.scale = scale
        self._get_kernel()
        pad = (self.weights_x.size(-1) + 1) // 2 - 1 - (scale - 1) // 2
        self.padder = nn.ReplicationPad2d(pad)

    def _get_kernel(self):
        weights = lanczos_down(self.scale)
        weights = torch.FloatTensor(weights).cuda()
        self.register_buffer("weights_x", weights.view(1, 1, 1, -1))
        self.register_buffer("weights_y", weights.view(1, 1, -1, 1))

    def forward(self, x):
        x = self.padder(x)
        b, c, h, w = x.size()
        x = x.view(b * c, 1, h, w)
        x = F.conv2d(x, self.weights_x, None, (1, self.scale), 0, 1, 1)
        x = F.conv2d(x, self.weights_y, None, (self.scale, 1), 0, 1, 1)
        x = x.view(b, c, x.size(2), x.size(3))

        return x


class UpSample(nn.Module):
    def __init__(self, scale: int):
        assert scale in [2, 3, 4], "scale must be 2, 3, or 4!"
        super(UpSample, self).__init__()
        self.scale = scale
        self._get_kernel()
        # the width of cubic kernel for up sample is always 5.
        self.padder = nn.ReplicationPad2d(3)

    def _get_kernel(self):
        w = lanczos_up(self.scale)  # (scale, width)
        weights = []
        for i in range(self.scale):
            for j in range(self.scale):
                weights.append(np.outer(w[i], w[j]))
        weights = np.stack(weights, 0)
        weights = torch.FloatTensor(weights).cuda()
        weights = weights.view(self.scale**2, 1, w.shape[1], w.shape[1])
        self.register_buffer("weights", weights)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * c, 1, h, w)
        x = self.padder(x)
        x = F.conv2d(x, self.weights, None, 1, 0, 1, 1)
        x = F.pixel_shuffle(x, self.scale)
        x = x.view(b, c, h * self.scale, w * self.scale)

        return x


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np.clip(0, 1) * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def codecImage(_tensor, qp=22, path="/home/jianghao/Code/Graduation/4k1/test"):
    """
    Use for test batch=1
    _tensor : 1CHW
    """
    rgb = tensor2img(_tensor.squeeze(0))
    image1 = osp.join(path, 'tmp.png')
    mkv = osp.join(path, 'tmp.mkv')
    image2 = osp.join(path, 'codec.png')

    cv2.imwrite(osp.join(path, 'tmp.png'), rgb)
    cmd_in = "/home/jianghao/opt/ffmpeg/bin/ffmpeg -i %s -c:v libx265 -x265-params \"qp=%d\" %s -y" % (image1, qp, mkv)
    sp.call(cmd_in, shell=True)
    cmd_out = "/home/jianghao/opt/ffmpeg/bin/ffmpeg -i %s -r 1 -f image2 %s -y" % (mkv, image2)
    sp.call(cmd_out, shell=True)

    img = cv2.imread(image2, cv2.IMREAD_COLOR)
    img = img.astype(np.float32) / 255.
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    img = img.unsqueeze(0).cuda()
    return img