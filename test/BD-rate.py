import numpy as np
import scipy.interpolate
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib


def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2 - int1) / (max_int - min_int)

    return avg_diff


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100
    return avg_diff


parser = argparse.ArgumentParser()
parser.add_argument('-savepath', type=str, required=True, help='path of current sequence logs.')
parser.add_argument('-path1', type=str, required=True, help='path of current sequence logs.')
parser.add_argument('-path2', type=str, required=True, help='path of current sequence logs.')
args = parser.parse_args()

bpp1 = []
bpp2 = []
psnr1 = []
psnr2 = []
ssim1 = []
ssim2 = []
f1 = open(args.path1, mode='r')
lines1 = f1.readlines()
for i in range(0, len(lines1), 3):
    psnr1.append(float(lines1[i].split('\n')[0]))
    ssim1.append(float(lines1[i + 1].split('\n')[0]))
    bpp1.append(float(lines1[i + 2].split('\n')[0]))

f2 = open(args.path2, mode='r')
lines2 = f2.readlines()
for i in range(0, len(lines2), 3):
    psnr2.append(float(lines2[i].split('\n')[0]))
    ssim2.append(float(lines2[i + 1].split('\n')[0]))
    bpp2.append(float(lines2[i + 2].split('\n')[0]))

print('BD-PSNR: ', BD_PSNR(bpp1, psnr1, bpp2, psnr2))
print('BD-RATE: ', BD_RATE(bpp1, psnr1, bpp2, psnr2))

plt.clf()
oursmse, = plt.plot(bpp1, psnr1, color='lightblue', marker='o', markersize=4, label="bdbr1")
oursmse, = plt.plot(bpp2, psnr2, color='brown', marker='o', markersize=4, label="bdbr2")

font = {'family': 'serif', 'weight': 'normal', 'size': 8}
matplotlib.rc('font', **font)
LineWidth = 1
plt.grid()
plt.xlabel('bpp')
plt.ylabel('PSNR(dB)')
plt.legend()
plt.savefig(args.savepath + "/" + "_PSNR.png", format='png', dpi=300, bbox_inches='tight')

