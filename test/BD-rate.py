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
parser.add_argument('-path', type=str, required=True, help='path of current sequence logs.')
args = parser.parse_args()

mode = ["y", "u", "v", "average"]
ssimmode = ["Y", "U", "V", "All"]
RateNet = []
Ratelanczos = []
PSNRNet = {}
PSNRlanczos = {}
SSIMNet = {}
SSIMlanczos = {}

for m in mode:
    PSNRNet[m] = []
    PSNRlanczos[m] = []
for m in ssimmode:
    SSIMNet[m] = []
    SSIMlanczos[m] = []

for qp in range(22, 38, 5):
    f = open(args.path + "/Avg_psnr_net_" + str(qp), mode='r')
    lines = f.readlines()
    line = lines[-1]
    for m in mode:
        PSNRNet[m].append(float(re.search(m + ":(\d+(\.\d+))", line).group(1)))

    f = open(args.path + "/Avg_ssim_net_" + str(qp), mode='r')
    lines = f.readlines()
    line = lines[-1]
    for m in ssimmode:
        SSIMNet[m].append(float(re.search(m + ":(\d+(\.\d+))", line).group(1)))

    f = open(args.path + "/transcode_net_" + str(qp), mode='r')
    lines = f.readlines()
    # line = lines[-1]
    # RateNet.append(float(re.search("(\d+(\.\d+)) kb", line).group(1)))
    line = lines[-4]
    RateNet.append(float(re.search("a\s+(\d+(\.\d+))", line).group(1)))

    f = open(args.path + "/Avg_psnr_lanczos_" + str(qp), mode='r')
    lines = f.readlines()
    line = lines[-1]
    for m in mode:
        PSNRlanczos[m].append(float(re.search(m + ":(\d+(\.\d+))", line).group(1)))

    f = open(args.path + "/Avg_ssim_lanczos_" + str(qp), mode='r')
    lines = f.readlines()
    line = lines[-1]
    for m in ssimmode:
        SSIMlanczos[m].append(float(re.search(m + ":(\d+(\.\d+))", line).group(1)))

    f = open(args.path + "/transcode_lanczos_" + str(qp), mode='r')
    lines = f.readlines()
    # line = lines[-1]
    # Ratelanczos.append(float(re.search("(\d+(\.\d+)) kb", line).group(1)))
    line = lines[-4]
    Ratelanczos.append(float(re.search("a\s+(\d+(\.\d+))", line).group(1)))

print("Test on " + args.path)
for m in mode:
    print("Evaluation on " + m)
    print('BD-PSNR: ', BD_PSNR(Ratelanczos, PSNRlanczos[m], RateNet, PSNRNet[m]))
    print('BD-RATE: ', BD_RATE(Ratelanczos, PSNRlanczos[m], RateNet, PSNRNet[m]))

    plt.clf()
    oursmse, = plt.plot(RateNet, PSNRNet[m], color='lightblue', marker='o', markersize=4, label="net")
    oursmse, = plt.plot(Ratelanczos, PSNRlanczos[m], color='brown', marker='o', markersize=4, label="lanczos")

    font = {'family': 'serif', 'weight': 'normal', 'size': 8}
    matplotlib.rc('font', **font)
    LineWidth = 1
    plt.grid()
    plt.xlabel('kbps')
    plt.ylabel('PSNR(dB)')
    plt.legend()
    plt.savefig(args.path + "/" + m + "_PSNR.png", format='png', dpi=300, bbox_inches='tight')
for m in ssimmode:
    plt.clf()
    oursmse, = plt.plot(RateNet, SSIMNet[m], color='lightblue', marker='o', markersize=4, label="net")
    oursmse, = plt.plot(Ratelanczos, SSIMlanczos[m], color='brown', marker='o', markersize=4, label="lanczos")

    font = {'family': 'serif', 'weight': 'normal', 'size': 8}
    matplotlib.rc('font', **font)
    LineWidth = 1
    plt.grid()
    plt.xlabel('kbps')
    plt.ylabel('SSIM')
    plt.legend()
    plt.savefig(args.path + "/" + m + "_SSIM.png", format='png', dpi=300, bbox_inches='tight')
