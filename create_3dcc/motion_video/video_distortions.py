#From https://github.com/Newbeeyoung/Video-Corruption-Robustness/blob/main/create_mini_ssv2_c.py

import subprocess
import pdb
import json

# Whole Video
def bit_error(src,dst,severity, img_path):
    c=[100000, 50000, 30000, 20000, 10000][severity-1]
    return_code = subprocess.run(
        ["ffmpeg","-y", "-i", src, "-c", "copy", "-bsf", "noise={}".format(str(c)),
         dst, "-loglevel",  "quiet"])

    return_code = subprocess.run(
        ["ffmpeg","-y", "-i", dst, "./motion_video/tmp/input%05d.png", "-loglevel",  "quiet" ])

    return_code = subprocess.run([
        "mv", "./motion_video/tmp/input00120.png", img_path
    ])

    return return_code

def h264_crf(src,dst,severity):
    c=[23,30,37,44,51][severity-1]
    return_code = subprocess.call(
        ["ffmpeg", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx264", "-crf", str(c), dst])

    return return_code

def h264_abr(src,dst,severity):

    c=[2,4,8,16,32][severity-1]
    result = subprocess.Popen(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", src],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    data = json.load(result.stdout)

    bit_rate = str(int(float(data['format']['bit_rate']) / c))

    return_code = subprocess.call(
        ["ffmpeg","-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx264", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize",
         bit_rate, dst])

    return return_code

def h265_crf(src, dst, severity, img_path):
    c = [24, 26, 32, 39, 41][severity - 1] #[27, 33, 39, 45, 51][severity - 1]
    return_code = subprocess.call(
        ["ffmpeg", "-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'","-vcodec", "libx265", "-loglevel",  "error", "-crf", str(c), dst, "-loglevel",  "0", "-nostats"])

    return_code = subprocess.run(
        ["ffmpeg","-y", "-i", dst, "./motion_video/tmp/input%05d.png", "-loglevel",  "0", "-nostats"])

    return_code = subprocess.run([
        "mv", "./motion_video/tmp/input00120.png", img_path
    ])

    return return_code

def h265_abr(src, dst, severity, img_path):
    c = [2, 4, 8, 16, 32][severity - 1]
    result = subprocess.Popen(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", src],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    data = json.load(result.stdout)

    bit_rate = str(int(float(data['format']['bit_rate']) / c))

    return_code = subprocess.call(
        ["ffmpeg", "-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx265", "-loglevel",  "error", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize",
         bit_rate, dst, "-loglevel",  "0", "-nostats"])

    return_code = subprocess.run(
        ["ffmpeg","-y", "-i", dst, "./motion_video/tmp/input%05d.png", "-loglevel",  "0", "-nostats"])

    return_code = subprocess.run([
        "mv", "./motion_video/tmp/input00120.png", img_path
    ])


    return return_code
