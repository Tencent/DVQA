import os
import sys
import re


def decode_bitstreams(src_dir, dst_dir, rule=None):

    cmd_to_exec = []
    for dirpath, dirs, files in os.walk(src_dir, topdown=False):
        for f in files:
            if not f.startswith('.') and f.endswith('.mp4'):
                src_path = os.path.join(src_dir, f)
                dst_path = os.path.join(dst_dir, re.sub('.mp4', '.yuv', f))
                print('decoding {f}'.format(f=src_path))
                cmd= "ffmpeg -i " + src_path + " -f rawvideo -pix_fmt yuv420p -an -vsync 0 -y " + dst_path
                cmd_to_exec.append(cmd)
                print(cmd)

    with open('decoding.sh', 'wt') as f:
        f.write("#!/bin/sh\n\n")
        for item in cmd_to_exec:
            f.write("%s\n" % item)


if __name__ == "__main__":

    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    decode_bitstreams(src_dir, dst_dir)
