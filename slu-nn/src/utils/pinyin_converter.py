#!/usr/bin/env python3
#coding=utf-8
from pypinyin import pinyin, lazy_pinyin, Style
import os
import sys
import time

def pinyin_convert(line):
  pinyin_list = lazy_pinyin(line)
  return "".join(pinyin_list)

def pinyin_list_convert(list):
  return [pinyin_convert(w) for w in list]

def main(argv):
  if len(argv) < 2:
    print("""Usage: python3 pinyin_converter.py filename1 filename2 ...""")
  filelist = argv[1:]
  start = time.clock();
  for filepath in filelist:
    outfile = "./outputs/pinyin_" + os.path.split(filepath)[-1]
    with open(outfile, "w") as Out:
      with open(filepath, "r") as In:
        for line in In:
          line = pinyin_convert(line)
          # print(line)
          Out.write(line)
          # break
  end = time.clock()
  print(str(end-start) + "s\n")


if __name__ == "__main__":
  # main(sys.argv)
  print(pinyin_convert("我 是 谁 啊 你 又是 谁 啊"))
  print(pinyin_list_convert(['我', '是', '周董', '的', '歌迷']))