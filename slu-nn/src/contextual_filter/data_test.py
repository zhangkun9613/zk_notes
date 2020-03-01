

if __name__ == "__main__":
  data_dir = "../data/train.txt"
  cnt, pre, cur = 0, 0, 0
  with open(data_dir, mode="r") as f:
    data = f.readlines()
    for n, line in enumerate(data):
      # line = line.split()
      print(n, line)
      if cnt == 0: pre = len(line)
      if cnt == 3: cur = len(line)
      if cnt == 1 or cnt == 2:
        if pre != len(line): print(n, line)
      if cnt == 4 or cnt == 5:
        if cur != len(line): print(n, line)
      if cnt == 6:
        if pre != len(line): print(n, line)
      cnt = (cnt + 1) % 7