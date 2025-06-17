from utils import parse_meta_paths
from arch import archs

if __name__ == '__main__':
    meta_path_strs = archs["DBLP"][0]
    meta_paths = [parse_meta_paths(s) for s in meta_path_strs if s]  # 过滤空字符串
    print(meta_paths)