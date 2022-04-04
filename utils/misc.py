from glob import glob 

def get_stl_num(style_dir):
    stl_num = len(glob(f"{style_dir}/*"))
    return stl_num