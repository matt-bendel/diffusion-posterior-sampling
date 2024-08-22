import shutil

initial_im = 69000
final_im = 69999

while initial_im != final_im:
    src = f'/storage/FFHQ/ffhq256/0/{initial_im}.png'
    dst = f'/storage/FFHQ/ffhq256_firetest/0/{initial_im}.png'
    shutil.copyfile(src, dst)
    initial_im += 1
