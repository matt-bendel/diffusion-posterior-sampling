import shutil

initial_im = 69007
final_im = 69997

while initial_im != final_im:
    src = f'/storage/FFHQ/ffhq256/{initial_im}.png'
    dst = f'/storage/FFHQ/ffhq256_diffpirtest/{initial_im}.png'
    shutil.copyfile(src, dst)
    initial_im += 10
