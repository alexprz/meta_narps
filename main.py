from meta_main import run_meta


filenames = [f'hypo{i}_thresh.nii.gz' for i in range(1, 10)]

for filename in filenames:
    print(filename)
    run_meta(filename, verbose=False)
