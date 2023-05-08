import os
os.system('python build_folder_structure.py')
os.system('python or-baselines.py')

for i in range(1, 5):
    print('running for index '+str(i))
    os.system('python mle.py')
    os.system('python e2e-cal.py')
    os.system('python e2e-opl.py')