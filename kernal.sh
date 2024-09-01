jupyter kernelspec list
# check out kernal list and get the currnet jupyter's path
# <current kernal's path> -m ipykernel install --prefix=$PWD/venv/share/jupyter/kernels/python3 --name 'myenv'
/opt/conda/bin/python3 -m ipykernel install --prefix=$PWD/venv --name 'myenv'