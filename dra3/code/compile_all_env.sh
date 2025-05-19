cd env/go
python setup.py build_ext --inplace --force

cd ../gobang
python setup.py build_ext --inplace --force

cd ../tictactoe
python setup.py build_ext --inplace --force