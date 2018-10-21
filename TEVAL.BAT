CALL SETENV.BAT
REM 
%PY_ROOT%\python eval.py --eval_train --batch_size=64 --checkpoint_dir="./runs/%TLOG%/checkpoints/" --data_file="./data/aclimdb_data_file_test.npz"

  