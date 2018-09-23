CALL SETENV.BAT
start %PY_ROOT%\Scripts\tensorboard --logdir ./runs/%TLOG%/summaries/ --host 127.0.0.1
start http://127.0.0.1:6006


REM %PY_ROOT%\python eval.py --eval_train --checkpoint_dir="./runs/%TLOG%/checkpoints/"