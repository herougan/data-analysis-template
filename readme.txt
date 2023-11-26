After downloading python 3.12, install the python extension in vs code.
Now, test hello.py. It should be able to run. If not, do "CTRL-SHIFT-P" and type "Python: Select Interpreter".
Select an interpreter. (Or select a venv you made - recommended! "CTRL-SHIFT-P" again and "Python: Create Environment")

Now configure your python terminal. While in hello.py, click the dropdown beside the play button. 
One of the options is "Run python file in dedicated terminal". That should reload your terminal properly, and now python commands should be available to you.

Now, in the python terminal do "pip install -r requirements.txt"
(do "pip freeze > requirements.txt" to put current installed modules into the text file)
(if pip doesn't work, try "python -m pip")