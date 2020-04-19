##Command to output running processes
WMIC /OUTPUT:C:\ProcessList.txt path win32_process get Caption,Processid,Commandline

####Kill multiple pids in one go
taskkill /pid 1108 /pid 1300 /pid 5468 /pid 5892 /pid 1344 /F

##Activate conda from command line
C:\ProgramData\Anaconda3\Scripts\activate.bat C:\Users\istoffa\.conda\envs\mlEnv
###or
path=%path%;C:\ProgramData\Anaconda3\Scripts\

conda init cmd.exe

cmd
###Run jupyter notebook remotely < this worked so far

path=%path%;C:\ProgramData\Anaconda3\Scripts\

conda activate mlEnv

start jupyter lab --NotebookApp.token='' --no-browser --port=8888

###or attach to same window with start /b and use folliwing to hide
ShowWindow(GetConsoleWindow(), SW_HIDE) & exit

https://superuser.com/a/1069983/377667

##Then ssh with tunneling
server to client - 8000 localhost:8888, port 8000 is local
