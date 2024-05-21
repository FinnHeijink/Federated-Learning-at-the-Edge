start cmd.exe /c src\\RunServerWrapped.bat
timeout 5 > NUL
FOR /L %%a IN (1,1,%1) DO start cmd.exe /c src\\RunClientWrapped.bat