# Import libraries
from fabric import Connection
import pickle 


# list with values to be transferred, can also be an array
list = [5,1,7,1]

# opens a file in the working directory in which the list is put
with open('weights_put.pkl','wb') as f:
    pickle.dump(list, f)
f.close()

# connect laptop to PYNQ Z2, IP address might have to be changed depending on the situation
lappie = Connection("xilinx@192.168.137.76", connect_kwargs={"password": 'xilinx'})

# put the file in the working directory of the PYNQ Z2 (/home/xilinx/)
lappie.put('weights_put.pkl')

# Open terminal of PYNQ Z2, move to the right directory and execute the relevant python file
command ="""cd jupyter_notebooks
            cd Tests 
            python test_ssh.py"""
result = lappie.run(command)

# get the new files from /home/xilinx/<filename>, this directory has to be specified when making the file on the PYNQ Z2
lappie.get('/home/xilinx/weights_get.pkl')

# open the new file from the working directory
with open('weights_get.pkl', 'rb') as f:
    sum = pickle.load(f)
f.close()


print("sum is:", sum)
