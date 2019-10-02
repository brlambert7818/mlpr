import numpy as np
import paramiko
import os.path
import tempfile

# access and open connect to host
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname='student.ssh.inf.ed.ac.uk',
                   username='s2000263',
                   password='Be@rlake7818')
ftp_client = ssh_client.open_sftp()

# how to run commands
stdin, stdout, stderr = ssh_client.exec_command('cd Desktop/ \n ls')
# print(stdout.readlines())

# create temp file to send data to
# dataTemp = tempfile.NamedTemporaryFile(prefix='matDataTemp')
# tempfile.tempdir = '~/Desktop/mlpr'

# access file from host
ftp_client.get('Desktop/dataTest.py', 'dataTest')
ftp_client.close()


