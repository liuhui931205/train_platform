# -*-coding:utf-8-*-
import scp
import os
from config import parallel_training, data_dest_data, data_sour_train
from Apps.utils.client import create_ssh_client


def upload_datas(data_name):
    data_path = os.path.join(data_dest_data, data_name)
    for dress in parallel_training:
        dest_ssh = create_ssh_client(dress["dest_scp_ip"], dress["dest_scp_port"], dress["dest_scp_user"],
                                     dress["dest_scp_passwd"])
        dest_scp = scp.SCPClient(dest_ssh.get_transport())
        dest_sftp = dest_ssh.open_sftp()
        files = dest_sftp.listdir(path=data_dest_data)
        if data_name not in files:
            dest_scp.put(data_path, data_path, recursive=True)
            dest_sftp.close()
            dest_scp.close()
            dest_ssh.close()


def upload_folder(task_name):
    task_path = os.path.join(data_sour_train, task_name)
    os.system("chown -R 1002:1002  {}".format(task_path))
    for dress in parallel_training:
        dest_ssh = create_ssh_client(dress["dest_scp_ip"], dress["dest_scp_port"], dress["dest_scp_user"],
                                     dress["dest_scp_passwd"])
        dest_scp = scp.SCPClient(dest_ssh.get_transport())
        dest_sftp = dest_ssh.open_sftp()
        files = dest_sftp.listdir(path=data_sour_train)
        if task_name not in files:
            dest_scp.put(task_path, task_path, recursive=True)
            dest_sftp.close()
            dest_scp.close()
            dest_ssh.close()


def rm_datas(data_name):
    data_path = os.path.join(data_dest_data, data_name)
    for dress in parallel_training:
        dest_ssh = create_ssh_client(dress["dest_scp_ip"], dress["dest_scp_port"], dress["dest_scp_user"],
                                     dress["dest_scp_passwd"])
        dest_scp = scp.SCPClient(dest_ssh.get_transport())
        dest_sftp = dest_ssh.open_sftp()
        files = dest_sftp.listdir(path=data_dest_data)
        if data_name in files:
            cmd = "rm -r {}".format(data_path)
            # dest_scp.put(data_path, data_path, recursive=True)
            dest_ssh.exec_command(cmd)
            dest_sftp.close()
            dest_scp.close()
            dest_ssh.close()


def rm_folder(task_name):
    task_path = os.path.join(data_sour_train, task_name)
    for dress in parallel_training:
        dest_ssh = create_ssh_client(dress["dest_scp_ip"], dress["dest_scp_port"], dress["dest_scp_user"],
                                     dress["dest_scp_passwd"])
        dest_scp = scp.SCPClient(dest_ssh.get_transport())
        dest_sftp = dest_ssh.open_sftp()
        files = dest_sftp.listdir(path=data_sour_train)
        if task_name in files:
            cmd = "rm -r {}".format(task_path)
            # dest_scp.put(task_path, task_path, recursive=True)
            dest_ssh.exec_command(cmd)
            dest_sftp.close()
            dest_scp.close()
            dest_ssh.close()
