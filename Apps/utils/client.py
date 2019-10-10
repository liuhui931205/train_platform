# -*- coding:utf-8 -*-
import paramiko
import scp
from config import ssh_dress11, ssh_dress12, ssh_dress13, ssh_dress14, ssh_dress15, ssh_dress00


def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def client(id=None):
    if id == '161':
        dest_scp_ip = ssh_dress11['dest_scp_ip']
        dest_scp_port = ssh_dress11['dest_scp_port']
        dest_scp_user = ssh_dress11['dest_scp_user']
        dest_scp_passwd = ssh_dress11['dest_scp_passwd']
    elif id == '162':
        dest_scp_ip = ssh_dress12['dest_scp_ip']
        dest_scp_port = ssh_dress12['dest_scp_port']
        dest_scp_user = ssh_dress12['dest_scp_user']
        dest_scp_passwd = ssh_dress12['dest_scp_passwd']
    elif id == '163':
        dest_scp_ip = ssh_dress13['dest_scp_ip']
        dest_scp_port = ssh_dress13['dest_scp_port']
        dest_scp_user = ssh_dress13['dest_scp_user']
        dest_scp_passwd = ssh_dress13['dest_scp_passwd']
    elif id == '164':
        dest_scp_ip = ssh_dress14['dest_scp_ip']
        dest_scp_port = ssh_dress14['dest_scp_port']
        dest_scp_user = ssh_dress14['dest_scp_user']
        dest_scp_passwd = ssh_dress14['dest_scp_passwd']
    elif id == '165':
        dest_scp_ip = ssh_dress15['dest_scp_ip']
        dest_scp_port = ssh_dress15['dest_scp_port']
        dest_scp_user = ssh_dress15['dest_scp_user']
        dest_scp_passwd = ssh_dress15['dest_scp_passwd']
    elif id == 'host':
        dest_scp_ip = ssh_dress00['dest_scp_ip']
        dest_scp_port = ssh_dress00['dest_scp_port']
        dest_scp_user = ssh_dress00['dest_scp_user']
        dest_scp_passwd = ssh_dress00['dest_scp_passwd']

    dest_ssh = create_ssh_client(
        server=dest_scp_ip,
        port=dest_scp_port,
        user=dest_scp_user,
        password=dest_scp_passwd
    )
    dest_scp = scp.SCPClient(dest_ssh.get_transport())
    dest_sftp = dest_ssh.open_sftp()
    return dest_scp, dest_sftp, dest_ssh
