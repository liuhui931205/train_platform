import paramiko
import scp

server = {
    "165": {
        "docker": {
            "dest_scp_ip": "10.11.5.165",
            "dest_scp_port": 76,
            "dest_scp_user": "root",
            "dest_scp_passwd": "12345678"
        },
        "host": {
            "dest_scp_ip": "10.11.5.165",
            "dest_scp_port": 22,
            "dest_scp_user": "hadoop",
            "dest_scp_passwd": "hadoop"
        },
    },
    "161": {
        "docker": {
            "dest_scp_ip": "10.11.5.161",
            "dest_scp_port": 76,
            "dest_scp_user": "root",
            "dest_scp_passwd": "1234567890"
        },
        "host": {
            "dest_scp_ip": "10.11.5.161",
            "dest_scp_port": 22,
            "dest_scp_user": "kdreg",
            "dest_scp_passwd": "kd-123"
        },
    },
    "162": {
        "docker": {
            "dest_scp_ip": "10.11.5.162",
            "dest_scp_port": 76,
            "dest_scp_user": "root",
            "dest_scp_passwd": "1234567890"
        },
        "host": {
            "dest_scp_ip": "10.11.5.162",
            "dest_scp_port": 22,
            "dest_scp_user": "kdreg",
            "dest_scp_passwd": "kd-123"
        },
    },
    "163": {
        "docker": {
            "dest_scp_ip": "10.11.5.163",
            "dest_scp_port": 76,
            "dest_scp_user": "root",
            "dest_scp_passwd": "1234567890"
        },
        "host": {
            "dest_scp_ip": "10.11.5.163",
            "dest_scp_port": 22,
            "dest_scp_user": "kdreg",
            "dest_scp_passwd": "kd-123"
        },
    },
    "164": {
        "docker": {
            "dest_scp_ip": "10.11.5.164",
            "dest_scp_port": 76,
            "dest_scp_user": "root",
            "dest_scp_passwd": "1234567890"
        },
        "host": {
            "dest_scp_ip": "10.11.5.164",
            "dest_scp_port": 22,
            "dest_scp_user": "kdreg",
            "dest_scp_passwd": "kd-123"
        },
    },
    "241": {
        "docker": {
            "dest_scp_ip": "10.11.5.241",
            "dest_scp_port": 77,
            "dest_scp_user": "root",
            "dest_scp_passwd": "12345678"
        }
    }
}


def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def client(ids=None, types="host"):
    if ids == '161':

        dest_scp_ip = server[ids][types]['dest_scp_ip']
        dest_scp_port = server[ids][types]['dest_scp_port']
        dest_scp_user = server[ids][types]['dest_scp_user']
        dest_scp_passwd = server[ids][types]['dest_scp_passwd']
    elif ids == '162':
        dest_scp_ip = server[ids][types]['dest_scp_ip']
        dest_scp_port = server[ids][types]['dest_scp_port']
        dest_scp_user = server[ids][types]['dest_scp_user']
        dest_scp_passwd = server[ids][types]['dest_scp_passwd']
    elif ids == '163':
        dest_scp_ip = server[ids][types]['dest_scp_ip']
        dest_scp_port = server[ids][types]['dest_scp_port']
        dest_scp_user = server[ids][types]['dest_scp_user']
        dest_scp_passwd = server[ids][types]['dest_scp_passwd']
    elif ids == '164':
        dest_scp_ip = server[ids][types]['dest_scp_ip']
        dest_scp_port = server[ids][types]['dest_scp_port']
        dest_scp_user = server[ids][types]['dest_scp_user']
        dest_scp_passwd = server[ids][types]['dest_scp_passwd']
    elif ids == '165':
        dest_scp_ip = server[ids][types]['dest_scp_ip']
        dest_scp_port = server[ids][types]['dest_scp_port']
        dest_scp_user = server[ids][types]['dest_scp_user']
        dest_scp_passwd = server[ids][types]['dest_scp_passwd']
    elif ids == '241':
        dest_scp_ip = server[ids][types]['dest_scp_ip']
        dest_scp_port = server[ids][types]['dest_scp_port']
        dest_scp_user = server[ids][types]['dest_scp_user']
        dest_scp_passwd = server[ids][types]['dest_scp_passwd']

    dest_ssh = create_ssh_client(server=dest_scp_ip, port=dest_scp_port, user=dest_scp_user, password=dest_scp_passwd)
    dest_scp = scp.SCPClient(dest_ssh.get_transport())
    dest_sftp = dest_ssh.open_sftp()
    return dest_scp, dest_sftp, dest_ssh
