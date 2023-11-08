import sys 
import os
from typing import List
from paramiko import AutoAddPolicy, SSHClient
from paramiko.auth_handler import AuthenticationException, SSHException
from scp import SCPClient, SCPException

#this file is expected to be modifed once for every single chromebook in our BCI lab
class fileTransfer:
    def __init__(self, host: str, username: str, private_key: str):
        self.host = host  # change
        self.username = username  # change
        self.private_key = private_key  # change
        self.server_conn = self.connect

    @property
    def connection(self):
        """Open SSH connection to remote host."""
        try:
            client = SSHClient()
            client.set_missing_host_key_policy(AutoAddPolicy())
            client.connect(
                hostname=self.host,
                username=self.username,
                key_filename=self.private_key
            )
            return client
        except AuthenticationException as e:
            print(
                f"AuthenticationException occurred; make sure you are providing your private key? {e}"
            )
        except Exception as e:
            print(f"Unexpected error occurred while connecting to host: {e}")
    
    def progress(self, filename, size, sent):
        sys.stdout.write("{}'s progress: {:.2f}%   \r".format(filename, float(sent)/float(size)*100)) 
    
    @property
    def connect(self) -> SCPClient:
        conn = self.connection
        return SCPClient(conn.get_transport(), progress=self.progress)
    
    def bulk_upload(self, local_path, remote_path: str):
        """
        Upload multiple files to a remote directory.

        :param List[str] filepaths: List of local files to be uploaded.
        """
        try:
            self.connect.put(local_path, remote_path=remote_path, recursive=True)
            print(
                f"Finished uploading files from {local_path} to {remote_path} on {self.host}"
            )
        except SCPException as e:
            print(f"SCPException during bulk upload: {e}")
        except Exception as e:
            print(f"Unexpected exception during bulk upload: {e}")

    def fetch_local_files(self, local_file_dir: str) -> List[str]:
        """
        Generate list of file paths.
        :param str local_file_dir: Local filepath of assets to SCP to host.
        :returns: List[str]
        """
        local_files = os.walk(local_file_dir)
        files = []
        for root, _ , filenames in local_files:
            for filename in filenames:
                files.append(os.path.join(root, filename))

        print(f"Files to be transferred: ", files)
        return files 

    def disconnect(self):
        """Close SSH & SCP connection."""
        if self.connect:
            self.connect.close()
        if self.server_conn:
            self.server_conn.close()

def main():
    pass
    # svrcon = fileTransfer()
    # src = sys.argv[1]
    # target = sys.argv[2]
    # svrcon.transfer(str(src), str(target))


if __name__ == '__main__':
    main()
