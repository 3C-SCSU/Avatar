import pysftp
import sys

#this file is expected to be modifed once for every single chromebook in our BCI lab
class fileTransfer:
    def __init__(self, host, username, private_key, private_key_pass, port=22):
        self.host = host
        self.username = username
        self.private_key = private_key
        self.private_key_pass = private_key_pass
        self.port = port
        self.serverconn = self.connect()

    def connect(self):
        """Connects to the sftp server and returns the sftp connection object"""
        try:
            # Get the sftp connection object
            serverconn = pysftp.Connection(
                host=self.host,
                username=self.username,
                private_key=self.private_key,  # make secret
                private_key_pass=self.private_key_pass,  # make secret
                port=self.port
            )
            if (serverconn):
                print("Connected to host...")
        except Exception as err:
            print(err)
            raise Exception(err)

        finally:
            return serverconn

    def transfer(self, src, target):
        """Recursivily places files in the target dir, copies everything inside of src dir"""
        try:
            print(f"Transfering files to {self.host} ...")
            self.serverconn.put_r(str(src), str(target))
            print("Files Successfully Transfered!")
            print(
                f"Src files placed in Dir: {self.serverconn.listdir(target)}")

        except Exception as err:
            raise Exception(err)


def main():
    svrcon = fileTransfer()
    src = sys.argv[1]
    target = sys.argv[2]
    svrcon.transfer(str(src), (target))


if __name__ == '__main__':
    main()
