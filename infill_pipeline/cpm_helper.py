import subprocess

class CPMHelper:
    """ Helper class to run EPSC executables
    """

    def __init__(self):
        pass

    def execute_exec(self, working_dir):
        p = subprocess.Popen(['./a.out'], cwd=working_dir,
                             stdin=None, stdout=None, stderr=None)
        # p.wait()
