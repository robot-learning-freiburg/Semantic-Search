import subprocess

import home_robot

PKG_ROOT_PATH = home_robot.__path__[0]
REPO_ROOT_PATH = (
    subprocess.run(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .stdout.strip()
    .decode("ascii")
)
