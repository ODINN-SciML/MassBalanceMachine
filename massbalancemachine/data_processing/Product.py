"""
The Product class is responsible for tracking processing results and updating them when required based on their dependencies.

Date Created: 06/01/2026
"""

import os
import git, json, datetime


class Product:
    def __init__(self, file_path, commit_dependent=False):
        self.file_path = os.path.abspath(file_path)
        self.commit_dependent = commit_dependent
        if self.commit_dependent:
            repo = git.Repo(search_parent_directories=True)
            self.commit_hash = repo.head.object.hexsha
        else:
            self.commit_hash = None
        d = os.path.dirname(self.file_path)
        if not os.path.isdir(d):
            os.makedirs(d)

    def is_up_to_date(self):
        up_to_date = False
        if os.path.isfile(self.file_path + ".chk"):
            with open(self.file_path + ".chk", "r") as f:
                d = json.load(f)
                if self.commit_dependent:
                    up_to_date = d.get("commit_hash") == self.commit_hash
                else:
                    up_to_date = True
        if not os.path.isfile(self.file_path):
            up_to_date = False
        return up_to_date

    def gen_chk(self):
        info = {
            "commit_hash": self.commit_hash,
            "date": datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            ),
        }
        with open(self.file_path + ".chk", "w") as f:
            json.dump(info, f, indent=4, sort_keys=True)
