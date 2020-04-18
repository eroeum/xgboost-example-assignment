"""
Preprocessing

Does nothing but added for nomanclature
"""

class Preprocesser(object):
    def __init__(self, data):
        self.data = data

    def run(self):
        self.data = self.data

    def get_data(self):
        return self.data
