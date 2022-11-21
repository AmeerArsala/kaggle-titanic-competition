import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi


class RemoteCommunicator:
    def __init__(self):
        super().__init__()

        # authenticate
        self.api = KaggleApi()
        self.api.authenticate()

    def download_data(self):
        COMPETITION = "titanic"

        self.api.competition_download_file(COMPETITION, "train.csv", path="../data", force=True)
        self.api.competition_download_file(COMPETITION, "test.csv", path="../data", force=True)
        self.api.competition_download_file(COMPETITION, "gender_submission.csv", path="../data", force=True)
