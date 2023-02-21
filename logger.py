import neptune.new as neptune
import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self, id=None):
        # self.args = args

        self.project = ""
        self.name = ""
        self.tags = []
        self.api_token = ""

        assert self.api_token != "", "Please specify your neptune api token"

        if id is not None:
            self.neptune_run = neptune.init(
                with_id=id,
                project=self.project,
                name=self.name,
                tags=self.tags,
                api_token=self.api_token,
            )
        else:
            self.neptune_run = neptune.init(
                project=self.project,
                name=self.name,
                tags=self.tags,
                api_token=self.api_token,
            )

    def get_id(self):
        run_id = self.neptune_run["sys/id"].fetch()
        return run_id

    def log(self, kwargs):
        for k, v in kwargs.items():
            self.neptune_run[k].log(v)

    def log_with_step(self, step, kwargs):
        for k, v in kwargs.items():
            self.neptune_run[k].log(
                value=v,
                step=step,
            )

            # del self.neptune_run[k]

    def log_image(self, kwargs):
        for k, v in kwargs.items():
            fig = plt.figure()
            ax  = fig.add_subplot()
            a, b, c, d = v.shape
            ax.imshow(v.transpose((0, 2, 3, 1)).reshape((a*c, d, b)))
            ax.axis('off')
            self.neptune_run[k].log(fig)
            plt.close(fig)

    def close(self):
        self.neptune_run.stop()
