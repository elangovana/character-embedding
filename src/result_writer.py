"""
Writes results
"""
import datetime
import json
import logging
import os
import uuid


class ResultWriter:

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def dump_object(self, object, output_dir, filename_prefix):
        """
Dumps the object as a json to a file
        :param object:
        """
        filename = os.path.join(output_dir,
                                "{}_Objectdump_{}_{}.json".format(filename_prefix,
                                                                  datetime.datetime.strftime(datetime.datetime.now(),
                                                                                             format="%Y%m%d_%H%M%S"),
                                                                  str(uuid.uuid4())))

        with open(filename, "w") as o:
            json.dump(object, o)
