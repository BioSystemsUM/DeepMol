class PreConditionViolationException(Exception):

    def __init__(self, condition=None, message="Pre-condition violation"):
        if condition:
            message = message + ": %s" % condition

        self.message = message
        super().__init__(self.message)
