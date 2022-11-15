class PreConditionViolationException(Exception):
    """
    Exception raised when a pre-condition is violated.
    """

    def __init__(self, condition: str = None, message: str = "Pre-condition violation"):
        """
        Initialize the exception with a message and a condition.

        condition: str
            The condition that was violated.
        message: str
            The message to show.
        """
        if condition:
            message = message + ": %s" % condition

        self.message = message
        super().__init__(self.message)
