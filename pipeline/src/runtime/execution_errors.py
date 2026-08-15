"""Errors shared by durable provider execution boundaries."""


class SubmissionUncertainError(RuntimeError):
    """A provider may have accepted a request whose job ID is unavailable."""


class ProviderEndpointChangedError(RuntimeError):
    """A persisted job cannot resume through a different provider endpoint."""


class ProviderPreparationError(RuntimeError):
    """Local provider inputs failed before any paid submission was attempted."""


class ProviderJobFailedError(RuntimeError):
    """The provider explicitly reported a terminal failed job state."""
