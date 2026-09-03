"""pipelie -- find the bugs that leave a pipeline reporting green.

    import pipelie
    print(pipelie.audit(df, target="built", key=["id"]))

Every check corresponds to a bug that shipped, in production, while the tests
passed. The write-up is at
https://savabs.github.io/2026/08/29/thirteen-ways-a-pipeline-lies.html
"""
from .api import accept, audit, audit_file, guard, snapshot
from .profile import drift, load_profile, profile, save_profile
from .finding import CRITICAL, INFO, WARNING, Finding, PipelineLied, Report

__version__ = "0.4.0"
__all__ = ["audit", "audit_file", "guard", "accept", "snapshot", "drift",
           "profile", "save_profile", "load_profile", "Report", "Finding", "PipelineLied",
           "CRITICAL", "WARNING", "INFO", "__version__"]
