"""Beanstalk scripts."""

from . import violations, dataset, summarize

commands = {
    "violations": violations,
    "dataset": dataset,
    "summarize": summarize
}
