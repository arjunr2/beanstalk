"""Beanstalk scripts."""

from . import violations, dataset, summarize, simulate, simulate2

commands = {
    "violations": violations,
    "dataset": dataset,
    "summarize": summarize,
    "simulate": simulate,
    "simulate2": simulate2
}
