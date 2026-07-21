"""Beanstalk scripts."""

from . import violations, dataset, delay_dataset, summarize, simulate, simulate2

commands = {
    "violations": violations,
    "dataset": dataset,
    "delay_dataset": delay_dataset,
    "summarize": summarize,
    "simulate": simulate,
    "simulate2": simulate2
}
