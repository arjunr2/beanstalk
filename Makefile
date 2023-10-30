#
#    ___   ____   __    _      __  _____   __    _     _
#   | |_) | |_   / /\  | |\ | ( (`  | |   / /\  | |   | |_/
#   |_|_) |_|__ /_/--\ |_| \| _)_)  |_|  /_/--\ |_|__ |_| \ 
#   ==== Data Race Detection for WebAssembly ==============
#


DATASETS=beanstalk baseline
DATASETS_RAW=$(addprefix data-raw/,$(DATASETS))
DATASETS_OUT=$(addprefix )

data:
	mkdir -p data

data/violations.json: data $(DATASETS_RAW)
	python manage.py violations -p $(DATASETS_RAW) -o $@

data/%: data-raw/%
	python manage.py dataset -p $< -v data/violations.json -o $@


BENCHMARKS=$(shell ls data/beanstalk)

summary:
	mkdir -p summary

summary/%: data/beanstalk/%
	python manage.py summarize -p data/beanstalk/$* -o summary/$*

simulations:
	mkdir -p simulations

simulations/density.npz: data/beanstalk
	python manage.py simulate2 -p $< -o simulations/density.npz

simulations/%: data/%
	python manage.py simulate -p data/$* -o simulations/$*.npz

FIGURES=$(shell ls plot)

figures/%.pdf: plot/%.py
	python plot/$*.py
