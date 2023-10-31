#
#    ___   ____   __    _      __  _____   __    _     _
#   | |_) | |_   / /\  | |\ | ( (`  | |   / /\  | |   | |_/
#   |_|_) |_|__ /_/--\ |_| \| _)_)  |_|  /_/--\ |_|__ |_| \ 
#   ==== Data Race Detection for WebAssembly ==============
#

# -- Dataset ------------------------------------------------------------------

DATASETS=beanstalk baseline
DATASETS_RAW=$(addprefix data-raw/,$(DATASETS))
DATASETS_OUT=$(addprefix )

data:
	mkdir -p data

data/violations.json: data $(DATASETS_RAW)
	python manage.py violations -p $(DATASETS_RAW) -o $@

data/%: data-raw/%
	python manage.py dataset -p $< -v data/violations.json -o $@


# -- Summary Statistics -------------------------------------------------------

BENCHMARKS=$(shell ls data/beanstalk)

summary:
	mkdir -p summary

summary/%: data/beanstalk/%
	python manage.py summarize -p data/beanstalk/$* -o summary/$*


# -- Simulations --------------------------------------------------------------

.phony: all-simulations
all-simulations: simulations \
	simulations/abl_density.npz simulations/abl_device.npz \
	simulations/density.npz simulations/beanstalk.npz simulations/baseline.npz

simulations:
	mkdir -p simulations

simulations/density.npz: data/beanstalk
	python manage.py simulate2 -p $< -o simulations/density.npz

simulations/abl_device.npz: data/beanstalk
	python manage.py simulate -p $< -o simulations/abl_device.npz \
		--ablation device

simulations/abl_density.npz: data/beanstalk
	python manage.py simulate -p $< -o simulations/abl_density.npz \
		--ablation density

simulations/beanstalk.npz: data/beanstalk
	python manage.py simulate -p data/$* -o simulations/beanstalk.npz

simulations/baseline.npz: data/baseline
	python manage.py simulate -p data/$* -o simulations/baseline.npz


# -- Figures ------------------------------------------------------------------

FIGURES=$(shell ls plot)

figures/%.pdf: plot/%.py
	python plot/$*.py
