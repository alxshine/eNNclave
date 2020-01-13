SIZE_OPTIONS=-resize 700

all: graphic_cpu graphic_gpu

graphic_cpu: preamble.tex graphic_cpu.tex
	latexmk	graphic_cpu.tex

graphic_gpu: preamble.tex graphic_gpu.tex
	latexmk graphic_gpu.tex

.PHONY: convert
convert: all
	convert -density 300 _build/graphic_gpu.pdf ${SIZE_OPTIONS} -quality 90 images/graphic_gpu.png
	convert -density 300 _build/graphic_cpu.pdf ${SIZE_OPTIONS} -quality 90 images/graphic_cpu.png
